from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker
from config.settings import settings
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from retriever.builder import score_documents
import logging

logger = logging.getLogger(__name__)

# Caps the research<->verify loop so a stubborn "NO" from the verifier can't spin
# forever - each retry is 2 more LLM calls, which matters on a rate-limited free tier.
MAX_RESEARCH_ATTEMPTS = 2

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    research_attempts: int
    # A speculative first-draft research call started concurrently with the relevance
    # check (see _check_relevance_step). The research node consumes this instead of
    # making a fresh call when it's present. None on re-research and when disabled.
    research_future: Optional[Future]

class AgentWorkflow:
    def __init__(self, client, model):
        # One OpenAI client (built from the user's own key) shared by all three agents.
        self.researcher = ResearchAgent(client, model)
        self.verifier = VerificationAgent(client, model)
        self.relevance_checker = RelevanceChecker(client, model)
        # Shared pool for overlapping independent I/O-bound work (the speculative
        # research draft and the citation scoring) with the main call chain. These
        # are network/local-retrieval waits, so threads - not processes - are the
        # right tool; there's no CPU-bound work here to parallelize.
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="verishelf")
        self.compiled_workflow = self.build_workflow()  # Compile once during initialization
        
    def build_workflow(self):
        """Create and compile the multi-agent workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)
        
        # Define edges
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )
        return workflow.compile()
    
    def _check_relevance_step(self, state: AgentState) -> Dict:
        # Kick off the first research draft concurrently with the relevance check.
        # Research only needs the retrieved passages, not the verdict, so overlapping
        # the two removes the relevance call's latency from the critical path on the
        # common in-scope path. If the question turns out to be out of scope, we simply
        # never read this future's result (and cancel it if it hasn't started).
        research_future = None
        if settings.SPECULATIVE_RESEARCH:
            research_future = self._executor.submit(
                self.researcher.generate, state["question"], state["documents"]
            )

        # Only the top few passages are needed to judge whether the question is in
        # scope - sending all 20 balloons the prompt's input tokens (which count against
        # per-minute token limits, especially Groq's), for no better scope decision.
        classification = self.relevance_checker.check(
            question=state["question"],
            documents=state["documents"],
            k=6
        )

        # CAN_ANSWER and PARTIAL both proceed; only NO_MATCH stops here.
        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True, "research_future": research_future}

        # NO_MATCH: discard the speculative draft (cancel if it never started).
        if research_future is not None:
            research_future.cancel()
        return {
            "is_relevant": False,
            "research_future": None,
            "draft_answer": "This question isn't related (or there's no data) for your query. Please ask another question relevant to the uploaded document(s)."
        }


    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        print(f"[DEBUG] _decide_after_relevance_check -> {decision}")
        return decision
    
    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        try:
            print(f"[DEBUG] Starting full_pipeline with question='{question}'")
            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents (from .invoke)")

            # Citation scores depend only on the question, not on the answer, so compute
            # them in the background while the LLM call chain runs and collect them at
            # the end - overlapping local retrieval-scoring with the network waits.
            scores_future = self._executor.submit(score_documents, retriever, question)

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
                research_attempts=0,
                research_future=None,
            )

            final_state = self.compiled_workflow.invoke(initial_state)

            # Citations: the same passages passed to the research/verification agents,
            # with a real (not fabricated) relevance score recovered from the hybrid
            # retriever's own ranking - see retriever.builder.score_documents. Skipped
            # for out-of-scope questions, since no answer was actually drafted from them.
            citations = []
            if final_state["verification_report"]:
                scores = scores_future.result()
                citations = sorted(
                    (
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "excerpt": doc.page_content[:220].strip(),
                            "score": scores.get(doc.page_content, 0.0),
                        }
                        for doc in documents
                    ),
                    key=lambda c: c["score"],
                    reverse=True,
                )[:5]
            else:
                # Out of scope: no citations needed, so don't wait on the scoring job.
                scores_future.cancel()

            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"],
                "citations": citations,
                "passages_consulted": len(documents),
                "re_researched": final_state.get("research_attempts", 0) > 1,
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    def _research_step(self, state: AgentState) -> Dict:
        print(f"[DEBUG] Entered _research_step with question='{state['question']}'")
        # On the first pass, consume the draft that was started speculatively during
        # the relevance check (likely already running or finished). On re-research the
        # future is gone, so make a fresh call from the (unchanged) documents.
        research_future = state.get("research_future")
        if research_future is not None:
            print("[DEBUG] Consuming speculative research draft.")
            result = research_future.result()
        else:
            result = self.researcher.generate(state["question"], state["documents"])
        print("[DEBUG] Researcher returned draft answer.")
        return {
            "draft_answer": result["draft_answer"],
            "research_attempts": state.get("research_attempts", 0) + 1,
            "research_future": None,  # consumed; a re-research pass must call fresh
        }
    
    def _verification_step(self, state: AgentState) -> Dict:
        print("[DEBUG] Entered _verification_step. Verifying the draft answer...")
        result = self.verifier.check(state["draft_answer"], state["documents"])
        print("[DEBUG] VerificationAgent returned a verification report.")
        return {"verification_report": result["verification_report"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        verification_report = state["verification_report"]
        print(f"[DEBUG] _decide_next_step with verification_report='{verification_report}'")
        needs_retry = "Supported: NO" in verification_report or "Relevant: NO" in verification_report
        if needs_retry and state.get("research_attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
            logger.info(f"[DEBUG] Hit MAX_RESEARCH_ATTEMPTS ({MAX_RESEARCH_ATTEMPTS}); ending workflow anyway.")
            return "end"
        elif needs_retry:
            logger.info("[DEBUG] Verification indicates re-research needed.")
            return "re_research"
        else:
            logger.info("[DEBUG] Verification successful, ending workflow.")
            return "end"
