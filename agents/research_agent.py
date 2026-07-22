from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings
from .llm_client import get_client
import logging

logger = logging.getLogger(__name__)

# Chunks from MarkdownHeaderTextSplitter aren't size-capped (a "chunk" can be a whole
# page), so an uncapped context can balloon the prompt and slow down free-tier model
# calls a lot. This keeps the context generous but bounded.
MAX_CONTEXT_CHARS = 6000


class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with an NVIDIA NIM-backed chat model.
        """
        self.client = get_client()
        self.model = settings.RESEARCH_MODEL
        logger.info(f"ResearchAgent using model '{self.model}' via NVIDIA NIM.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        logger.debug(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Combine the top document contents into one string, capped to keep the
        # prompt (and therefore the free-tier model's response time) reasonable.
        context = "\n\n".join([doc.page_content for doc in documents])[:MAX_CONTEXT_CHARS]

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)

        # Call the LLM to generate the answer. API/network/rate-limit errors are left
        # to propagate so the UI can tell them apart (e.g. a specific rate-limit message)
        # instead of collapsing everything into one generic failure.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        # Extract and process the LLM's response
        try:
            llm_response = response.choices[0].message.content.strip()
        except (IndexError, AttributeError, TypeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        logger.debug(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
