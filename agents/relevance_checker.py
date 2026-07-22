from config.settings import settings
from .llm_client import get_client
import logging

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(self):
        self.client = get_client()
        self.model = settings.RELEVANCE_MODEL
        logger.info(f"RelevanceChecker using model '{self.model}' via NVIDIA NIM.")

    def check(self, question: str, documents, k=3) -> str:
        """
        1. Take the top-k already-retrieved document chunks (retrieval happened once,
           up front, in workflow.py - re-invoking the retriever here would just repeat
           the same BM25 + vector search for no benefit).
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        if not documents:
            logger.debug("No documents retrieved. Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in documents[:k])

        # Create a prompt for the LLM to classify relevance
        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        # Call the LLM. Note: API/network/auth errors are intentionally NOT caught here -
        # they must propagate up so the UI can tell "the model said no" apart from
        # "the model call failed" instead of misreporting a broken key as an
        # out-of-scope question.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        # Extract the content from the response
        try:
            llm_response = response.choices[0].message.content.strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except (IndexError, AttributeError, TypeError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"

        # Validate the response
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            # Some models wrap the label in extra text despite instructions - salvage it if possible.
            matched = next((label for label in valid_labels if label in llm_response), None)
            classification = matched or "NO_MATCH"
            logger.debug(f"LLM did not respond with a bare valid label. Resolved to '{classification}'.")
        else:
            classification = llm_response

        return classification
