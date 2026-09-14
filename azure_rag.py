import os
import sys

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv(override=True)

SEARCH_ENDPOINT = (
    os.environ["AZURE_SEARCH_ENDPOINT"].strip().rstrip("/")
)
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"].strip()
INDEX_NAME = os.environ.get(
    "AZURE_SEARCH_INDEX",
    "medical-rag-index",
).strip()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = os.environ.get(
    "GEMINI_GENERATION_MODEL",
    "gemini-3.6-flash",
)
EMBEDDING_DIMENSIONS = 768

credential = AzureKeyCredential(SEARCH_KEY)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential,
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


def create_query_embedding(question):
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="QUESTION_ANSWERING",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini did not return a query embedding."
        )

    return response.embeddings[0].values


def search_medical_documents(question, top_k=5):
    query_embedding = create_query_embedding(question)

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=[
            "content",
            "title",
            "source",
            "page",
        ],
        top=top_k,
    )

    documents = []

    for result in results:
        documents.append(
            {
                "content": result["content"],
                "title": result["title"],
                "source": result["source"],
                "page": result["page"],
                "score": result.get("@search.score", 0),
            }
        )

    return documents


def generate_answer(question, documents):
    if not documents:
        return (
            "I could not find relevant information "
            "in the available WHO documents."
        )

    context_parts = []

    for number, document in enumerate(
        documents,
        start=1,
    ):
        context_parts.append(
            f"[Source {number}]\n"
            f"Document: {document['title']}\n"
            f"File: {document['source']}\n"
            f"Page: {document['page']}\n"
            f"Content: {document['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI medical information assistant.

Answer the question using ONLY the supplied WHO document
context. Do not invent facts. If the context does not contain
enough information, clearly say so.

Use source citations such as [Source 1] after relevant claims.
Give a clear, concise and educational answer.

This application provides general educational information and
does not replace professional medical advice, diagnosis, or
treatment.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini did not return an answer."
        )

    return response.text.strip()


def ask_question(question, top_k=5):
    question = question.strip()

    if not question:
        raise ValueError(
            "A question is required."
        )

    documents = search_medical_documents(
        question,
        top_k=top_k,
    )

    answer = generate_answer(
        question,
        documents,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": documents,
    }


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python azure_rag.py '
            '"What causes anaemia?"'
        )
        return

    question = " ".join(sys.argv[1:])
    result = ask_question(question)

    print()
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print()
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:
        print(
            f"- {source['source']}, "
            f"page {source['page']}, "
            f"score {source['score']:.4f}"
        )


if __name__ == "__main__":
    main()
