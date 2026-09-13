import os
from sentence_transformers import SentenceTransformer
import oracledb
import ollama


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

def search_similar_chunks(question, top_k=3, max_distance=0.50):

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert user question into a 384-dimensional vector
    query_embedding = model.encode(question).tolist()

  
    # Connect to Oracle
    print("Connecting to Oracle...")

    conn = oracledb.connect(
        user=os.environ["ORACLE_DB_USER"],
        password=os.environ["ORACLE_DB_PASSWORD"],
        dsn=os.environ["ORACLE_DB_DSN"]
    )

    cursor = conn.cursor()
    # Convert Python list to Oracle VECTOR string
    vector_string = "[" + ",".join(map(str, query_embedding)) + "]"

    # ======================================================
    # VECTOR SEARCH
    # ======================================================

    sql = """
        SELECT source_file,
               content,
               distance
        FROM (
            SELECT source_file,
                   content,
                   VECTOR_DISTANCE(
                       embedding,
                       :query_vector,
                       COSINE
                   ) AS distance
            FROM pdf_chunks
            ORDER BY distance
        )
        WHERE distance <= :max_distance
        FETCH FIRST :top_k ROWS ONLY
    """

    cursor.execute(
        sql,
        query_vector=vector_string,
        max_distance=max_distance,
        top_k=top_k
    )

    results = cursor.fetchall()

    print("\n===== SEARCH RESULTS =====")

    # Store retrieved text for LLM
    contexts = []

    if not results:

        print("No sufficiently relevant results found.")

    else:

        for i, row in enumerate(results, 1):

            source_file = row[0]
            content = row[1]
            distance = row[2]

            # ------------------------------------------------
            # IMPORTANT:
            # Oracle CLOB -> Python string
            # ------------------------------------------------

            if hasattr(content, "read"):
                content_text = content.read()
            else:
                content_text = str(content)

            print(f"\nResult {i}")
            print(f"Source: {source_file}")
            print(f"Distance: {distance}")
            print(f"Content:\n{content_text}")

            # Store text for LLM
            contexts.append(content_text)

    cursor.close()
    conn.close()

    # Return retrieved chunks
    return contexts


# ==========================================================
# LOCAL LLM GENERATION
# ==========================================================

def generate_answer(question, contexts):

    # Combine all retrieved chunks into one context
    context = "\n\n".join(contexts)

    prompt = f"""
You are a helpful document-based question answering assistant.

Answer the user's question using ONLY the information provided
in the context below.

IMPORTANT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not make up information.
4. If the answer cannot be found in the context, say:
   "I could not find this information in the provided documents."
5. Give a clear and concise answer.

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
"""

    print("\n===== GENERATING ANSWER =====")

    # Send context + question to local Ollama model
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ==========================================================
# MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    question = input("\nEnter your question: ")

    # ------------------------------------------------------
    # STEP 1: Semantic search
    # ------------------------------------------------------

    contexts = search_similar_chunks(question)

    # ------------------------------------------------------
    # STEP 2: Send retrieved context to LLM
    # ------------------------------------------------------

    if contexts:

        answer = generate_answer(
            question,
            contexts
        )

        print("\n===== FINAL ANSWER =====")
        print(answer)

    else:

        print("\nNo relevant information found.")
