import os
import array
import re
import oracledb
import ollama

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_THRESHOLD = 0.50

# Number of chunks retrieved from vector search
VECTOR_TOP_K = 5

# Number of chunks retrieved from keyword search
KEYWORD_TOP_K = 5

# Number of final chunks sent to Gemini
# Increased from 3 to 5 so multi-part questions
# can receive more complete context.
FINAL_TOP_K = 5


# ============================================================
# 1. CHECK GEMINI API KEY
# ============================================================

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY is not set.")
    print()
    print("Run:")
    print("Set GEMINI_API_KEY in your environment before running this program.")
    exit(1)


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ============================================================
# 3. ORACLE DATABASE CONNECTION
# ============================================================

# ============================================================
# 3. ORACLE DATABASE CONNECTION
# ============================================================

print("Connecting to Oracle...")

connection = oracledb.connect(
    user=os.environ["ORACLE_DB_USER"],
    password=os.environ["ORACLE_DB_PASSWORD"],
    dsn=os.environ["ORACLE_DB_DSN"]
)

cursor = connection.cursor()

print("Oracle connection successful.")

# ============================================================
# 4. USER QUESTION
# ============================================================

while True:

    user_query = input("\nEnter your question: ").strip()

    if not user_query:
        print("Please enter a question.")
        continue

    if user_query.lower() in ["exit", "quit"]:
        break


    # ========================================================
    # 5. CREATE QUERY EMBEDDING USING OLLAMA
    # ========================================================

    print("\nConverting query to vector...")

    try:

        embedding_response = ollama.embed(
            model="nomic-embed-text",
            input=user_query
        )

        query_embedding = embedding_response["embeddings"][0]

    except Exception as e:

        print("\nOllama embedding error:")
        print(e)
        continue


    print(
        "Embedding dimensions:",
        len(query_embedding)
    )


    # ========================================================
    # 6. CHECK EMBEDDING DIMENSIONS
    # ========================================================

    if len(query_embedding) != 768:

        print(
            f"ERROR: Expected 768 dimensions, "
            f"but received {len(query_embedding)}."
        )

        continue


    # ========================================================
    # 7. CONVERT EMBEDDING TO ORACLE VECTOR
    # ========================================================

    query_vector = array.array(
        "f",
        query_embedding
    )


    # ========================================================
    # 8. VECTOR SEARCH
    # ========================================================

    print("\n========== VECTOR SEARCH ==========")

    vector_sql = f"""
        SELECT
            id,
            text_data,
            source_file,
            page_number,
            VECTOR_DISTANCE(
                embedding,
                :query_vector,
                COSINE
            ) AS distance
        FROM rag_user.rag_test
        ORDER BY distance ASC
        FETCH FIRST {VECTOR_TOP_K} ROWS ONLY
    """

    try:

        cursor.execute(
            vector_sql,
            {
                "query_vector": query_vector
            }
        )

        vector_rows = cursor.fetchall()

    except Exception as e:

        print("\nVector search error:")
        print(e)
        continue


    vector_results = []

    for row in vector_rows:

        chunk_id = row[0]

        if hasattr(row[1], "read"):
            text = row[1].read()
        else:
            text = str(row[1])

        source_file = row[2]
        page_number = row[3]
        distance = float(row[4])

        vector_results.append(
            {
                "id": chunk_id,
                "text": text,
                "source": source_file,
                "page": page_number,
                "distance": distance,
                "keyword_score": 0
            }
        )


    # ========================================================
    # 9. DISPLAY VECTOR RESULTS
    # ========================================================

    print("\n===== VECTOR RESULTS =====")

    if not vector_results:

        print("No vector results found.")

    else:

        for result in vector_results:

            print(
                f"Distance: {result['distance']:.4f}"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Chunk: {result['text']}"
            )

            print("-" * 60)


    # ========================================================
    # 10. PREPARE KEYWORDS
    # ========================================================

    keywords = re.findall(
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
        user_query
    )


    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "or",
        "how",
        "why",
        "when",
        "where",
        "can",
        "do",
        "does",
        "tell",
        "me",
        "about"
    }


    keywords = [
        word
        for word in keywords
        if word.lower() not in stop_words
    ]


    # ========================================================
    # 11. KEYWORD SEARCH USING ORACLE TEXT
    # ========================================================

    print("\n========== KEYWORD SEARCH ==========")

    keyword_results = []

    if keywords:

        oracle_text_query = " OR ".join(
            keywords
        )


        keyword_sql = f"""
            SELECT
                id,
                text_data,
                source_file,
                page_number,
                SCORE(1) AS keyword_score
            FROM rag_user.rag_test
            WHERE CONTAINS(
                text_data,
                :keyword_query,
                1
            ) > 0
            ORDER BY SCORE(1) DESC
            FETCH FIRST {KEYWORD_TOP_K} ROWS ONLY
        """


        try:

            cursor.execute(
                keyword_sql,
                {
                    "keyword_query": oracle_text_query
                }
            )

            keyword_rows = cursor.fetchall()

        except Exception as e:

            print("\nKeyword search error:")
            print(e)

            keyword_rows = []


        for row in keyword_rows:

            chunk_id = row[0]

            if hasattr(row[1], "read"):
                text = row[1].read()
            else:
                text = str(row[1])

            source_file = row[2]
            page_number = row[3]
            keyword_score = float(row[4])

            keyword_results.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": source_file,
                    "page": page_number,
                    "distance": None,
                    "keyword_score": keyword_score
                }
            )


    print("\n===== KEYWORD RESULTS =====")

    if not keyword_results:

        print("No keyword results found.")

    else:

        for result in keyword_results:

            print(
                f"Keyword Score: "
                f"{result['keyword_score']:.4f}"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Chunk: {result['text']}"
            )

            print("-" * 60)


    # ========================================================
    # 12. COMBINE VECTOR + KEYWORD RESULTS
    # ========================================================

    print("\n========== HYBRID SEARCH ==========")

    combined = {}


    # --------------------------------------------------------
    # Add vector results
    # --------------------------------------------------------

    for result in vector_results:

        chunk_id = result["id"]

        combined[chunk_id] = result.copy()


    # --------------------------------------------------------
    # Add / merge keyword results
    # --------------------------------------------------------

    for result in keyword_results:

        chunk_id = result["id"]

        if chunk_id in combined:

            combined[chunk_id]["keyword_score"] = (
                result["keyword_score"]
            )

        else:

            combined[chunk_id] = result.copy()


    # ========================================================
    # 13. CALCULATE HYBRID SCORE
    # ========================================================

    combined_results = list(
        combined.values()
    )


    max_keyword_score = max(
        [
            result["keyword_score"]
            for result in combined_results
        ],
        default=0
    )


    for result in combined_results:

        distance = result["distance"]
        keyword_score = result["keyword_score"]


        # --------------------------------------------
        # Vector component
        # --------------------------------------------

        if distance is not None:

            vector_score = max(
                0,
                1 - distance
            )

        else:

            vector_score = 0


        # --------------------------------------------
        # Keyword component
        # --------------------------------------------

        if max_keyword_score > 0:

            normalized_keyword_score = (
                keyword_score /
                max_keyword_score
            )

        else:

            normalized_keyword_score = 0


        # --------------------------------------------
        # Hybrid score
        # --------------------------------------------

        hybrid_score = (
            (0.70 * vector_score)
            +
            (0.30 * normalized_keyword_score)
        )


        result["vector_score"] = vector_score

        result["normalized_keyword_score"] = (
            normalized_keyword_score
        )

        result["hybrid_score"] = hybrid_score


    # ========================================================
    # 14. SORT HYBRID RESULTS
    # ========================================================

    combined_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    # ========================================================
    # 15. APPLY VECTOR RELEVANCE THRESHOLD
    # ========================================================

    best_vector_distance = min(
        [
            result["distance"]
            for result in vector_results
        ],
        default=None
    )


    print(
        f"\nBest vector distance: "
        f"{best_vector_distance}"
    )

    print(
        f"Vector relevance threshold: "
        f"{VECTOR_THRESHOLD}"
    )


    # --------------------------------------------------------
    # If vector search has no relevant result,
    # do NOT send unrelated keyword matches to Gemini.
    # --------------------------------------------------------

    if (
        best_vector_distance is None
        or best_vector_distance > VECTOR_THRESHOLD
    ):

        print("\n========================================")
        print("          INFORMATION NOT FOUND")
        print("========================================")

        print(
            "I could not find this information "
            "in the provided documents."
        )

        print("========================================")

        continue


    # ========================================================
    # 16. FINAL HYBRID RESULTS
    # ========================================================

    final_results = combined_results[
        :FINAL_TOP_K
    ]


    print(
        "\n===== FINAL HYBRID RESULTS ====="
    )


    retrieved_chunks = []

    used_sources = []


    for result in final_results:

        print(
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Vector Score: "
            f"{result['vector_score']:.4f}"
        )

        print(
            f"Keyword Score: "
            f"{result['normalized_keyword_score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Chunk: {result['text']}"
        )

        print("-" * 60)


        retrieved_chunks.append(
            result["text"]
        )


        source_info = (
            result["source"],
            result["page"]
        )


        if source_info not in used_sources:

            used_sources.append(
                source_info
            )


    # ========================================================
    # 17. CHECK CONTEXT
    # ========================================================

    if not retrieved_chunks:

        print(
            "\nNo relevant context was retrieved."
        )

        continue


    # ========================================================
    # 18. BUILD RAG CONTEXT
    # ========================================================

    context = "\n\n".join(
        retrieved_chunks
    )


    # ========================================================
    # 19. BUILD RAG PROMPT
    # ========================================================

    prompt = f"""
You are a helpful medical assistant.

Answer the user's question using ONLY the information
contained in the retrieved context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside medical knowledge.
3. If the answer is not present in the context,
   say exactly:

"I could not find this information in the provided documents."

4. Keep the answer clear and concise.
5. Base the answer directly on the retrieved context.
6. Do not add medical information that is not present
   in the retrieved context.
7. If the question asks for multiple things,
   answer each part separately when the retrieved
   context contains information for that part.

RETRIEVED CONTEXT:

{context}

USER QUESTION:

{user_query}

ANSWER:
"""


    # ========================================================
    # 20. SEND PROMPT TO GEMINI
    # ========================================================

    print(
        "\nGenerating answer using Gemini..."
    )


    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

    except Exception as e:

        print(
            "\n========================================"
        )

        print(
            "          GEMINI API ERROR"
        )

        print(
            "========================================"
        )

        print(e)

        print(
            "========================================"
        )

        continue


    # ========================================================
    # 21. GET GEMINI ANSWER
    # ========================================================

    answer = response.text


    # ========================================================
    # 22. DISPLAY FINAL ANSWER
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "              FINAL ANSWER"
    )

    print(
        "========================================"
    )

    print(answer)

    print(
        "========================================"
    )


    # ========================================================
    # 23. DISPLAY SOURCES
    # ========================================================

    print("\n📚 SOURCES USED:")

    for source, page in used_sources:

        print(
            f"• {source} — Page {page}"
        )


    # ========================================================
    # 24. CONVERSATION MEMORY
    # ========================================================

    print(
        "\nConversation memory: "
        "current session"
    )


# ============================================================
# 25. CLOSE DATABASE
# ============================================================

cursor.close()
connection.close()

print("\nDatabase connection closed.")
