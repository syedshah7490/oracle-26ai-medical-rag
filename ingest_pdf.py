import os
import glob
import array
import subprocess

import oracledb
import ollama
import pytesseract

from PIL import Image
from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FOLDER = "data"

DB_USER = os.environ["ORACLE_DB_USER"]
DB_PASSWORD = os.environ["ORACLE_DB_PASSWORD"]
DB_DSN = os.environ["ORACLE_DB_DSN"]
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


# ============================================================
# 2. FIND PDF FILES
# ============================================================

pdf_files = glob.glob(
    os.path.join(DATA_FOLDER, "*.pdf")
)

print("\n========================================")
print("       PDF INGESTION WITH METADATA")
print("========================================")

print(f"\nFound {len(pdf_files)} PDF file(s):")

for pdf in pdf_files:
    print(f" - {os.path.basename(pdf)}")

if not pdf_files:
    print("\nNo PDF files found in the data folder.")
    exit(1)


# ============================================================
# 3. CONNECT TO ORACLE
# ============================================================

print("\nConnecting to Oracle...")

connection = oracledb.connect(
    user=DB_USER,
    password=DB_PASSWORD,
    dsn=DB_DSN
)

cursor = connection.cursor()

print("Oracle connection successful.")


# ============================================================
# 4. REMOVE OLD DATA
# ============================================================

print("\nRemoving old data from rag_test...")

cursor.execute(
    "DELETE FROM rag_user.rag_test"
)

connection.commit()

print("Old data removed successfully.")


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\n", " ")

    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


# ============================================================
# 6. CREATE CHUNKS
# ============================================================

def create_chunks(text):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# 7. PROCESS PDF FILES
# ============================================================

total_chunks_inserted = 0

for pdf_path in pdf_files:

    file_name = os.path.basename(pdf_path)

    print("\n========================================")
    print(f"PROCESSING: {file_name}")
    print("========================================")

    reader = PdfReader(pdf_path)

    total_pages = len(reader.pages)

    print(f"Total pages: {total_pages}")


    # ========================================================
    # 8. PROCESS EACH PAGE
    # ========================================================

    for page_index, page in enumerate(reader.pages):

        page_number = page_index + 1

        print(
            f"\nProcessing page "
            f"{page_number}/{total_pages}..."
        )


        # ====================================================
        # 9. NORMAL PDF TEXT EXTRACTION
        # ====================================================

        text = page.extract_text() or ""

        text = clean_text(text)


        # ====================================================
        # 10. OCR FALLBACK
        # ====================================================

        if len(text) < 50:

            print(
                f"Page {page_number}: "
                f"Not enough text found."
            )

            print("Using OCR...")

            output_prefix = (
                f"/tmp/{file_name}_page_{page_number}"
            )

            try:

                subprocess.run(
                    [
                        "pdftoppm",
                        "-f", str(page_number),
                        "-singlefile",
                        "-png",
                        "-r", "200",
                        pdf_path,
                        output_prefix
                    ],
                    check=True
                )

                image_path = output_prefix + ".png"

                image = Image.open(image_path)

                text = pytesseract.image_to_string(
                    image
                )

                text = clean_text(text)

            finally:

                image_path = output_prefix + ".png"

                if os.path.exists(image_path):
                    os.remove(image_path)


        # ====================================================
        # 11. SKIP EMPTY PAGES
        # ====================================================

        if not text:

            print(
                f"WARNING: Page {page_number} "
                f"contains no extractable text."
            )

            continue


        print(
            f"Extracted characters: "
            f"{len(text):,}"
        )


        # ====================================================
        # 12. CREATE CHUNKS FOR THIS PAGE
        # ====================================================

        chunks = create_chunks(text)

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )


        # ====================================================
        # 13. CREATE EMBEDDINGS
        # ====================================================

        for chunk_index, chunk in enumerate(
            chunks,
            start=1
        ):

            print(
                f"\nCreating embedding "
                f"{chunk_index}/{len(chunks)}..."
            )

            embedding_response = ollama.embed(
                model=EMBEDDING_MODEL,
                input=chunk
            )

            embedding = (
                embedding_response["embeddings"][0]
            )

            print(
                f"Embedding dimensions: "
                f"{len(embedding)}"
            )


            # ================================================
            # 14. CHECK VECTOR DIMENSIONS
            # ================================================

            if len(embedding) != 768:

                raise ValueError(
                    f"Expected 768 dimensions, "
                    f"but received {len(embedding)}."
                )


            # ================================================
            # 15. CONVERT TO ORACLE VECTOR
            # ================================================

            vector = array.array(
                "f",
                embedding
            )


            # ================================================
            # 16. INSERT CHUNK + METADATA
            # ================================================

            cursor.execute(
                """
                INSERT INTO rag_user.rag_test
                (
                    text_data,
                    embedding,
                    source_file,
                    page_number
                )
                VALUES
                (
                    :text_data,
                    :embedding,
                    :source_file,
                    :page_number
                )
                """,
                {
                    "text_data": chunk,
                    "embedding": vector,
                    "source_file": file_name,
                    "page_number": page_number
                }
            )

            total_chunks_inserted += 1

            print(
                f"Inserted chunk "
                f"{total_chunks_inserted}"
            )

        # Commit after each page
        connection.commit()


# ============================================================
# 17. FINAL VERIFICATION
# ============================================================

print("\n========================================")
print("       PDF INGESTION COMPLETED")
print("========================================")

print(
    f"Total chunks inserted: "
    f"{total_chunks_inserted}"
)


# Count rows
cursor.execute(
    "SELECT COUNT(*) FROM rag_user.rag_test"
)

row_count = cursor.fetchone()[0]

print(
    f"Rows currently in rag_test: "
    f"{row_count}"
)


# Check vector dimensions
cursor.execute(
    """
    SELECT VECTOR_DIMENSION_COUNT(embedding)
    FROM rag_user.rag_test
    FETCH FIRST 1 ROW ONLY
    """
)

result = cursor.fetchone()

if result:
    print(
        f"Stored vector dimensions: "
        f"{result[0]}"
    )


# ============================================================
# 18. SHOW METADATA VERIFICATION
# ============================================================

print("\n========================================")
print("       METADATA VERIFICATION")
print("========================================")

cursor.execute(
    """
    SELECT
        source_file,
        page_number,
        SUBSTR(text_data, 1, 80)
    FROM rag_user.rag_test
    FETCH FIRST 10 ROWS ONLY
    """
)

metadata_rows = cursor.fetchall()

for row in metadata_rows:

    print(
        f"\nSource: {row[0]}"
    )

    print(
        f"Page: {row[1]}"
    )

    print(
        f"Text: {row[2]}"
    )


# ============================================================
# 19. CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()

print("\nOracle connection closed.")

print("\n========================================")
print("              SUCCESS")
print("========================================")
