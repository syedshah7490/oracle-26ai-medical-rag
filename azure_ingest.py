import glob
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytesseract
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

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
EMBEDDING_DIMENSIONS = 768

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

credential = AzureKeyCredential(SEARCH_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# CREATE AZURE AI SEARCH INDEX
# ============================================================

def create_index():
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=credential,
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="medical-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="medical-hnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="medical-vector-profile",
                algorithm_configuration_name="medical-hnsw",
            )
        ],
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_or_update_index(index)

    print(f"Azure AI Search index ready: {INDEX_NAME}")


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# PDF TEXT EXTRACTION WITH OCR FALLBACK
# ============================================================

def extract_page_text(pdf_path, page_number, page):
    text = page.extract_text() or ""

    if text.strip():
        return text

    print(f"  OCR page {page_number}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_prefix = str(
            Path(temp_dir) / "page"
        )

        command = [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            "200",
            "-jpeg",
            pdf_path,
            output_prefix,
        ]

        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if process.returncode != 0:
            print(
                f"  Warning: PDF conversion failed "
                f"for page {page_number}"
            )
            return ""

        image_files = sorted(
            glob.glob(
                str(Path(temp_dir) / "page-*.jpg")
            )
        )

        if not image_files:
            print(
                f"  Warning: No image generated "
                f"for page {page_number}"
            )
            return ""

        extracted_parts = []

        for image_file in image_files:
            try:
                with Image.open(image_file) as image:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang="eng",
                    )

                    if page_text.strip():
                        extracted_parts.append(page_text)

            except Exception as error:
                print(
                    f"  Warning: OCR failed on "
                    f"page {page_number}: {error}"
                )

        return "\n".join(extracted_parts)


def extract_documents():
    documents = []

    pdf_files = sorted(
        glob.glob("data/*.pdf")
    )

    if not pdf_files:
        raise RuntimeError(
            "No PDF files were found inside data/."
        )

    for pdf_path in pdf_files:
        source = Path(pdf_path).name
        title = (
            Path(pdf_path)
            .stem
            .replace("_", " ")
            .title()
        )

        reader = PdfReader(pdf_path)

        print(
            f"Extracting: {source} "
            f"({len(reader.pages)} pages)"
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = extract_page_text(
                pdf_path,
                page_number,
                page,
            )

            if not text.strip():
                print(
                    f"  Warning: No text found "
                    f"on page {page_number}"
                )
                continue

            page_chunks = split_text(text)

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1,
            ):
                raw_id = (
                    f"{source}-"
                    f"{page_number}-"
                    f"{chunk_number}"
                )

                document_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()

                documents.append(
                    {
                        "id": document_id,
                        "content": chunk,
                        "title": title,
                        "source": source,
                        "page": page_number,
                    }
                )

    return documents


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def create_embeddings(documents, batch_size=20):
    for start in range(
        0,
        len(documents),
        batch_size,
    ):
        batch = documents[
            start:start + batch_size
        ]

        prepared_contents = [
            (
                f"title: {item['title']} | "
                f"text: {item['content']}"
            )
            for item in batch
        ]

        response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=prepared_contents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                "Gemini returned an unexpected "
                "embedding count."
            )

        for item, embedding in zip(
            batch,
            response.embeddings,
        ):
            item["embedding"] = embedding.values

        completed = min(
            start + batch_size,
            len(documents),
        )

        print(
            f"Embedded {completed}/"
            f"{len(documents)} chunks"
        )

        time.sleep(1)

    return documents


# ============================================================
# UPLOAD TO AZURE AI SEARCH
# ============================================================

def upload_documents(documents, batch_size=100):
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential,
    )

    successful = 0

    for start in range(
        0,
        len(documents),
        batch_size,
    ):
        batch = documents[
            start:start + batch_size
        ]

        results = search_client.upload_documents(
            documents=batch
        )

        successful += sum(
            1
            for result in results
            if result.succeeded
        )

        print(
            f"Uploaded {successful}/"
            f"{len(documents)} chunks"
        )

    if successful != len(documents):
        raise RuntimeError(
            f"Only {successful} of "
            f"{len(documents)} chunks uploaded."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    create_index()

    documents = extract_documents()

    if not documents:
        raise RuntimeError(
            "No readable PDF content was found in data/."
        )

    print(
        f"Created {len(documents)} text chunks"
    )

    documents = create_embeddings(documents)

    upload_documents(documents)

    print()
    print(
        "Azure-native ingestion completed successfully."
    )
    print(f"Index: {INDEX_NAME}")
    print(
        f"Documents uploaded: {len(documents)}"
    )


if __name__ == "__main__":
    main()
