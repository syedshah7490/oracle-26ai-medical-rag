from pypdf import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    text = extract_text("sample.pdf")
    print("----- FULL EXTRACTED TEXT -----")
    print(text)

    chunks = chunk_text(text)
    print(f"\n----- TOTAL CHUNKS: {len(chunks)} -----\n")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(c)
        print("---")
