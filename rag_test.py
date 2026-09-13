import os
import oracledb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

# 1. Database Connection
connection = oracledb.connect(
    user=os.environ["ORACLE_DB_USER"],
    password=os.environ["ORACLE_DB_PASSWORD"],
    dsn=os.environ["ORACLE_DB_DSN"]
)
cursor = connection.cursor()

# 2. Text File Read Karein
with open('data/test_data.txt', 'r') as file:
    text = file.read()

# 3. Text ko Chunks mein Divide Karein
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_text(text)

# 4. Har Chunk ko Vector mein Convert Karein
print("Converting text to vectors...")
for chunk in chunks:
    response = ollama.embed(model='nomic-embed-text', input=chunk)
    embedding = response['embeddings'][0]
    
    # 5. Database mein Store Karein (setinputsizes USE KAREIN!)
    cursor.setinputsizes(2, oracledb.DB_TYPE_VECTOR)  # ✅ Ye sabse important line hai!
    cursor.execute(
        "INSERT INTO rag_test (text_data, embedding) VALUES (:1, :2)",
        [chunk, embedding]  # ✅ Simple list bind karein
    )

connection.commit()
print(f"Successfully stored {len(chunks)} chunks in the database!")

# 6. Close Connection
cursor.close()
connection.close()
