from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
import os

# 🔹 Optional: Check file exists
TEXT_FILE_PATH = r"C:\Users\Kartik\Desktop\chatbot\extracted\military_weapons.txt"
if not os.path.isfile(TEXT_FILE_PATH):
    raise FileNotFoundError(f"❌ File '{TEXT_FILE_PATH}' not found. Make sure it exists.")

def load_docs():
    loader = TextLoader(TEXT_FILE_PATH)
    return loader.load()

def create_vector_db():
    docs = load_docs()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory="db")
    vectordb.persist()
    print("✅ Data indexed in vector store.")

if __name__ == "__main__":
    create_vector_db()
