import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_database(pdf_path: str, persist_directory: str):
    print(f"1. Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("3. Downloading Open-Source Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("4. Generating Vectors and Saving to ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )
    
    print(f"\nSuccess! Vector database saved locally at: {persist_directory}")
    return vector_db

if __name__ == "__main__":
    PDF_FILE = "data/rbi_kyc_master.pdf"
    DB_DIR = "vector_db"
    
    if os.path.exists(PDF_FILE):
        build_vector_database(PDF_FILE, DB_DIR)
    else:
        print(f"Error: Could not find {PDF_FILE}. Did you put the PDF in the 'data' folder?")