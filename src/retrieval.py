import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

def get_hybrid_context(user_query: str, pdf_path: str, db_directory: str):
    print(f"Executing Custom Hybrid Search for: '{user_query}'...\n")
    
    # --- 1. THE SEMANTIC SEARCH (ChromaDB) ---
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
    semantic_docs = vector_db.similarity_search(user_query, k=5)
    
    # --- 2. THE LEXICAL SEARCH (BM25) ---
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    lexical_retriever = BM25Retriever.from_documents(chunks)
    lexical_retriever.k = 5
    lexical_docs = lexical_retriever.invoke(user_query)
    
    # --- 3. CUSTOM ENSEMBLE MERGER ---
    unique_docs = {}
    for doc in (semantic_docs + lexical_docs):
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc.page_content
    
    # --- 4. FORMAT OUTPUT ---
    context = ""
    for i, content in enumerate(unique_docs.values()):
        context += f"\n--- Source {i+1} ---\n{content}\n"
            
    return context

if __name__ == "__main__":
    DB_DIR = "vector_db"
    PDF_FILE = "data/rbi_kyc_master.pdf"
    
    QUESTION = input("Enter your query: ")
    
    found_text = get_hybrid_context(QUESTION, PDF_FILE, DB_DIR)
    print(found_text)