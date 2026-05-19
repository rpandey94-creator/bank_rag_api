import requests
from src.retrieval import get_hybrid_context

def generate_answer(user_query: str, pdf_path: str, db_directory: str):
    context = get_hybrid_context(user_query, pdf_path, db_directory)
    
    if not context.strip():
        return "I could not find relevant information in the RBI policy document."

    prompt = f"""You are an expert Bank of Baroda AI Assistant. 
You must answer the user's question based ONLY on the following context from the RBI Master Circular. 
If the answer is not in the context, say "I cannot answer this based on the provided policy." Do not make up information.

CONTEXT:
{context}

USER QUESTION: 
{user_query}

ANSWER:"""

    print("Synthesizing answer with local Llama 3...\n")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error connecting to LLM: {response.status_code}"

if __name__ == "__main__":
    DB_DIR = "vector_db"
    PDF_FILE = "data/rbi_kyc_master.pdf"
    
    QUESTION = input("Enter your policy question: ")
    
    final_answer = generate_answer(QUESTION, PDF_FILE, DB_DIR)
    
    print("\n--- FINAL AI RESPONSE ---")
    print(final_answer)