import os
import json
import time
import redis
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
INDEX_PATH = "faiss_index"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Initialize Embeddings & Vector Store
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Warning: Could not load FAISS index from {INDEX_PATH}. Ensure you run build_index.py first.")
    vectorstore = None

# Initialize Groq Client
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set.")
groq_client = Groq(api_key=GROQ_API_KEY)

def get_answer(question: str):
    start_time = time.time()
    
    # Check Redis Cache
    cache_key = f"qa:{question}"
    try:
        cached_response = redis_client.get(cache_key)
        if cached_response:
            response_data = json.loads(cached_response)
            response_data['response_time'] = round(time.time() - start_time, 4)
            response_data['cached'] = True
            return response_data
    except Exception as e:
        print(f"Redis cache error: {e}")
    
    if vectorstore is None:
        return {
            "answer": "Error: FAISS index not loaded. Run build_index.py first.",
            "source_chunks": [],
            "response_time": round(time.time() - start_time, 4),
            "cached": False
        }

    # Retrieve top 3 chunks
    docs = vectorstore.similarity_search(question, k=3)
    source_chunks = [{"content": doc.page_content, "source": doc.metadata.get("source", "unknown")} for doc in docs]
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Groq LLM API Call
    prompt = f"Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and fast QA assistant. Answer concisely based on the context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=300,
        )
        answer = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error calling Groq API: {str(e)}"
        
    response_time = round(time.time() - start_time, 4)
    
    response_data = {
        "answer": answer,
        "source_chunks": source_chunks,
        "response_time": response_time,
        "cached": False
    }
    
    # Cache the result for 24 hours (86400 seconds)
    try:
        redis_client.setex(cache_key, 86400, json.dumps({
            "answer": answer,
            "source_chunks": source_chunks
        }))
    except Exception as e:
        print(f"Redis cache write error: {e}")
        
    return response_data
