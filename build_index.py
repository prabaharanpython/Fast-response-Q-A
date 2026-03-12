import os
import glob
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = "documents"
INDEX_PATH = "faiss_index"

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def build_index():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created directory {DOCS_DIR}. Please add some .docx files and run again.")
        return
        
    docx_files = glob.glob(os.path.join(DOCS_DIR, "*.docx"))
    if not docx_files:
        print(f"No .docx files found in {DOCS_DIR}.")
        return

    documents = []
    
    # Extract text
    for file_path in docx_files:
        print(f"Processing {file_path}...")
        text = extract_text_from_docx(file_path)
        documents.append({"text": text, "source": file_path})

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    
    texts = []
    metadatas = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"]}] * len(chunks))

    print(f"Created {len(texts)} chunks from {len(docx_files)} files.")
    
    # Build FAISS index
    print("Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save to disk
    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}.")

if __name__ == "__main__":
    build_index()
