import chromadb
import re
import pdfplumber
from chromadb.utils import embedding_functions
import os
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


default_ef = embedding_functions.DefaultEmbeddingFunction()

def extract_text_from_pdf(pdf_path, chunk_size=1500):
    text = ""
    chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                if page_text:
                    page_text = re.sub(r'\s+', ' ', page_text)  
                    page_text = re.sub(r'[\r\n]+', '\n', page_text)  
                    page_text = re.sub(r'(\s*Page\s*\d+\s*)', '', page_text)  
                    page_text = page_text.strip() 

                    text += page_text + '\n'
                else:
                    print(f"Warning: No text found on page {page_num + 1}")

                if page.extract_tables():
                    for table in page.extract_tables():
                        for row in table:
                            row_text = ' | '.join(str(cell) if cell is not None else '' for cell in row)
                            text += row_text + '\n'
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    
    while len(text) > chunk_size:
        split_point = text.rfind('\n', 0, chunk_size)
        if split_point == -1:
            split_point = chunk_size
        
        chunk = text[:split_point].strip()
        if chunk:
            chunks.append(chunk)
        text = text[split_point:].strip()

    if text:
        chunk = text.strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks


client = chromadb.PersistentClient(path="DB")
collection = client.get_or_create_collection(name="pdf_collections", embedding_function=default_ef)

def add_chunks_to_chromadb(chunks):
    max_batch = 5461
    batch_counter = 0

    for i in range(0, len(chunks), max_batch):
        batch_docs = chunks[i:i+max_batch]
        batch_ids = [f"doc_{batch_counter + idx}" for idx in range(len(batch_docs))]

     
        print(f"Adding batch IDs: {batch_ids}")

   
        collection.add(documents=batch_docs, ids=batch_ids)
        batch_counter += len(batch_docs)

    print(f"Number of documents in collection: {collection.count()}")


pdf_paths = [
    "C:/Users/USER/Desktop/LLMproject/Pdf/SocialMedia.pdf",
    "C:/Users/USER/Desktop/LLMproject/Pdf/TheBenefitsofSamplingSportsDuringChildhood.pdf",
    "C:/Users/USER/Desktop/LLMproject/Pdf/ai-report.pdf",
    "C:/Users/USER/Desktop/LLMproject/Pdf/FACTORSAFFECTINGDIGITALECONOMYEDUCATION.pdf"
]


for pdf_path in pdf_paths:
    if os.path.exists(pdf_path):
        try:
            print(f"Processing file: {pdf_path}")  
            chunks = extract_text_from_pdf(pdf_path)
            if chunks:
                add_chunks_to_chromadb(chunks)
                print(f"PDF content from {pdf_path} successfully added to the ChromaDB collection.")
            else:
                print(f"No content extracted from {pdf_path}.")
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    else:
        print(f"File not found: {pdf_path}")
