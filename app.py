import gradio as gr
import os
import re
import fitz  
import tempfile
from typing import List
from langchain_experimental.chat_models import Llama2Chat
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
import pdfplumber
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


prompt_template = """
You are an assistant tasked with answering questions based on the context provided.
All answers must be derived from the context provided.
Only provide direct answers to the questions asked, without adding any extra information

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """You are an assistant tasked with answering questions based on the context provided.
All answers must be derived from the context provided.
Only provide direct answers to the questions asked, without adding any extra information
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


pdf_path = ""
collection = None
llm = None
chain = None
docsearch = None

def initialize_llm():
    global collection, llm, chain, docsearch
    model_path = "C:/Program Files/Llama2_GGUF/llama-2-13b-chat.Q3_K_M.gguf"
    
    
    default_ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    collection = Chroma(
        collection_name="pdf_collections",
        embedding_function=default_ef,
        persist_directory="DB",
    )

   
    docsearch = collection.as_retriever()

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = -1
    n_batch = 512

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
    )

    model = Llama2Chat(llm=llm, verbose=True, callback_manager=callback_manager)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='text', return_messages=True)

    chain = load_qa_chain(model, chain_type="stuff", callback_manager=callback_manager, prompt=CHAT_PROMPT)

def process_pdf(pdf_file):
    try:
        
        print(f"Type of pdf_file: {type(pdf_file)}")
        print(f"Attributes of pdf_file: {dir(pdf_file)}")

        
        if hasattr(pdf_file, 'name'):
            
            file_path = pdf_file.name
            with open(file_path, 'rb') as f:
                pdf_file_content = f.read()
        elif hasattr(pdf_file, 'value'):
            
            pdf_file_content = pdf_file.value
        else:
            return "Unsupported file type."

        
        global pdf_path
        pdf_path = "C:/Users/USER/Desktop/LLMproject/Temp/uploaded_pdf.pdf"

        
        with open(pdf_path, 'wb') as temp_file:
            temp_file.write(pdf_file_content)  

      
        original_size = len(pdf_file_content)
        saved_size = os.path.getsize(pdf_path)
        if original_size != saved_size:
            raise ValueError("The saved file size does not match the original file size.")

        print(f"PDF file saved successfully at {pdf_path}, size: {saved_size} bytes")

        
        if not collection:
            initialize_llm()

        chunks = extract_text_from_pdf(pdf_path)
        if chunks:
            add_chunks_to_chromadb(chunks)
        return "PDF processed successfully. You can now ask questions about it."
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return f"Error processing PDF: {e}"


def extract_text_from_pdf(pdf_path, chunk_size=1500):
    text = ""
    chunks = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return "Error: The PDF has no pages."

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

def add_chunks_to_chromadb(chunks):
    max_batch = 5461
    batch_counter = 0

    try:
        for i in range(0, len(chunks), max_batch):
            batch_docs = chunks[i:i+max_batch]
            batch_ids = [f"doc_{batch_counter + idx}" for idx in range(len(batch_docs))]

         
            documents = [Document(page_content=chunk) for chunk in batch_docs]

           
            print(f"Adding batch IDs: {batch_ids}")

            
            collection.add_documents(documents=documents, ids=batch_ids)
            batch_counter += len(batch_docs)

       
        print(f"Documents added to collection: {batch_counter}")
    except Exception as e:
        print(f"Error adding chunks to ChromaDB: {e}")


def answer_question(question):
    if not collection:
        initialize_llm()

    try:
        
        docs_with_score = docsearch.vectorstore.similarity_search_with_relevance_scores(question, k=10, score_threshold=0.3)
        
        combined_text = ""
        for doc, _ in docs_with_score:
            if isinstance(doc, Document):
                combined_text += doc.page_content + '\n'
            else:
                print(f"Unexpected document type: {type(doc)}")
                return f"Error: Unexpected document type retrieved: {type(doc)}"

        max_context_length = 2048
        if len(combined_text) > max_context_length:
            combined_text = combined_text[:max_context_length]

        response = chain.invoke({"question": question, "input_documents": [Document(page_content=combined_text)]})

        print(f"Response from chain.invoke(): {response}")

     
        return response.get('output_text', 'No relevant information found.')
    except Exception as e:
        return f"Error answering question: {e}"





with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF")
            process_button = gr.Button("Process PDF")
            pdf_output = gr.Textbox(label="PDF Status", placeholder="Processing status will be displayed here", lines=1)

            process_button.click(process_pdf, inputs=pdf_input, outputs=pdf_output)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question here")
            submit_button = gr.Button("Submit Question")
            answer_output = gr.Textbox(label="Answer", placeholder="Answer will be displayed here", lines=5)

            submit_button.click(answer_question, inputs=question_input, outputs=answer_output)

demo.launch()

