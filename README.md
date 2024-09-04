Machine Learning Pipeline Design to LLM base on PDf files 
==============================
This project is a PDF reading and question-answering pipeline designed to interact with the content of PDF files. It utilizes an integration of ChromaDB and a large language model (LLM) based on the Llama2 model to read, process, and answer questions about the content of PDF documents. The pipeline includes PDF loading, text extraction, chunking, embedding, and question-answering stages. The Llama2 model, known for its robust performance, is used to handle the complexities of natural language processing tasks. The interactive interface, built with Gradio, allows users to upload a PDF file, process it, and ask questions based on the content of the file.

Directory Structure
-------------------
- `LLMPdf` Python scriptto extract text from PDF files, split the text into manageable chunks, and then store these chunks in a persistent ChromaDB collection.
- `qry_chroma` Python script to sets up a chatbot that answers questions based on the content stored in a ChromaDB collection.
- `app` Python script to creates a Gradio interface for a chatbot that allows users to upload a PDF file, process it, and then ask questions about the content of that PDF and also its a combination of LLMpdf codes and qry_croma script.

Download LLama2
------------
Open this link : https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q3_K_M.gguf and download the file, next put the files in your localDisk/Program Files 

Installation (Terminal)
------------
pip install gradio pymupdf pdfplumber langchain langchain_community langchain_experimental chromadb 

Make sure to install the necessary libraries and modules before running the project.

Additional Notes:
------------
Ensure that the paths to your models (e.g., Llama2 model) are correct.
Depending on your system, you may need additional dependencies to support GPU acceleration (like CUDA for PyTorch, if you plan to use GPU).

The Example of the prototype.
------------

![image](https://github.com/user-attachments/assets/f5a2133d-2b89-43de-9b62-2a9add1612ab)
