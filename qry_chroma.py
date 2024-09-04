from langchain_experimental.chat_models import Llama2Chat
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from os.path import expanduser
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

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

model_path = expanduser("C:/Program Files/Llama2_GGUF/llama-2-13b-chat.Q3_K_M.gguf")

# Using SentenceTransformer embeddings
default_ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to ChromaDB collection
collection = Chroma(
    collection_name="pdf_collections",
    embedding_function=default_ef,
    persist_directory="DB",
)

# Create document retriever
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

print("Welcome to the chatbot! Type 'exit' to end the conversation.")

while True:
    prompt = input("You: ")
    if prompt.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    docs_with_score = docsearch.vectorstore.similarity_search_with_relevance_scores(prompt, k=100, score_threshold=0.3)
    docs = [doc[0] for doc in docs_with_score]
    response = chain.invoke({"question": prompt, "input_documents": docs})
    
