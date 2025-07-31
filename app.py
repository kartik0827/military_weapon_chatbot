from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import gradio as gr
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load and split the PDF document
loader = PyPDFLoader("data/military_weapons_dataset.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

# Generate embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding)

# Create retriever
retriever = vectorstore.as_retriever()

# Use Groq model (via OpenAI-compatible API)
llm = ChatOpenAI(
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1",
    model_name="llama3-8b-8192",
    temperature=0.5
)

# prompt template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a military expert chatbot. Use the following context to answer the user's question briefly and accurately. 
If the context doesn't fully help, use your own knowledge to complete the answer. Don't say "Based on the provided context" or anything like that.
    Context: {context}
    Question: {question}
    brief Answer:"""
)

# Build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
     chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
#    return_source_documents=False
)

# Chat function
def chat_with_bot(message, chat_history):
    response = qa_chain.run(message)
    return response

# Gradio interface
interface = gr.ChatInterface(
    fn=chat_with_bot,
    title="🪖 Military Weapon Chatbot",
    description="Ask anything about military weapons.",
    examples=[
        "range of brahmos",
        "Tell me about the Rafale jet",
        "List types of military drones"
    ],
    chatbot=gr.Chatbot(label="🪖 Military Weapon Chatbot", type='messages'),
)

interface.launch(share=True)




