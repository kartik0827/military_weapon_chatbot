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
You are a military expert chatbot.
Never mention context, documents, provided lists, datasets, or whether information is missing.
Always give a direct factual answer, even if you have to rely on your own knowledge.
If the information below helps, use it silently. If it doesn't, ignore it and answer from your own knowledge.
Information:
{context}
Question:
{question}
Direct Answer:
"""
)


# Build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
     chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
#    return_source_documents=False
)

# Clean and remove vague or unnecessary intro phrases from the final response
def clean_response(text):
    # Lowercase copy for matching
    lower_text = text.lower()

    # Patterns that indicate weak intro phrases to be removed
    bad_starts = [
        "i'm not aware of", 
        "there is no information", 
        "i could not find", 
        "it's not mentioned", 
        "unfortunately", 
        "however", 
        "i can inform you that", 
        "i'm sorry",
        "not mentioned in the provided information"
    ]

    # Remove these phrases if they appear at the start or early in the response
    for phrase in bad_starts:
        if lower_text.startswith(phrase) or lower_text.startswith(" " + phrase):
            text = text[text.find(".") + 1:].strip()
            break

    # Also clean specific fallback phrases
    phrases_to_remove = [
        "mentioned in the provided list", "mentioned in the context", "based on the provided context",
        "in the provided context", "from the context", "according to the context",
        "the context does not mention", "however, I can tell you that", "unfortunately",
        "not mentioned in the provided list", "not listed in the provided context", "not mentioned in the provided information"
    ]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "").strip()

    return text


# Chat function
#def chat_with_bot(message, chat_history):
 #   response = qa_chain.run(message)
  #  return response
from langchain_core.exceptions import OutputParserException

def chat_with_bot(message, chat_history):
    try:
        response = qa_chain.run(message)
        # If response is empty or too generic, fallback to LLM only
        if not response or "I don't know" in response:
            response = llm.invoke(message)
    except OutputParserException:
        response = llm.invoke(message)
    return clean_response(response)


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
