import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# Load Environment
load_dotenv()

# Streamlit Configuration

st.set_page_config(
    page_title="Indian Food RAG",
    page_icon="🍛",
    layout="wide"
)

# PDF Path

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "Food_Recipes.pdf"
)

CHROMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "chroma_db"
)

# Check PDF

if not os.path.exists(PDF_PATH):
    st.error("❌ Food_Recipes.pdf not found.")
    st.stop()

# Check API Key

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is missing in .env")
    st.stop()

# Load PDF

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    return loader.load()

# Split Documents

def split_documents(pdf_path):

    documents = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)

# Create Vector Store

def create_vectorstore(chunks):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore

# Create Chunks

chunks = split_documents(PDF_PATH)

print("Number of chunks:", len(chunks))

# Create Vector Store

vectorstore = create_vectorstore(chunks)
print("✅ Vector store created successfully!")

# Create Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
print("✅ Retriever created successfully!")

# Create LLM

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Create Retrieval QA Chain

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("✅ RetrievalQA chain created successfully!")

# Ask Question
question = "Explain Recipe of Idli?"

response = qa.invoke({
    "query": question
})

# Display Answer

print(response)
