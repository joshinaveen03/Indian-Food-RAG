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

@st.cache_data
def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    return loader.load()


# Split Documents

@st.cache_data
def split_documents(pdf_path):

    documents = load_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


# Create Vector Store

@st.cache_resource
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

print(
    "Number of chunks:",
    len(chunks)
)


# Create Vector Store

vectorstore = create_vectorstore(chunks)

print(
    "✅ Vector store created successfully!"
)


# Create Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

print(
    "✅ Retriever created successfully!"
)


# Create LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Create Retrieval QA Chain

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print(
    "✅ RetrievalQA chain created successfully!"
)


# Sidebar

with st.sidebar:

    st.header("📚 PDF Information")

    # File

    st.write("**File:**")

    st.code(
        os.path.basename(PDF_PATH)
    )


# Pages

page_count = len(
    load_pdf(PDF_PATH)
)

st.write(
    "**Pages:**",
    page_count
)


# Main Application

st.title(
    "About INDIAN FOOD RECIPES RAG Chatbot"
)

st.write(
    "Ask questions about the INDIAN FOOD RECIPES."
)


# Chat History

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# Display Previous Messages

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# Chat Input

question = st.chat_input(
    "Ask a question about Food Recipe..."
)


# Process Question

if question:

    if not os.path.exists(CHROMA_PATH):

        st.error(
            "Vector database is not ready."
        )

        st.stop()


    # User Message

    with st.chat_message("user"):

        st.markdown(question)


    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Assistant Message

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching Food Recipes document..."
        ):

            try:

                # Invoke QA Chain

                response = qa.invoke(
                    {
                        "query": question
                    }
                )


                # Get Answer

                answer = response["result"]


                # Display Answer

                st.markdown(answer)


                # Get Retrieved Documents

                documents = response.get(
                    "source_documents",
                    []
                )


                # Display Sources

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for i, doc in enumerate(
                        documents
                    ):

                        st.write(
                            f"### 📄 Chunk {i + 1}"
                        )

                        st.write(
                            doc.page_content[:500]
                        )


                        if doc.metadata:

                            source = doc.metadata.get(
                                "source",
                                "Unknown"
                            )

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            # Convert zero-based PDF page
                            # number to normal page number

                            if isinstance(page, int):

                                page = page + 1


                            st.caption(
                                f"Source: {source} | Page: {page}"
                            )


            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )

                st.info(
                    "Check your PDF, ChromaDB, .env, "
                    "and OpenAI API key."
                )


    # Save Assistant Answer

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )