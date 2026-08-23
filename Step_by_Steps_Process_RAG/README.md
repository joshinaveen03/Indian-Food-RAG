# Food Recipe RAG – Data Ingestion & Retrieval

A simple Retrieval-Augmented Generation (RAG) workflow built with Python, LangChain, OpenAI embeddings, ChromaDB, and Streamlit-compatible components.

The notebook processes a 202-page `Food_Recipes.pdf` document containing South Indian, North Indian, and regional recipes, then prepares the content for semantic retrieval and question answering.

## Project Overview

This project demonstrates the main stages of a RAG pipeline:

1. Load environment variables
2. Load the PDF document
3. Split the document into smaller chunks
4. Generate OpenAI embeddings
5. Store document chunks in ChromaDB
6. Create a retriever
7. Connect the retriever to an OpenAI chat model
8. Ask questions against the document
9. Retrieve the most relevant document chunks

## RAG Architecture

```text
Food_Recipes.pdf
       |
       v
   PyPDFLoader
       |
       v
   PDF Documents
       |
       v
CharacterTextSplitter
(chunk_size=500,
 chunk_overlap=50)
       |
       v
   Document Chunks
       |
       v
OpenAI Embeddings
text-embedding-3-small
       |
       v
     ChromaDB
   ./chroma_db
       |
       v
    Retriever
       |
       v
   GPT-4o-mini
       |
       v
   Question / Answer
```

## Files

```text
.
├── Dataingestion(1).ipynb
├── requirements.txt
├── Food_Recipes.pdf
├── chroma_db/
└── README.md
```

## Technologies Used

- Python
- Jupyter Notebook
- LangChain
- LangChain Community
- LangChain Text Splitters
- PyPDF / PyMuPDF
- OpenAI
- ChromaDB
- FAISS
- Streamlit
- RAGAS
- Pandas

The provided `requirements.txt` contains the project dependencies, including LangChain, PDF libraries, OpenAI integrations, ChromaDB, Streamlit, and RAGAS.

## Document Loading

The notebook uses `PyPDFLoader` to load:

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "Food_Recipes.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()
```

The notebook successfully loads 202 pages from the recipe PDF.

## Text Chunking

The document is split using `CharacterTextSplitter`.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_documents(docs)
```

### Chunk Configuration

| Parameter | Value |
|---|---:|
| Separator | New line |
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Length function | `len` |

## Embeddings

The project uses OpenAI's `text-embedding-3-small` model to convert document chunks into vector representations.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

## Vector Database

The generated embeddings are stored in ChromaDB.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

The local vector database is stored in:

```text
./chroma_db
```

## Retriever

The Chroma vector store is converted into a retriever.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

The retriever returns the most relevant document chunks for a question.

The notebook also demonstrates retrieving the top 3 relevant chunks:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

top_chunks = retriever.invoke(question)
```

## Question Answering

The notebook connects the retriever with an OpenAI chat model using `RetrievalQA`.

```python
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

A question can then be submitted:

```python
question = "What is the main topic of the document?"

response = qa.invoke({
    "query": question
})

print(response["result"])
```

## Example

Example question:

```text
What is the main topic of the document?
```

The notebook retrieves an answer describing the document as a collection of South and North Indian recipes with cooking notes and serving information.

## Installation

Create and activate a Python virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## OpenAI API Key

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not upload the `.env` file or your API key to GitHub.

Add this to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
chroma_db/
.ipynb_checkpoints/
```

## Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Dataingestion(1).ipynb
```

Run the notebook cells in order.

Make sure `Food_Recipes.pdf` is available in the project directory before running the PDF loading cell.

## Learning Objectives

This project is useful for learning:

- PDF document ingestion
- Document chunking
- Text embeddings
- Vector databases
- Semantic search
- Retriever creation
- Retrieval-Augmented Generation
- LangChain document processing
- Connecting retrieved context with an LLM

## Future Improvements

Possible extensions include:

- Add a Streamlit chat interface
- Allow users to upload their own PDF
- Add source/page citations to answers
- Add conversation memory
- Improve chunking strategy
- Add metadata filtering
- Add RAG evaluation using RAGAS
- Add support for multiple documents
- Add persistent ChromaDB configuration
- Add a production-ready retrieval pipeline

## Disclaimer

This repository is intended for educational and demonstration purposes. Recipe content is used as the source document for demonstrating document ingestion, vector search, retrieval, and question answering.
