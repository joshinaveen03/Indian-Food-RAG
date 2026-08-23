# 🍛 Indian Food Recipes RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit, LangChain, OpenAI, and ChromaDB**.

The application uses `Food_Recipes.pdf` as the knowledge source. Users can ask questions about Indian food recipes, and the application retrieves relevant document chunks before generating an answer with OpenAI.

## 🚀 Project Overview

This project demonstrates an end-to-end RAG application:

```text
Food_Recipes.pdf
       ↓
   PyPDFLoader
       ↓
Document Loading
       ↓
Text Chunking
(500 characters
 overlap = 50)
       ↓
OpenAI Embeddings
text-embedding-3-small
       ↓
   ChromaDB
       ↓
   Retriever
   Top 4 chunks
       ↓
   GPT-4o-mini
       ↓
    Answer
       ↓
Retrieved Sources
```

## ✨ Features

- 📄 Load Indian food recipes from a PDF
- ✂️ Split PDF content into smaller chunks
- 🔢 Generate vector embeddings using OpenAI
- 🗄️ Store embeddings in ChromaDB
- 🔎 Retrieve the 4 most relevant chunks
- 🤖 Generate answers using GPT-4o-mini
- 💬 Interactive Streamlit chatbot interface
- 📚 Display retrieved source chunks
- 📖 Display PDF page numbers for retrieved sources
- 🧠 Maintain chat history during the Streamlit session
- ⚡ Streamlit caching for PDF processing and vector-store creation

## 📁 Project Structure

```text
Indian-Food-RAG/
│
├── foodMenu.py
├── foodMenuApp.py
├── Food_Recipes.pdf
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── chroma_db/
```

> **Important:** Do not upload `.env` or your API key to GitHub.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web application |
| LangChain | RAG framework |
| PyPDFLoader | PDF document loading |
| RecursiveCharacterTextSplitter | Document chunking |
| OpenAI Embeddings | Text vectorization |
| ChromaDB | Vector database |
| GPT-4o-mini | Question answering |
| python-dotenv | Environment variable management |

The project dependencies include LangChain, PDF processing libraries, OpenAI integrations, ChromaDB, Streamlit, RAGAS, Pandas, and related packages. 

## 📦 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Indian-Food-RAG
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The uploaded requirements file includes packages such as `langchain-community`, `langchain-text-splitters`, `pypdf`, `pymupdf`, `langchain-openai`, `chromadb`, `langchain-chroma`, and `streamlit`. 

## 🔑 Configure OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

The application checks that the API key exists before continuing.

## 📄 Add the PDF

Place the recipe PDF in the project root:

```text
Food_Recipes.pdf
```

The application automatically checks whether this file exists.

If the PDF is missing, the application displays:

```text
❌ Food_Recipes.pdf not found.
```

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run foodMenuApp.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

## 🧩 RAG Pipeline

### 1. Load PDF

The application uses `PyPDFLoader`:

```python
loader = PyPDFLoader(pdf_path)
return loader.load()
```

### 2. Split Documents

The PDF is divided using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### 3. Create Embeddings

The application uses:

```python
OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

### 4. Store in ChromaDB

The chunks and embeddings are stored locally:

```python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
```

The vector database is stored in:

```text
chroma_db/
```

### 5. Create Retriever

The application retrieves the top 4 relevant chunks:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)
```

### 6. Create LLM

The chatbot uses:

```python
ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
```

### 7. RetrievalQA

The RAG chain is created using:

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
```

### 8. Ask Questions

Users can ask questions such as:

```text
Explain the recipe of Idli?
```

or:

```text
What ingredients are required for Masala Dosa?
```

or:

```text
How is Sambar prepared?
```

## 💬 Streamlit Chat Interface

The application provides:

- Application title
- PDF information in the sidebar
- Number of PDF pages
- Chat history
- Chat input
- Assistant responses
- Retrieved source documents
- Source PDF page numbers

The application title is:

```text
About INDIAN FOOD RECIPES RAG Chatbot
```

## 📚 Retrieved Sources

For each answer, the application can display retrieved chunks inside:

```text
📚 Retrieved Sources
```

Each retrieved source shows:

```text
Chunk 1
Chunk 2
Chunk 3
Chunk 4
```

The application also displays the source filename and PDF page number when available.

## ⚡ Streamlit Caching

The application uses:

```python
@st.cache_data
```

for PDF loading and document splitting.

It uses:

```python
@st.cache_resource
```

for ChromaDB vector-store creation.

This helps avoid repeating expensive processing unnecessarily during Streamlit reruns.

## 🔒 Security

Never commit your OpenAI API key.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
chroma_db/
.ipynb_checkpoints/
```

## 🧪 Example Questions

Try asking:

```text
Explain the recipe of Idli.
```

```text
What are the ingredients for Dosa?
```

```text
How do I prepare Sambar?
```

```text
Tell me about North Indian recipes.
```

```text
What is the preparation method for Paneer Butter Masala?
```

## 🎯 Learning Outcomes

This project helps demonstrate:

- PDF document ingestion
- Document preprocessing
- Text chunking
- Embeddings
- Vector databases
- Semantic retrieval
- Retrieval-Augmented Generation
- LangChain
- OpenAI models
- Streamlit application development
- Source-document retrieval
- Chat history management

## 🔮 Future Enhancements

Possible improvements:

- Support multiple PDF uploads
- Allow users to upload their own documents
- Add PDF upload directly from Streamlit
- Add source citations with clickable pages
- Add conversation memory
- Add recipe-category filters
- Add multilingual recipe questions
- Add RAG evaluation using RAGAS
- Improve retrieval with metadata filtering
- Add a reset-chat button
- Deploy the application to Streamlit Cloud

## ⚠️ Troubleshooting

### API key error

If you see:

```text
❌ OPENAI_API_KEY is missing in .env
```

check that your `.env` file contains:

```env
OPENAI_API_KEY=your_openai_api_key
```

### PDF error

If you see:

```text
❌ Food_Recipes.pdf not found.
```

place `Food_Recipes.pdf` in the same project directory as the Python application.

### Vector database error

If the application reports that the vector database is not ready, make sure the `chroma_db` directory has been created successfully.

## 👨‍💻 Author

**Naveen Joshi**

AI & Technology | Python | Generative AI | RAG | LangChain | Streamlit

## 📜 License

This project is intended for educational and demonstration purposes.
