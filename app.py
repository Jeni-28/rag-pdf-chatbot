import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="AI PDF Chatbot (RAG)", page_icon="📚")
st.title("📚 AI PDF Chatbot — Ask Questions From Your PDF")

# --- API Key input ---
api_key = st.sidebar.text_input("Enter your Google AI Studio API Key", type="password")
st.sidebar.markdown("[Get a free key here](https://aistudio.google.com/app/apikey)")

if not api_key:
    st.warning("Please enter your Google AI Studio API key in the sidebar to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- Cache the vectorstore so we don't rebuild it every time ---
@st.cache_resource(show_spinner="Reading and indexing your PDF...")
def build_vectorstore(file_bytes, api_key):
    # Save uploaded PDF to a temp file (PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 1. Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

    # 4. Store embeddings in FAISS vector database
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)  # cleanup temp file
    return vectorstore


if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    vectorstore = build_vectorstore(file_bytes, api_key)

    st.success("✅ PDF processed! You can now ask questions below.")

    # --- LLM setup ---
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2
)

    # --- RAG chain: retriever + LLM ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": query})
                answer = result["result"]
                sources = result["source_documents"]

                st.markdown(answer)

                with st.expander("📄 Source chunks used"):
                    for i, doc in enumerate(sources):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i+1} (Page {page}):**")
                        st.write(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF to get started.")