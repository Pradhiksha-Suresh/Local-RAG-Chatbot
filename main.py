import tempfile
from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


# -----------------------------
# Helpers
# -----------------------------
def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def load_documents(file_path: str, file_type: str):
    if file_type == "pdf":
        return PyPDFLoader(file_path).load()
    if file_type == "csv":
        return CSVLoader(file_path=file_path).load()
    if file_type == "docx":
        return Docx2txtLoader(file_path).load()
    raise ValueError(f"Unsupported file type: {file_type}")


@st.cache_resource(show_spinner=False)
def build_qa_chain(file_path: str, file_type: str, k: int, ollama_model: str):
    documents = load_documents(file_path, file_type)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Local embeddings (downloads model once)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = tempfile.mkdtemp(prefix="chroma_")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Local LLM via Ollama
    llm = ChatOllama(model="llama3", temperature=0)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("Local RAG Chatbot")

    st.sidebar.header("Settings")
    ollama_model = st.sidebar.text_input("Ollama model name", value="llama3.1")
    k = st.sidebar.slider("Top-K retrieved chunks", 2, 10, 4, 1)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv", "docx"])
    if uploaded_file is None:
        st.info("Upload a PDF, CSV, or DOCX to start.")
        return

    file_type = uploaded_file.name.split(".")[-1].lower()

    try:
        file_path = save_uploaded_file(uploaded_file)
        qa_chain = build_qa_chain(file_path, file_type, k=k, ollama_model=ollama_model)
        st.success(f"Loaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to load/build RAG chain: {e}")
        return

    # Show prior chat
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    user_query = st.chat_input("Ask a question about your document...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_query, "chat_history": st.session_state.chat_history})

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for i, doc in enumerate(source_docs, start=1):
                    st.markdown(f"**Source {i}**")
                    st.write(doc.page_content)

        st.session_state.chat_history.append((user_query, answer))


if __name__ == "__main__":
    main()
