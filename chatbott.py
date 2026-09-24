import os
import streamlit as st
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# INTELLICHAT - RAG CHATBOT
# STREAMLIT + GROQ API + GPT-OSS 20B
# ============================================================

MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliChat",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are IntelliChat, a helpful, intelligent and friendly AI assistant.

Rules:
- Answer questions accurately.
- Explain difficult topics clearly.
- Use examples when useful.
- Do not make up information.
- Maintain conversation context.
- When document context is provided, use it to answer the user's question.
- If the answer is not available in the provided document context,
  clearly say that the information is not available in the uploaded document.
"""


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(uploaded_file):

    file_path = "uploaded_document.pdf"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings and FAISS vector database
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Remove temporary file
    os.remove(file_path)

    return vector_db


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🤖 IntelliChat")

st.write(
    "An intelligent RAG-powered chatbot using GPT-OSS 20B."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("Process Document"):

            with st.spinner("Processing document..."):

                st.session_state.vector_db = (
                    create_vector_database(uploaded_file)
                )

            st.success("Document processed successfully!")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask IntelliChat something..."
)


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    context = ""

    if st.session_state.vector_db is not None:

        with st.spinner("Searching the knowledge base..."):

            retrieved_docs = (
                st.session_state.vector_db
                .similarity_search(
                    user_input,
                    k=4
                )
            )

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )


    # ========================================================
    # AUGMENT PROMPT WITH RETRIEVED CONTEXT
    # ========================================================

    if context:

        user_prompt = f"""
Use the following retrieved document context to answer
the user's question.

---------------- DOCUMENT CONTEXT ----------------

{context}

---------------- END CONTEXT ----------------

User Question:
{user_input}

Answer using the provided context.
"""

    else:

        user_prompt = user_input


    # ========================================================
    # GPT-OSS 20B RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            answer = response.choices[0].message.content

        st.markdown(answer)


    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
