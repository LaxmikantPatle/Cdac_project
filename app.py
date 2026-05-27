"""
AI-Powered CDAC Information Chatbot
Built with Streamlit, LangChain, RAG, FAISS & Sentence-Transformers
Answer CDAC-related queries from PDFs and website data
"""

import os
import streamlit as st
import pickle
import faiss
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import requests

# Page configuration
st.set_page_config(
    page_title="CDAC Information Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# ============================================
# SIDEBAR - CONFIGURATION & DATA MANAGEMENT
# ============================================
with st.sidebar:
    st.title("🎓 CDAC Chatbot")
    st.markdown("---")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input(
        "HuggingFace API Token:",
        type="password",
        help="Get free token: https://huggingface.co/settings/tokens"
    )
    
    st.markdown("---")
    
    # Data Management
    st.markdown("### 📚 Data Management")
    
    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload CDAC PDFs (courses, syllabus, placements)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload CDAC brochures, syllabus PDFs, placement reports"
    )
    
    # Build Index Button
    if st.button("📥 Build Vector Index", type="primary"):
        with st.spinner("🔄 Building vector index... This may take 1-2 minutes"):
            try:
                documents = []
                
                # Process uploaded PDFs
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                documents.append(text)
                    st.info(f"✅ Loaded {len(uploaded_files)} PDFs")
                
                # Load existing PDFs from data folder
                pdf_dir = Path("data/cdac_pdfs")
                if pdf_dir.exists():
                    for pdf_file in pdf_dir.glob("*.pdf"):
                        reader = PdfReader(str(pdf_file))
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                documents.append(text)
                    st.info(f"✅ Loaded PDFs from data/cdac_pdfs/")
                
                # Load text files
                txt_dir = Path("data/cdac_webdata")
                if txt_dir.exists():
                    for txt_file in txt_dir.glob("*.txt"):
                        with open(txt_file, "r", encoding="utf-8") as f:
                            documents.append(f.read())
                
                if not documents:
                    st.warning("⚠️ No documents found! Upload PDFs or add to data/cdac_pdfs/")
                    st.stop()
                
                st.success(f"✅ Loaded {len(documents)} document pages")
                
                # Chunk documents
                with st.spinner("📝 Chunking documents..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len
                    )
                    chunks = text_splitter.create_documents(documents)
                    st.success(f"✅ Created {len(chunks)} document chunks")
                
                # Create embeddings
                with st.spinner("🧠 Creating sentence-transformer embeddings..."):
                    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    st.session_state.embedding_model = embedding_model
                    
                    texts = [chunk.page_content for chunk in chunks]
                    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
                    st.success(f"✅ Embeddings created (dimension: {embeddings.shape[1]})")
                
                # Create FAISS index
                with st.spinner("📊 Creating FAISS vector index..."):
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(embeddings)
                    st.success("✅ FAISS index created")
                
                # Save index
                vector_store_dir = Path("vectorstore")
                vector_store_dir.mkdir(exist_ok=True)
                
                with open(vector_store_dir / "documents.pkl", "wb") as f:
                    pickle.dump(chunks, f)
                faiss.write_index(index, vector_store_dir / "index.faiss")
                
                st.session_state.vector_store = {
                    "index": index,
                    "documents": chunks,
                    "embeddings": embeddings
                }
                st.session_state.index_built = True
                
                st.success("🎉 Vector index built successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error building index: {str(e)}")
                st.exception(e)
    
    if st.session_state.index_built:
        st.success("✅ Index ready for queries!")
    
    st.markdown("---")
    
    # Quick Questions
    st.markdown("### ⚡ Quick Questions")
    quick_questions = [
        "What is PG-DAC course?",
        "CDAC admission process?",
        "Placement records & salary?",
        "CCAT exam pattern?",
        "What courses does CDAC offer?",
        "CDAC centers in India?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=question):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# ============================================
# MAIN HEADER
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎓 CDAC Information Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered assistant for CDAC courses, admissions, placements & more</p>', unsafe_allow_html=True)

# ============================================
# RAG CHAIN INITIALIZATION
# ============================================
def initialize_rag_chain(api_key):
    """Initialize the RAG chain with conversational memory"""
    if st.session_state.rag_chain and st.session_state.chat_memory:
        return st.session_state.rag_chain
    
    if not api_key:
        st.error("⚠️ Please enter your HuggingFace API token in the sidebar")
        return None
    
    try:
        # Initialize LLM
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=api_key,
            model_kwargs={"temperature": 0.7, "max_length": 500}
        )
        
        # Initialize conversational memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.session_state.chat_memory = memory
        
        # Create custom retriever using FAISS
        vector_store = st.session_state.vector_store
        embedding_model = st.session_state.embedding_model
        
        def retriever(query):
            """Search for similar documents using FAISS"""
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
            distances, indices = vector_store["index"].search(query_embedding, k=3)
            
            results = []
            for idx in indices[0]:
                if idx < len(vector_store["documents"]):
                    doc = vector_store["documents"][idx]
                    results.append(Document(
                        page_content=doc.page_content,
                        metadata={"source": "cdac_document"}
                    ))
            return results
        
        # Build Conversational RAG Chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            chain_type="stuff"
        )
        
        st.session_state.rag_chain = rag_chain
        return rag_chain
        
    except Exception as e:
        st.error(f"❌ Error initializing RAG chain: {str(e)}")
        return None

# ============================================
# DISPLAY CHAT MESSAGES
# ============================================
def display_messages():
    """Display all chat messages with sources"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources (click to expand)"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        st.markdown(f"""
                        **Source {i}:**
                        ```
                        {source[:400]}...
                        ```
                        """)

# ============================================
# SEND MESSAGE TO CHATBOT
# ============================================
def send_message(message, api_key):
    """Send message to chatbot and get RAG response"""
    if not st.session_state.index_built:
        return "Please build the vector index first using the button in the sidebar.", []
    
    if not api_key:
        return "Please enter your HuggingFace API key in the sidebar.", []
    
    chain = initialize_rag_chain(api_key)
    if not chain:
        return "Please configure your API key first.", []
    
    try:
        with st.spinner("🤔 Thinking..."):
            result = chain({"question": message})
            answer = result["answer"]
            sources = [doc.page_content for doc in result.get("source_documents", [])]
            return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# CHAT INTERFACE
# ============================================
display_messages()

# Chat input
if prompt := st.chat_input("Ask about CDAC courses, admissions, placements, syllabus..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display bot response
    with st.chat_message("assistant"):
        response, sources = send_message(prompt, api_key)
        st.markdown(response)
        
        if sources:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources[:3], 1):
                    st.markdown(f"**Source {i}:** {source[:300]}...")
    
    # Add bot message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 <strong>CDAC Information Chatbot</strong></p>
    <p>Built with ⚡ Streamlit | 🦜️ LangChain | 🧠 RAG | 📊 FAISS | 🤖 Sentence-Transformers</p>
    <p style="font-size: 0.9rem;">Retrieval-Augmented Generation for CDAC Institute Information</p>
</div>
""", unsafe_allow_html=True)