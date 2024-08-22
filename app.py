import time
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from docx import Document
from io import BytesIO

# Set the page configuration at the top
st.set_page_config(page_title="Chat with me", page_icon="🤖", layout="wide")

# Environment variables ko load karte hain .env file se
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CSS for custom font colors and copy button styling
st.markdown(
    """
    <style>
    .title { color: #fbfeff; }  /* Orange-red */
    .header { color: #3498DB; }  /* Light blue */
    .text-input { color: #36ff33; }  /* Dark red */
    .success { color: #28B463; }  /* Green */
    .menu-title { color: #3498DB; }  /* Blue */
    .file-info { color: #FF6347; }  /* Tomato */
    .message { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .copy-button {
        cursor: pointer;
        border: none;
        background: none;
        color: #007bff;
        font-size: 1.2rem;
        margin-left: 10px;
        vertical-align: middle;
    }
    .copy-button:hover {
        color: #0056b3;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    .icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .message-box {
        max-width: 75%;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .user-message-box {
        
        border-color: #36ff33;
    }
    .assistant-message-box {
        
        border-color: #3498DB;
    }
    .message-text {
        font-size: 14px;
        line-height: 1.5;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script>
    function copyToClipboard(elementId) {{
                        const textToCopy = document.getElementById(elementId).innerText;
                        navigator.clipboard.writeText(textToCopy)
                            .then(() => {{
                                console.log('Text copied to clipboard');
                                alert('Text copied to clipboard: ' + textToCopy);
                            }})
                            .catch(err => {{
                                console.error('Failed to copy text: ', err);
                            }});
                    }}
    </script>
    """,
    unsafe_allow_html=True
)

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from uploaded DOCX files
def get_txt_text(txt_files):
    text = ""
    for txt_file in txt_files:
        text += txt_file.read().decode("utf-8") + "\n"
    return text

# Function to extract text from web URLs
def get_url_text(urls):
    text = ""
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.find_all(['p', 'h1', 'h2', 'li'])
            for element in elements:
                text += element.get_text() + "\n"
        except RequestException as e:
            st.warning(f"Failed to fetch or process URL '{url}': {str(e)}")
    return text

# Function to split extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store for the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain using the Gemini LLM
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using Markdown formatting. Include headings, bullet points, and bold text where appropriate.
    Ensure the answer is clear and well-structured. If the answer is not in the provided context, say, "Answer is not available in the context."; don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input, search for similar content, and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Use FAISS to search for relevant documents
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Save the question and response to session state
    st.session_state.chat_history.append({"user": user_question, "assistant": response["output_text"]})
    
    user_icon_url = "https://via.placeholder.com/40/36ff33/FFFFFF/?text=U"  # Placeholder for user icon
    bot_icon_url = "https://via.placeholder.com/40/3498DB/FFFFFF/?text=A"   # Placeholder for bot icon

    
    st.markdown(
        f"""
        <div class="message-container">
            <img src="{user_icon_url}" class="icon" alt="User Icon">
            <div class="message-box user-message-box">
                <div class="message-text"><strong>User:</strong><br>{user_question}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display assistant response with icon
    st.markdown(
        f"""
        <div class="message-container">
            <img src="{bot_icon_url}" class="icon" alt="Bot Icon">
            <div class="message-box assistant-message-box">
                <div class="message-text"><strong>Assistant:</strong><br>{response['output_text']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# Main function to run Streamlit app
def main():
    st.markdown("<h1 class='title'>Chat with your GenieBot</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='header'> Upload your PDF, DOCX, or URL: Transform Text into Insights </h2>", unsafe_allow_html=True)

    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
        
    if st.session_state.chat_history:
        st.markdown("<h3 class='menu-title'>Chat History:</h3>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            user_icon_url = "https://via.placeholder.com/40/36ff33/FFFFFF/?text=U"  # Placeholder for user icon
            bot_icon_url = "https://via.placeholder.com/40/3498DB/FFFFFF/?text=A"   # Placeholder for bot icon
            st.markdown(
                f"""
                <div class="message-container">
                    <img src="{user_icon_url}" class="icon" alt="User Icon">
                    <div class="message-box user-message-box">
                        <div class="message-text"><strong>User:</strong><br>{chat['user']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )  
            st.markdown(
                f"""
                <div class="message-container">
                    <img src="{bot_icon_url}" class="icon" alt="Bot Icon">
                    <div class="message-box assistant-message-box">
                        <div class="message-text"><strong>Assistant:</strong><br>{chat['assistant']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("No chat history available.")
    
    # Input field for user's message
    user_question = st.chat_input("Ask a Question from the Processed Content")

    if user_question:
        with st.spinner("Generating response..."):
            user_input(user_question)
    
    
    with st.sidebar:
        # Upload PDF files with size check
        st.markdown("<h2 class='menu-title'>Upload Files</h2>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload your files (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
        
        if st.button("Submit & Process Files"):
                if not uploaded_files:
                    st.warning("Please upload PDF files before processing.")
                else:
                    with st.spinner("Processing files.."):
                        for uploaded_file in uploaded_files:
                            file_extension = uploaded_file.name.split(".")[-1].lower()
                            if file_extension == "pdf":
                                raw_text = get_pdf_text([uploaded_file])
                            elif file_extension == "txt":
                                raw_text = get_txt_text([uploaded_file])
                            else:
                                st.warning(f"Unsupported file type: {file_extension}")
                                continue

                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                        st.success("Documents processed successfully!")
            
        # Input URLs with validation
        urls = st.text_area("Enter web URLs (one per line)").split("\n")
        urls = [url.strip() for url in urls if url.strip()]  # Clean empty entries
        
        if st.button("Submit & Process URLs"):
            if not urls:
                st.warning("Please enter web URLs before processing.")
            else:
                with st.spinner("Processing URLs..."):
                    raw_text = get_url_text(urls)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.balloons()
                    st.success("URLs have been processed, you can ask questions now!")



if __name__ == "__main__":
    main()
