import streamlit as st
from dotenv import load_dotenv
from ragbot import get_chatbot_response 
from langchain_core.messages import HumanMessage
from langchain.embeddings import HuggingFaceEmbeddings
from wordembedding import create_vectordb

load_dotenv()

embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
st.title("---------->    Askify    <-----------")

uploaded_file = st.file_uploader("Drag n drop ur text/.txt file here",type='txt')

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")

if uploaded_file:
    file_content = uploaded_file.read().decode('utf-8', errors='ignore')
    if not file_content.strip():
        st.warning("Uploaded file is empty or unreadable. Please check the file.")
        st.stop()
    doc_count = create_vectordb(file_content)
    if doc_count == 0:
        st.warning("No valid content found in the uploaded file. Please upload a file with meaningful text.")
    
if user_input:
    # user response
    st.session_state.messages.append({"role": "user", "content": user_input})

    chat_history = [HumanMessage(content=msg["content"]) for msg in st.session_state.messages if msg["role"] == "user"]
    response = get_chatbot_response(user_input, chat_history)

    # bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()