import streamlit as st
import requests
import time

# API_URL = "http://api:8080/ask"

# To this:
API_URL = "http://localhost:8080/ask"


st.set_page_config(page_title="Local Agentic RAG", page_icon="🧠")
st.title("Local Agentic RAG")
st.caption("Powered by Llama 3.1, LangGraph, and ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json={"question": prompt, "chat_history": st.session_state.messages}, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("answer", "No answer found.")
            exec_time = data.get("execution_time_ms", 0) / 1000
            
            formatted_response = f"{answer}\n\n*(Generated in {exec_time:.1f}s)*"
            message_placeholder.markdown(formatted_response)
            
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"API Error: {e}")