import streamlit as st

st.set_page_config(
    page_title="AI Medical RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical RAG Assistant")

st.caption(
    "WHO Medical Knowledge • Oracle AI Vector Search • Ollama • Gemini"
)

st.divider()

st.info(
    "GUI setup is working. Next we will connect this interface "
    "to your existing RAG engine."
)

st.text_input(
    "Ask a medical question",
    placeholder="Example: What are the main causes of anemia?"
)

st.button("Send")
