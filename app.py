import streamlit as st

from azure_rag import ask_question


st.set_page_config(
    page_title="AI Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 AI Medical RAG Assistant")

st.caption(
    "WHO Medical Knowledge • Azure AI Search • Gemini"
)

st.warning(
    "This application provides general educational information "
    "and does not replace professional medical advice, diagnosis, "
    "or treatment."
)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("About")

    st.write(
        "This Azure-native RAG chatbot retrieves information "
        "from WHO documents about anaemia, diabetes and "
        "hypertension."
    )

    st.write("**Architecture**")
    st.write("WHO PDFs → OCR → Gemini Embeddings")
    st.write("Azure AI Search → Gemini Answer")

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved WHO sources"):
                for number, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**Source {number}: "
                        f"{source['source']} — "
                        f"Page {source['page']}**"
                    )
                    st.caption(
                        f"Search score: "
                        f"{source['score']:.4f}"
                    )
                    st.write(source["content"])
                    st.divider()


question = st.chat_input(
    "Ask a question about anaemia, diabetes or hypertension"
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching WHO documents and generating an answer..."
        ):
            try:
                result = ask_question(question)

                st.markdown(result["answer"])

                with st.expander(
                    "View retrieved WHO sources"
                ):
                    for number, source in enumerate(
                        result["sources"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Source {number}: "
                            f"{source['source']} — "
                            f"Page {source['page']}**"
                        )
                        st.caption(
                            f"Search score: "
                            f"{source['score']:.4f}"
                        )
                        st.write(source["content"])
                        st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

            except Exception as error:
                error_message = (
                    "The assistant could not process the request. "
                    f"Details: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
