import streamlit as st
import os

from dotenv import load_dotenv

from src.create_pdf_vectorstore import get_text_from_pdf, get_text_chunks, create_vectorstore
from src.rag_steps import get_response, get_llm_model, RAGAugmentation
from src.augmented_rag_methods import synth_queries

load_dotenv()
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    vector_store = None
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text, chunk_size=256)
                vector_store = create_vectorstore(text_chunks, hf_token)
                st.session_state.vector_store = vector_store
                st.success("Done")

        
    if st.button("Submit Questions"):
        with st.spinner("Generating..."):
            response = get_response(user_question,
                                    vector_store=st.session_state.vector_store,
                                    n_rank = 3)
            st.write(response)

    if st.button("Synthesize queries"):
        with st.spinner("Generating..."):
            llm = get_llm_model(model_repo="HuggingFaceH4/zephyr-7b-beta", temperature=0.2)
            augmentation = RAGAugmentation()
            response = augmentation.synth_queries(user_question, llm)
            st.write(response)


if __name__ == "__main__":
    main()
