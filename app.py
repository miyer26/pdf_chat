import streamlit as st
import os
import google.generativeai as genai

from dotenv import load_dotenv

from src.create_pdf_vectorstore import get_text_from_pdf, get_text_chunks, create_vectorstore
from src.rag_steps import get_response

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = create_vectorstore(text_chunks)

    response = get_response(user_question, vector_store=vector_store)
    st.success("Done")
    st.write("Reply: ", response["output_text"])


if __name__ == "__main__":
    main()
