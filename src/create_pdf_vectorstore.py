from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Union

def get_text_from_pdf(pdf_docs: List[Union[str, bytes]]) -> str:
    """
    Extracts text content from a list of PDF documents.

    Parameters:
        pdf_docs (List[Union[str, bytes]]): A list containing file paths (as strings)
            or actual PDF content (as bytes).

    Returns:
        str: A concatenated string of text content extracted from all pages of the
             input PDF documents.

    Notes:
        - This function utilizes the PdfReader class to read PDF documents. Make sure
          the PdfReader class is appropriately configured or imported from the relevant
          PDF processing library.
        - If the input PDF documents contain multiple pages, the text from all pages
          is concatenated into a single string.
        - The output string may include formatting artifacts from the PDF content.

    Example:
        >>> pdf_files = ['/path/to/file1.pdf', '/path/to/file2.pdf']
        >>> text_content = get_text_from_pdf(pdf_files)
        >>> print(text_content)
    """

    text=""

    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text: str, chunk_size: int=5000, chunk_overlap: int=1000) -> str:
    """
    Splits a given text into chunks using a RecursiveCharacterTextSplitter.

    Parameters:
        text (str): The input text to be split into chunks.

        chunk_size (int, optional): The desired size of each text chunk. Defaults to 5000.

        chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 1000.

    Returns:
        List[str]: A list of text chunks.

    Notes:
        - The function uses a RecursiveCharacterTextSplitter to split the input text
          into chunks based on the specified parameters.
        - Adjust the chunk_size and chunk_overlap parameters of the text_splitter
          according to your requirements.
        - The output is a list of text chunks, and the length of each chunk may vary
          based on the specified parameters.

    Example:
        >>> input_text = "This is a long piece of text that needs to be split into chunks."
        >>> chunks = get_text_chunks(input_text)
        >>> print(chunks)
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks: str,
                       hf_token: str,
                       embedding_model: str = "BAAI/bge-base-en-v1.5",
                       ) -> Chroma:
    """
    Creates a vector store using text chunks and an embedding model.

    Parameters:
        text_chunks (List[Document]): A list of Document objects representing text chunks.

        emb_model (str, optional): The path or identifier of the pre-trained embedding model.
            Defaults to "models/embedding-001".

    Returns:
        Chroma: A vector store generated from the input text chunks using the specified
            embedding model.
    """

    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key = hf_token,
        model_name = embedding_model
    )
    
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embedding)

    return vector_store
