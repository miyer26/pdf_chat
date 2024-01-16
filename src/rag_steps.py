from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
import os

def create_conversational_chain() -> StuffDocumentsChain:
    """
    Creates a conversational chain for an informative assistant using a 
    ChatGoogleGenerativeAI model.

    Returns:
        StuffDocumentsChain: A conversational chain tailored for providing detailed answers
            based on context, with citations. The chain is designed for answering questions
            using a specified LLM model.
    """
 
    prompt_template="""
    Only sse the context below to answer the question provided. If the question is not relevant to the context, explicitly state 
    "This question is is irrelevant to the document".
    Do not make up an answer.

    Context:\n {context} \n
    Question:\n {question}\n

    Only return the answer and nothing else.
    Answer:
    """

    hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    model = HuggingFaceHub(repo_id="google/flan-t5-base",
                           model_kwargs={"temperature":0.3, "max_length":200},
                           huggingfacehub_api_token=hf_token)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_response(user_question: str, vector_store: Chroma):
    """
    Retrieves and generates a response to a user's question using a conversational chain.

    Parameters:
        user_question (str): The user's question for which a response is sought.

        vector_store (Chroma): A vector store containing pre-processed text chunks.

    Returns:
        str: The generated response to the user's question.

    Notes:
        - The function uses the vector_store to retrieve relevant documents for the user's
          question using a specified retriever with search type 'mmr' and search kwargs.
        - A conversational chain is then created using the create_conversational_chain
          function, and the user's question along with the retrieved context is passed to
          the chain for generating a response.
        - Adjust the search_type and search_kwargs in the retriever to customize the document
          retrieval process.
        - Ensure that the vector_store, chain, and retriever are properly configured and
          imported from the relevant modules.
    """

    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k":10})
    context = retriever.get_relevant_documents(user_question)
    chain = create_conversational_chain()
    stuff_answer = chain({"input_documents":context, "question":user_question},
                         return_only_outputs=True)

    return stuff_answer["output_text"]
