from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

def create_conversational_chain() -> StuffDocumentsChain:
    """
    Creates a conversational chain for an informative assistant using a 
    ChatGoogleGenerativeAI model.

    Returns:
        StuffDocumentsChain: A conversational chain tailored for providing detailed answers
            based on context, with citations. The chain is designed for answering questions
            using a specified GoogleGenerativeAI model.

    Notes:
        - The function initializes a ChatGoogleGenerativeAI model configured for informative
          assistance with safety settings to block harassment and hate speech.
        - The prompt_template defines the structure for generating prompts, incorporating
          the context and question placeholders for dynamic input.
        - The chain is loaded using the specified model, chain_type="stuff", and the
          configured prompt template.
    """
    prompt_template="""
    System: You are an informative assistant. Provide a detailed answer based on the 
    context that is provided and provide a citation. If the answer is not present
    in the context, explicitly mention that.
    Context: {context} \n
    Question: {question} \n

    Answer:    
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", 
                                   temperature=0.3,
                                   safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
    ])

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_response(user_question: str, vector_store: Chroma) -> str:
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
    stuff_answer = chain({"context": context, "question": user_question},
                         return_only_outputs=True)
    
    return stuff_answer
