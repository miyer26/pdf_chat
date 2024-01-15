from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_genai import ChatGoogleGenerativeAI

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

def get_response(user_question, vector_store):
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k":10})
    context = retriever.get_relevant_documents(user_question)
    chain = create_conversational_chain()
    stuff_answer = chain({"context": context, "question": user_question},
                         return_only_outputs=True)
    return stuff_answer
