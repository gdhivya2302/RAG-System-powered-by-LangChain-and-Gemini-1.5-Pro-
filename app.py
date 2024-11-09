import streamlit as st
import nltk
nltk.download('punkt')
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key
with open("gemini_key.txt") as f:
    key = f.read()

# Streamlit title
st.title('Query me about the "Leave No Context Behind paper by Google."')

# User input
user_prompt = st.text_area("What's your question?")

# Button click event
if st.button("Query"):
    # Load and split the document
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader('RAG-ResearchPaper-2404.07143.pdf')
    pages = loader.load_and_split()

    # Split the document into chunks
    from langchain_text_splitters import NLTKTextSplitter
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Set up embedding model
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, model='models/embedding-001')

    # Initialize Chroma with a persistence directory
    from langchain_community.vectorstores import Chroma
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db.persist()  # Persist the data explicitly

    # Reconnect to the persisted Chroma vector store
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

    # Convert db_connection to retriever object
    retriever = db_connection.as_retriever(search_kwargs={'k': 5})

    # Define chat template
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a helpful AI bot.
        You take the context and question from the user.
        Your answer should be based on the specific context."""),
        HumanMessagePromptTemplate.from_template("""
        Answer the question based on the given context.
        Context: 
        {context}
        
        Question:
        {question}

        Answer:
        """)
    ])

    # Define the chat model
    chat_model = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-pro-latest")

    # Output parser
    output_parser = StrOutputParser()

    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG chain
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

    # Generate response
    if user_prompt:
        response = rag_chain.invoke(user_prompt)
        st.write(response)
