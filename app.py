import streamlit as st  # Importing Streamlit for creating the web app
import os  # Importing os for environment variable handling
import time  # Importing time for measuring response time
from langchain_groq import ChatGroq  # Importing ChatGroq for the language model
from langchain_core.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate for prompt creation
from langchain_community.vectorstores import FAISS  # Importing FAISS for vector storage
from langchain_community.document_loaders import PyPDFLoader  # Importing PyPDFLoader for loading PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter for splitting text
from langchain_ai21.embeddings import AI21Embeddings  # Importing AI21Embeddings for creating embeddings
from langchain.chains import create_retrieval_chain  # Importing create_retrieval_chain for retrieval chain creation
from langchain.chains.combine_documents import create_stuff_documents_chain  # Importing create_stuff_documents_chain for document chaining
from dotenv import load_dotenv  # Importing load_dotenv for loading environment variables

# Load environment variables from a .env file
load_dotenv()

# Set AI21 API key and Groq API key from environment variables
os.environ['AI21_API_KEY'] = os.getenv('AI21_API_KEY')
groq_api = os.environ.get('GROQ_API')

# Set the title of the Streamlit app
st.title("Scaler Inquiry Bot")

# Initialize the language model with the Groq API key and model name
llm = ChatGroq(groq_api_key=groq_api, model_name='Llama3-8b-8192')

# Define the prompt template for the language model
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

# Function to handle vector embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize AI21 embeddings
        st.session_state.embeddings = AI21Embeddings()
        # Load the PDF document
        st.session_state.loader = PyPDFLoader("Scaler_data.pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        # Split the documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        # Create vector embeddings using FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector AI21 embeddings

# Input field for the user prompt
prompt1 = st.text_input("Input your prompt here")

# Button to trigger the vector embedding process
if st.button("Click here after writing your query"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Check if the vectors are initialized before using them
if prompt1 and "vectors" in st.session_state:
    # Create the document chain with the language model and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Get the retriever from the vector store
    retriever = st.session_state.vectors.as_retriever()
    # Create the retrieval chain with the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Measure the start time
    start = time.process_time()
    # Invoke the retrieval chain with the user prompt
    response = retrieval_chain.invoke({'input': prompt1})
    # Print the response time
    print("Response time:", time.process_time() - start)
    # Display the response in the Streamlit app
    st.write(response['answer'])
else:
    if not prompt1:
        st.write("Please enter a prompt.")
    else:
        st.write("Please wait")



