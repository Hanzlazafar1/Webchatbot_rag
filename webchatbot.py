import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
import os
from langchain_community.vectorstores import FAISS
from langchain_community import  GoogleGenerativeAIEmbeddings
# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load LLM
llm = ChatGroq(
    model_name="llama-3.2-11b-text-preview",
    temperature=0.3
)

# Load the URL and process the data
URLs = ['https://devvibe.com/']
loaders = UnstructuredURLLoader(urls=URLs)
data = loaders.load()

# Split data into chunks
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=300)
text_chunks = text_splitter.split_documents(data)

# Load embeddings model
# Initialize the SentenceTransformer model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents( text_chunks, embeddings)

 


# Create the QA chain
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit UI
st.set_page_config(page_title="DevVibe Services QA", layout="wide")

# Header
st.title("📘 DevVibe Services QA System")
st.write("Ask a question about the services provided by DevVibe!")

# Input section
question = st.text_input("Enter your question", placeholder="e.g., What are the services DevVibe provides?")

# Button to get the answer
if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Fetching answer..."):
            try:
                result = chain({"question": question}, return_only_outputs=True)
                answer = result['answer']
                sources = result.get('sources', 'No sources found.')
                
                # Display the answer and sources
                st.success("### Answer:")
                st.write(answer)

                st.info("### Sources:")
                st.write(sources)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid question.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io/) and [LangChain](https://langchain.readthedocs.io/).")
