import streamlit as st
import os
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq # Import ChatGroq
from dotenv import load_dotenv     



def main ():
    st.set_page_config(page_title="Chat with PDF")
    st.header("💬 Chat with PDF")

    #Upload File
    pdf = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)

    #Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #Split into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base  = FAISS.from_texts(chunks, embeddings)
        
        user_question = st.text_input("Ask any question about the PDF file:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=4)

            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY not found. Please set it in your .env file.")
                st.stop()
            

            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="deepseek-r1-distill-llama-70b"
            )
            chain = load_qa_chain(llm, chain_type="map_reduce")
            with st.spinner("Generating response..."):
                response = chain.run(input_documents=docs, question=user_question)
            total_chars = sum(len(doc.page_content) for doc in docs)
            st.markdown(f"Total characters being sent to LLM:  :blue[{total_chars}]")
            st.write(response)
        


if __name__ == '__main__':
    main()