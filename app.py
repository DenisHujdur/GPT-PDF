from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openai


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")
    # st.image('./header.png')
    # Insert a chat message container.

    # uploading the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #split into chunks
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)

        #create embeddings
        embeddings=OpenAIEmbeddings()
        knowledgebase=faiss.FAISS.from_texts(chunks,embeddings)
    
        user_question=st.text_input("Ask a question about your PDF")
        if user_question:
            docs=knowledgebase.similarity_search(user_question)

            llm=openai.OpenAI()
            chain=load_qa_chain(llm, chain_type="stuff")
            response=chain.run(input_documents=docs, question=user_question)

            st.write(response)



if __name__=='__main__':
    main()


