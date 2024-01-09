import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

def get_text(pdfs):
    text=""
    for pdf in pdfs:
        doc=PdfReader(pdf)
        for page in doc.pages:
            text+=page.extract_text()
    return text

def get_chunks(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=200,length_function=len)
    return splitter.split_text(text)
def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    store=FAISS.from_texts(chunks,embeddings)
    store.save_local('faiss_index')

def get_chain():
    template='''
    Answering the question as detailed as possible from the provided context. You have to give answer based on your provided context
    and try to give answer from provided context.
    Context:\n {context}
    Question:\n{question}
    '''
    llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)
    prompt=PromptTemplate(template=template,input_variables=['context','question'])
    chain=load_qa_chain(llm,chain_type='stuff',prompt=prompt)
    return chain

def get_output(question):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db=FAISS.load_local('faiss_index',embeddings)
    docs=new_db.similarity_search(question)
    chain=get_chain()
    return chain({'input_documents':docs,'question':question},return_only_outputs=True)



def main():
    load_dotenv()
    st.header("Gemini Multi PDFs Chatting :books:")
    with st.sidebar:
        st.header("Upload your PDFs")
        pdf_files=st.file_uploader("Add your PDF files here to fuel your queries",type=['.pdf'],accept_multiple_files=True)
        if pdf_files:
            if st.button("Upload"):
                with st.spinner("Processing....."):
                    text=get_text(pdf_files)
                    chunks=get_chunks(text)
                    get_vector_store(chunks)
                    st.write("Uploaded successfully")
    prompt=st.text_input("Enter your question or prompt to extract insights from your PDFs")
    if prompt:
        response=get_output(prompt)
        st.write(response['output_text'])
    st.markdown(
    """
    <footer style="position: fixed; bottom: 0; width: 100%; text-align: left; padding: 10px;">
        Developed by Abdullah Sajid
    </footer>
    """,
    unsafe_allow_html=True
)

if __name__=='__main__':
    main()
