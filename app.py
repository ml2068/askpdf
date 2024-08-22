import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=20,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OllamaEmbeddings(model="znbang/bge:large-en-v1.5-f16")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      #save vetorstore
      # knowledge_base.save_local('./datas')
      #loading vetorstore
      # knowledge_base = FAISS.load_local(folder_path='./datas', embeddings=embedding)

      # Prompt
      template = """Use the following pieces of context to answer the question at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      {context}
      Question: {question}
      Helpful Answer:"""
      
      QA_CHAIN_PROMPT = PromptTemplate(
          input_variables=["context", "question"],
          template=template,
      )
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        llm = Ollama(model="llama3.1:8b") 
        qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )
        response = qa_chain.invoke({"query": user_question})
        st.write(response['result'])
    

if __name__ == '__main__':
    main()
