import os
from typing_extensions import Concatenate
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains.summarize import load_summarize_chain
from constants import open_ai_key
from PyPDF2 import PdfReader
from langchain.document_loaders import YoutubeLoader
import streamlit as st


os.environ['OPENAI_API_KEY']= open_ai_key
# streamlit framework
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




def page_grammar_check():
    
    st.title('Grammar Genius')
    session_state = SessionState(command_executed=False)
    content= st.text_input('Enter the content')
 
    if st.button("Execute") and not session_state.command_executed:
        if content:
            llm = ChatOpenAI(temperature=0)


            first_prompt = ChatPromptTemplate.from_template(
                "You are an English grammar professor \
                Evaluate the the text delimited by delimitted by ``  and find out the grammatical and spelling mistakes in it . If any sentence is poorly framed list that out as well.\
                List all the the findings in structured way  ``{content}``"
                )

            chain_one = LLMChain(llm=llm, prompt=first_prompt,output_key="Mistakes")
            # second_prompt = ChatPromptTemplate.from_template("Can you rewrite the above paragraph with all the following mistakes corrected. Mistakes are delimitted by ##  ##{Mistakes}##")

            # chain_two = LLMChain(llm=llm, prompt=second_prompt,output_key="correction")
            overall_chain = SequentialChain( chains=[chain_one],input_variables=["content"],output_variables=["Mistakes"],verbose=True)
            st.write(overall_chain(content))
            session_state.command_executed = True
        

def page_doc_qa():
    st.title(" Document Q&A")
    st.text("Query any PDF Document")
    session_state2 = SessionState(command_executed=False)
    uploaded_file= st.file_uploader('Upload your pdf file',type="pdf")
    input_text= st.text_input("Enter your Query")
    if st.button("Process") and not session_state2.command_executed:
        
        
        Pdf = PdfReader(uploaded_file)
        raw_text= ''
        for i, page in enumerate(Pdf.pages):
            content= page.extract_text()
            if content:
                raw_text+= content

        text_splitter= CharacterTextSplitter(separator='\n',chunk_size=800,chunk_overlap=100,length_function=len)
        texts = text_splitter.split_text(raw_text)
        embeddings = OpenAIEmbeddings()
        document_db= FAISS.from_texts(texts,embeddings)
        chain = load_qa_chain(OpenAI(),chain_type='stuff')
        docs= document_db.similarity_search(input_text)
        st.write(chain.run(input_documents=docs,question='query'))
        session_state2.command_executed = True

def page_youtube_summarizer():
    
    st.title("Youtube Video Summarizer")
    session_state3 = SessionState(command_executed=False)
    url= st.text_input("Enter the URl of Youtube video")
    llm= OpenAI()
    if st.button("Summarize") and not session_state3.command_executed:
        loader= YoutubeLoader.from_youtube_url(url,add_video_info=True)
        result= loader.load()
        st.write(print(f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long"))
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
        text= text_splitter.split_documents(result)
        chain= load_summarize_chain(llm,chain_type='map_reduce',verbose=False)
        st.header("Here is a Summary of the video")
        st.write(chain.run(text))


st.sidebar.title("Choose an Option")
menu = ["Grammar Check", "Query a Document","Youtube Video Summarizer"]
choice = st.sidebar.selectbox("Select a page", menu)

if choice == "Grammar Check":
    page_grammar_check()
elif choice == "Query a Document":
    page_doc_qa()

elif choice == "Youtube Video Summarizer":
    page_youtube_summarizer()



#read text from pdf



