from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from urllib.request import urlretrieve

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size =700,
    chunk_overlap = 50
    )


if add_documents:
    documents = []
    ids = []
    
    loader = PyPDFDirectoryLoader("files-pdf/")
    docs_before_split = loader.load()
    docs_after_split = text_splitter.split_documents(docs_before_split)
   


if add_documents:
    vectorstore = Chroma.from_documents(docs_after_split, embeddings, persist_directory="chrome_langchain_db")

vector_store = Chroma(persist_directory="chrome_langchain_db", embedding_function=embeddings)   
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)