from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from urllib.request import urlretrieve
## i have downloaded ollam on my laptop and i have used this model for embedding
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
## using Chroma to save in local data base instead of memory
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)
## chunk the data
text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size =700,
    chunk_overlap = 50
    )

## check if the data base not create then go inside the below 
if add_documents:
    documents = []
    ids = []
    
    loader = PyPDFDirectoryLoader("files-pdf/")
    docs_before_split = loader.load()
    ## we split the data as the embedding model does not accept a big size of data
    docs_after_split = text_splitter.split_documents(docs_before_split)
   
## save the document spliter into the vectore data base
if add_documents:
    vectorstore = Chroma.from_documents(docs_after_split, embeddings, persist_directory="chrome_langchain_db")
## retrive the data from the vector data base
vector_store = Chroma(persist_directory="chrome_langchain_db", embedding_function=embeddings)   
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)