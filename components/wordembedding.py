import os 
import dotenv
import torch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

dotenv.load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
vector_db = os.path.join(current_dir, "db")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                                                          
def convert_to_float32(embeddings):
    return torch.tensor(embeddings, dtype=torch.float32)

def create_vectordb(file_content):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([file_content])

    embeddings = convert_to_float32(
    embedding_model.embed_documents([doc.page_content for doc in docs])
    )

    db = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory='db')
    db.add_documents(docs)
    db.persist()
    return len(docs)