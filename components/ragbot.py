import os
import sys
sys.path.append("path_to_your_installed_packages")
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env
load_dotenv()

# persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db")

# Creates a new embedding model
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
db = Chroma(embedding_function=embedding_model,persist_directory=persistent_directory)

# Creates  a retriver
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.6}
)

# Creates a new llm model 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

#contextualize prompt. It helps the llm to reformulate the question(user's query) to give a standalone question from
#chat history
contextualize_prompt = (
    "Reformulate the latest user question into a fully self-contained question "
    "that can be understood without any prior context. Ensure all necessary details "
    "from the chat history are included in the question itself. "
    "Do NOT answer the question, just return the reformulated version. "
    "If the question is already standalone, return it as is."
)

#template for the context
contextualize_prompt_template = ChatPromptTemplate.from_messages([
    ("system",contextualize_prompt),
    (MessagesPlaceholder("chat_history")),
    ("human","{input}")
])

# creates a history aware retriver to reformat the question based on the chat history
History_aware_retriver = create_history_aware_retriever(llm, retriever, contextualize_prompt_template)

#answer prompt forces the ai to answer the questions from the given context, so that it does not hallucinate the answers
answer_prompt = (    
    "You are an assistant for question-answering tasks. "
    "only use the provided context to answer the question. "
    "If you are unsure or the context doesn't provide an answer, respond with: 'I don't know'. "
    "Use three sentences maximum and keep the answer concise."
    "However you can greet the users and assist with questions."
    "\n\n"
    "{context}"
)

#template for the answer prompt
answer_prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system",answer_prompt),
    (MessagesPlaceholder("chat_history")),
    ("human","{input}")
    ])

question_answer_chain = create_stuff_documents_chain(llm=llm,prompt=answer_prompt_template)
rag_chain = create_retrieval_chain(History_aware_retriver,question_answer_chain)

def chat():
    print("Start chatting with bot. Enter 'exit' to quit")
    #Stores all the chat history
    chat_history = []
    while True:
        query = input("you: ")
        if query == "exit":
            break
        #process the query through chain
        result = rag_chain.invoke({"input":query,"chat_history":chat_history})
        print(f"ai: {result['answer']}")
        #updates the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    chat()

def get_chatbot_response(user_input,chat_history):
    result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    return result["answer"]