from flask import Flask, request

import os
import requests

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from langchain.vectorstores import Weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


app=Flask(__name__)

# res = requests.post("")


loader = PyPDFLoader("src.pdf")
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(docs, embeddings, weaviate_url=os.environ['WEAVIATE_API_URL'], by_text=False)

PROMPT="""The Aide is a sophisticated language model devised by OpenAI.

The Aide's design enables it to support an array of duties, from replying to straightforward inquiries to offering comprehensive clarifications and discourse on an extensive variety of subjects. Being a language model, the Aide has the capacity to generate text that mirrors human conversation based on the prompts it receives, allowing it to partake in naturalistic dialogues and offer replies that are logical and pertinent to the current discussion.

The Aide is perpetually learning and advancing, with its abilities ceaselessly developing. It possesses the capacity to digest and comprehend substantial volumes of text, and can utilize this acquired knowledge to give precise and insightful answers to a broad spectrum of queries. In addition, the Aide can produce its own text depending on the prompts it gets, enabling it to participate in dialogues and provide explanations and portrayals on an extensive variety of topics.

In a nutshell, the Aide is a potent tool that can assist with numerous duties and provide significant knowledge and data on a broad variety of subjects. Whether you require assistance with a specific query or merely wish to have a discussion on a certain subject, the Aide is here to support.

{context}
User: {question}
Aide:"""


prompt_template = PromptTemplate(
    template=PROMPT,
    input_variables=["context", "question"]
)


@app.route("/api/query", methods=["POST"])
def query():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    question_answer = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), vectorstore.as_retriever(), 
                                                            combine_docs_chain_kwargs={'prompt': prompt_template},
                                                            memory=memory)
    question = {"question": request.form["query"]}
    result = question_answer(question)
    return {"answer":result["answer"]}


def create_app():
    return app
