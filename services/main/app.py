from flask import Flask, request

import os
import requests

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain


app=Flask(__name__)

# res = requests.post("")


loader = PyPDFLoader("src.pdf")
document = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()
vectorstore = Weaviate.from_documents(docs, embeddings, weaviate_url=os.environ['WEAVIATE_API_URL'], by_text=False)

PROMPT="""The AI, a state-of-the-art language model devised by OpenAI, is highly versatile in its operations. From providing simple answers to engaging in in-depth discussions on a vast array of topics, the AI's design is well-suited to these tasks. As a language model, it generates text resembling human conversation, guided by the prompts it receives. This capability allows it to engage in organic conversations and deliver responses that align seamlessly with the context at hand.

The AI is in a constant state of learning and growth, its capabilities ever-expanding. It has the aptitude to ingest and understand vast amounts of text, and can leverage this acquired information to provide accurate and thoughtful responses to a diverse range of inquiries. Additionally, the Assistant has the ability to generate unique text based on received prompts, making it possible for it to partake in conversations and offer detailed descriptions and explanations on a multitude of topics.

In essence, the AI serves as a robust instrument capable of aiding with a wide range of tasks, supplying valuable insights and information on a vast array of subjects. Whether you need help with a particular question or simply want to engage in a discussion on a specific topic, the Assistant stands ready to assist.

{context}
User: {question}
AI:"""


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
