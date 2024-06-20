from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from llama_cpp import Llama
from typing import Optional, Dict
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from llama_cpp import Llama
from typing import Optional, Dict
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain import hub
# PDF 파일 경로 설정
pdf_directory = './pdf'
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# 모든 PDF 파일의 텍스트 추출
all_texts = ""
for pdf_path in pdf_files:
    pdf_document = fitz.open(pdf_path)
    pdf_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text()
    all_texts += pdf_text

# 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_text(all_texts)
documents = [Document(page_content=text) for text in texts]
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs = {'device': 'cpu'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
    encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
)
# 벡터 저장소 생성
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents, embeddings)


# 벡터 저장소 경로 설정
## 현재 경로에 'vectorstore' 경로 생성
vectorstore_path = 'vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)

# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=vectorstore_path)
# 벡터스토어 데이터를 디스크에 저장
vectorstore.persist()
from langchain_community.chat_models import ChatOllama

# Ollama 를 이용해 로컬에서 LLM 실행
## llama3-ko-instruct 모델 다운로드는 Ollama 사용법 참조
model = ChatOllama(model="llama3:8b", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import datetime

time = datetime.datetime.now()
bazi = None
template = (
    "You are a great translator from English to Korean,"
    "You noly translate user input,"
    "No additional information is provided,"
    "you do not respond to the user input,: \n{input}"
    )


prompt = ChatPromptTemplate.from_template(template)
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
