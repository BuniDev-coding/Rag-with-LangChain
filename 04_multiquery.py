# ============================================================
# Multi-query Retriever — แปลงคำถามเดียวเป็นหลายคำถาม
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

load_dotenv()

# ────────────────────────────────────────────
# 1. โหลดเอกสารและสร้าง Vector Store
# ────────────────────────────────────────────
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_multiquery",
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ────────────────────────────────────────────
# 2. Multi-query Retriever
# ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# เปิด logging เพื่อดูว่า LLM สร้างคำถามอะไรบ้าง
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    # LLM จะสร้างคำถามใหม่ 3 คำถาม (default)
    # แล้วดึง chunks จากทุกคำถาม และ deduplicate อัตโนมัติ
)

# ────────────────────────────────────────────
# 3. ทดสอบดูว่าดึงได้ chunks อะไรบ้าง
# ────────────────────────────────────────────
query = "อยากเป็น AI engineer ต้องทำอะไรบ้าง"

print(f"Q: {query}\n")
results = multi_retriever.invoke(query)
print(f"\nดึงได้ {len(results)} chunks (หลัง deduplicate)")
for i, doc in enumerate(results):
    print(f"  [{i+1}] {doc.page_content[:80].strip()}...")

# ────────────────────────────────────────────
# 4. เปรียบเทียบกับ Base Retriever
# ────────────────────────────────────────────
print("\n--- Base Retriever (คำถามเดียว) ---")
base_results = base_retriever.invoke(query)
print(f"ดึงได้ {len(base_results)} chunks")

print("\n--- Multi-query Retriever ---")
print(f"ดึงได้ {len(results)} chunks (ครอบคลุมกว่า)")

# ────────────────────────────────────────────
# 5. RAG Chain
# ────────────────────────────────────────────
template = """ตอบคำถามโดยใช้ข้อมูลจากเอกสารที่ให้มา
ถ้าไม่มีข้อมูล ให้บอกว่า "ไม่พบข้อมูลในเอกสาร"

Context:
{context}

คำถาม: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": multi_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nQ: {query}")
answer = rag_chain.invoke(query)
print(f"A: {answer}")
