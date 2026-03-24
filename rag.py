# ============================================================
# RAG (Retrieval-Augmented Generation) — Step by Step
# ============================================================
# pip install langchain langchain-community langchain-openai chromadb

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────
# 0. Config
# ────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk-..."   # ใส่ API key ของคุณ

# ────────────────────────────────────────────
# 1. เอกสารตัวอย่าง (แทน PDF หรือ database)
# ────────────────────────────────────────────
raw_documents = [
    Document(page_content="""
        LangChain คือ framework สำหรับสร้าง application ที่ใช้ LLM
        รองรับการเชื่อมต่อกับ OpenAI, Anthropic, Google และอื่นๆ
        มี component หลักคือ Chain, Agent, Memory และ Tool
    """, metadata={"source": "langchain_intro"}),

    Document(page_content="""
        RAG ย่อมาจาก Retrieval-Augmented Generation
        เป็นเทคนิคที่ช่วยให้ LLM ตอบคำถามจากเอกสารของเราได้
        ขั้นตอนคือ: โหลดเอกสาร → แบ่ง chunks → embed → เก็บ → ค้นหา → ตอบ
    """, metadata={"source": "rag_intro"}),

    Document(page_content="""
        Vector Database คือฐานข้อมูลที่เก็บข้อมูลในรูปแบบ vector (ตัวเลข)
        ใช้สำหรับค้นหาเนื้อหาที่ "ความหมายใกล้เคียง" กับ query
        ตัวอย่างเช่น Chroma, Pinecone, Weaviate, FAISS
    """, metadata={"source": "vector_db_intro"}),
]

# ────────────────────────────────────────────
# 2. แบ่งเอกสารเป็น Chunks
# ────────────────────────────────────────────
# chunk_size  = ขนาด chunk (ตัวอักษร)
# chunk_overlap = ส่วนที่ทับซ้อนกัน เพื่อไม่ให้ตัดประโยคกลาง
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)
chunks = splitter.split_documents(raw_documents)

print(f"แบ่งได้ {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  chunk[{i}]: {chunk.page_content[:60].strip()}...")

# ────────────────────────────────────────────
# 3. สร้าง Embeddings และเก็บใน Vector Store
# ────────────────────────────────────────────
# Embedding = แปลงข้อความเป็น vector ตัวเลข เพื่อเปรียบเทียบความหมาย
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",   # เก็บลง disk ใช้ซ้ำได้
)

# ────────────────────────────────────────────
# 4. สร้าง Retriever
# ────────────────────────────────────────────
# retriever จะค้นหา chunks ที่ใกล้เคียงกับ query มากที่สุด
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},   # ดึงมา 2 chunks
)

# ────────────────────────────────────────────
# 5. สร้าง Prompt Template
# ────────────────────────────────────────────
template = """ตอบคำถามโดยใช้ข้อมูลจาก context ที่ให้มาเท่านั้น
ถ้าไม่มีข้อมูลใน context ให้บอกว่า "ไม่มีข้อมูลในเอกสาร"

Context:
{context}

คำถาม: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# ────────────────────────────────────────────
# 6. สร้าง LLM
# ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ────────────────────────────────────────────
# 7. ประกอบ RAG Chain
# ────────────────────────────────────────────
def format_docs(docs):
    """รวม chunks ทั้งหมดเป็น string เดียว"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────────
# 8. ทดสอบถามคำถาม
# ────────────────────────────────────────────
questions = [
    "RAG คืออะไร?",
    "Vector Database ใช้ทำอะไร?",
    "LangChain มี component อะไรบ้าง?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")
