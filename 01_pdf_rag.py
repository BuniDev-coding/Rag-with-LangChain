# ============================================================
# RAG จาก PDF จริงๆ
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ────────────────────────────────────────────
# วิธีที่ 1: โหลด PDF ไฟล์เดียว
# ────────────────────────────────────────────
# loader = PyPDFLoader("./documents/my_file.pdf")
# docs = loader.load()

# ────────────────────────────────────────────
# วิธีที่ 2: โหลด PDF ทุกไฟล์ในโฟลเดอร์ 
# ────────────────────────────────────────────
# สร้างโฟลเดอร์ documents/ แล้วใส่ PDF ที่ต้องการ
loader = DirectoryLoader(
    "./documents",          # โฟลเดอร์เก็บ PDF
    glob="**/*.pdf",        # โหลดทุก .pdf ในโฟลเดอร์
    loader_cls=PyPDFLoader,
)
docs = loader.load()

print(f"โหลดได้ {len(docs)} หน้า")
# แต่ละ doc = 1 หน้าของ PDF
# doc.metadata มี source (ชื่อไฟล์) และ page (เลขหน้า)
for doc in docs[:3]:
    print(f"  ไฟล์: {doc.metadata['source']} | หน้า: {doc.metadata['page']}")
    print(f"  เนื้อหา: {doc.page_content[:80].strip()}...")

# ────────────────────────────────────────────
# แบ่ง chunks
# ────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"\nแบ่งได้ {len(chunks)} chunks จาก {len(docs)} หน้า")

# ────────────────────────────────────────────
# Embed และเก็บใน Chroma
# ────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_pdf",   # แยก db จากอันก่อน
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ────────────────────────────────────────────
# RAG Chain
# ────────────────────────────────────────────
template = """ตอบคำถามโดยใช้ข้อมูลจากเอกสารที่ให้มา
ถ้าไม่มีข้อมูล ให้บอกว่า "ไม่พบข้อมูลในเอกสาร"
ระบุด้วยว่าข้อมูลมาจากหน้าไหน

Context:
{context}

คำถาม: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    # รวม content พร้อมบอก source และ page
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')} หน้า {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ────────────────────────────────────────────
# ทดสอบ
# ────────────────────────────────────────────
question = "เอกสารนี้พูดถึงอะไร?"   # เปลี่ยนตามเนื้อหา PDF ของคุณ
print(f"\nQ: {question}")
answer = rag_chain.invoke(question)
print(f"A: {answer}")
