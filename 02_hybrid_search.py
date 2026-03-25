# ============================================================
# Hybrid Search — BM25 + Vector Search
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ────────────────────────────────────────────
# 1. โหลดเอกสาร
# ────────────────────────────────────────────
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"chunks ทั้งหมด: {len(chunks)}")

# ────────────────────────────────────────────
# 2. สร้าง Retriever 2 ตัว
# ────────────────────────────────────────────

# BM25 Retriever — ค้นหาแบบ keyword (ไม่ต้อง embed)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3   # ดึง 3 chunks

# Vector Retriever — ค้นหาแบบ semantic
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_hybrid",
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ────────────────────────────────────────────
# 3. รวมเป็น Hybrid (EnsembleRetriever)
# ────────────────────────────────────────────
# weights = สัดส่วน [bm25, vector]
# 0.5, 0.5 = ให้น้ำหนักเท่ากัน
# ปรับได้ตามลักษณะข้อมูล เช่น ถ้าข้อมูลเป็นเอกสารทางเทคนิคมีชื่อเฉพาะ → bm25 มากกว่า
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

# ────────────────────────────────────────────
# 4. ทดสอบ Retriever เปรียบเทียบกัน
# ────────────────────────────────────────────
query = "อยากเป็น AI engineer ต้องทำอะไรบ้าง"

print("\n--- BM25 Results ---")
bm25_results = bm25_retriever.invoke(query)
for doc in bm25_results:
    print(f"  {doc.page_content[:80].strip()}...")

print("\n--- Vector Results ---")
vector_results = vector_retriever.invoke(query)
for doc in vector_results:
    print(f"  {doc.page_content[:80].strip()}...")

print("\n--- Hybrid Results ---")
hybrid_results = hybrid_retriever.invoke(query)
for doc in hybrid_results:
    print(f"  {doc.page_content[:80].strip()}...")

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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nQ: {query}")
answer = rag_chain.invoke(query)
print(f"A: {answer}")
