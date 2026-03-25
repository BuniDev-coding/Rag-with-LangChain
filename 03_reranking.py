# ============================================================
# Re-ranking — เรียงลำดับ chunks ใหม่หลัง retrieve
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    persist_directory="./chroma_rerank",
)

# ────────────────────────────────────────────
# 2. Base Retriever — ดึงมาก่อน (k=10)
# ────────────────────────────────────────────
# ดึงมาเยอะๆ ก่อน แล้วค่อย re-rank เอา top 3
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ────────────────────────────────────────────
# 3. Re-ranker (FlashRank — รันบน CPU ได้ ฟรี)
# ────────────────────────────────────────────
reranker = FlashrankRerank(
    top_n=3,   # เอาแค่ top 3 หลัง re-rank
)

# ────────────────────────────────────────────
# 4. ประกอบ ContextualCompressionRetriever
# ────────────────────────────────────────────
# retriever นี้จะ: retrieve → rerank → return top_n
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

# ────────────────────────────────────────────
# 5. เปรียบเทียบผลก่อน/หลัง Re-rank
# ────────────────────────────────────────────
query = "อยากเป็น AI engineer ต้องทำอะไรบ้าง"

print("--- Before Re-rank (top 3 จาก 10) ---")
base_results = base_retriever.invoke(query)
for i, doc in enumerate(base_results[:3]):
    print(f"  [{i+1}] {doc.page_content[:80].strip()}...")

print("\n--- After Re-rank (top 3) ---")
reranked_results = compression_retriever.invoke(query)
for i, doc in enumerate(reranked_results):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"  [{i+1}] score={score:.3f} | {doc.page_content[:80].strip()}...")

# ────────────────────────────────────────────
# 6. RAG Chain
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
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nQ: {query}")
answer = rag_chain.invoke(query)
print(f"A: {answer}")
