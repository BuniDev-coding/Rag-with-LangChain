# ============================================================
# Contextual Compression — ตัด chunk ให้เหลือแค่ส่วนที่เกี่ยวข้อง
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
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
    persist_directory="./chroma_compress",
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ────────────────────────────────────────────
# 2. LLM Extractor — ตัดเฉพาะส่วนที่เกี่ยวข้อง
# ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLMChainExtractor จะส่ง chunk + คำถาม ให้ LLM แล้วให้ตอบ
# เฉพาะข้อความที่เกี่ยวข้องกับคำถาม
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# ────────────────────────────────────────────
# 3. เปรียบเทียบก่อน/หลัง Compression
# ────────────────────────────────────────────
query = "ใส่คำถามที่ต้องการทดสอบ"

print("--- Before Compression ---")
base_results = base_retriever.invoke(query)
total_chars_before = sum(len(doc.page_content) for doc in base_results)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {len(doc.page_content)} ตัวอักษร: {doc.page_content[:80].strip()}...")
print(f"  รวม: {total_chars_before} ตัวอักษร")

print("\n--- After Compression ---")
compressed_results = compression_retriever.invoke(query)
total_chars_after = sum(len(doc.page_content) for doc in compressed_results)
for i, doc in enumerate(compressed_results):
    print(f"  [{i+1}] {len(doc.page_content)} ตัวอักษร: {doc.page_content[:80].strip()}...")
print(f"  รวม: {total_chars_after} ตัวอักษร (ลดลง {total_chars_before - total_chars_after} ตัวอักษร)")

# ────────────────────────────────────────────
# 4. RAG Chain
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
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nQ: {query}")
answer = rag_chain.invoke(query)
print(f"A: {answer}")
