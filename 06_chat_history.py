# ============================================================
# Chat History — RAG ที่จำบทสนทนาก่อนหน้าได้
# ============================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ────────────────────────────────────────────
# 1. โหลดเอกสาร
# ────────────────────────────────────────────
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_chat",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ────────────────────────────────────────────
# 2. Contextualize Question
# ใช้ history เพื่อ rewrite คำถามให้ standalone
# ────────────────────────────────────────────
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """พิจารณาประวัติการสนทนาและคำถามล่าสุด
แปลงคำถามให้เป็น standalone question ที่เข้าใจได้โดยไม่ต้องอาศัย context ก่อนหน้า
ถ้าคำถามชัดเจนอยู่แล้ว ให้คืนค่าเดิม ห้ามตอบคำถาม"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

# ────────────────────────────────────────────
# 3. QA Prompt
# ────────────────────────────────────────────
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """ตอบคำถามโดยใช้ข้อมูลจากเอกสารที่ให้มา
ถ้าไม่มีข้อมูล ให้บอกว่า "ไม่พบข้อมูลในเอกสาร"

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ────────────────────────────────────────────
# 4. RAG Chain with History
# ────────────────────────────────────────────
def rag_with_history(question: str, chat_history: list) -> str:
    # Step 1: rewrite คำถามถ้ามี history
    if chat_history:
        standalone_question = contextualize_chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })
        print(f"  [rewritten] {standalone_question}")
    else:
        standalone_question = question

    # Step 2: retrieve
    context_docs = retriever.invoke(standalone_question)
    context = format_docs(context_docs)

    # Step 3: ตอบ
    answer = (qa_prompt | llm | StrOutputParser()).invoke({
        "input": question,
        "chat_history": chat_history,
        "context": context,
    })

    return answer

# ────────────────────────────────────────────
# 5. ทดสอบแบบ multi-turn
# ────────────────────────────────────────────
chat_history = []

questions = [
    "เอกสารนี้พูดถึงอะไร?",
    "ขยายความเพิ่มเติมได้มั้ย?",      # "มัน" = เอกสาร จาก history
    "สรุปสั้นๆ ให้หน่อย",
]

for q in questions:
    print(f"\nYou: {q}")
    answer = rag_with_history(q, chat_history)
    print(f"Bot: {answer}")

    # เพิ่ม history
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=answer))
