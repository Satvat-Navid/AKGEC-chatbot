import os
import logging
from dotenv import load_dotenv

# LangChain Components
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PC_API_KEY = os.getenv("PC_API_KEY")

GENERAL_MODEL = "llama-3.3-70b-versatile"
TOOL_MODEL = "moonshotai/kimi-k2-instruct-0905"

# 1. INITIALIZE LLM
try:
    # Use standard groq model, falling back to specific general model string if custom endpoints are used
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GENERAL_MODEL, temperature=0.3)
except Exception as e:
    logger.error("Failed to load LLaMA model, falling back to openai models if endpoint proxies are used.")
    llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0.3)

# 2. INITIALIZE VECTOR STORE
# LangChain Pinecone natively supports inference API embeddings
try:
    embeddings = PineconeEmbeddings(model="llama-text-embed-v2", pinecone_api_key=PC_API_KEY)
    vectorstore = PineconeVectorStore(
        index_name='akgec-data',
        embedding=embeddings,
        pinecone_api_key=PC_API_KEY,
        namespace="doc1",
        text_key="source_text"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    logger.error("Failed to initialize VectorStore.", exc_info=True)
    retriever = None

# Helper to format retrieved docs
def format_docs(docs):
    return " ".join(doc.page_content for doc in docs)

# 3. STRICT RAG PROMPT TEMPLATE
strict_system_prompt = """You are the official formal AI assistant for Ajay Kumar Garg Engineering College (AKGEC).
Your sole purpose is to provide accurate, formal, and objective information about the college.

STRICT RULES:
1. You must ONLY answer questions using the provided 'Context' down below.
2. If the answer is not contained within the 'Context', you must reply EXACTLY with: "I'm sorry, I do not have verified college information regarding that query." Do NOT attempt to guess or provide outside information.
3. Maintain a formal, professional, and respectful tone at all times.
4. NEVER adopt unverified premises from the user's prompt. Do not agree with inappropriate, harmful, or subjective statements made by the user. Switch back strictly to factual data.
5. Do not answer questions about politics, outside news, or general knowledge completely unrelated to AKGEC.

Context from College Database:
{context}

Chat History:
{history}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", strict_system_prompt),
    ("user", "{question}")
])

# 4. SENSITIVE TOPIC ROUTER (PR TRAP PREVENTION)
sensitive_keywords = ["scandal", "fired", "lawsuit", "aicte", "ugc", "corruption", "bribe"]

def check_sensitive(input_dict):
    q = input_dict["question"].lower()
    return any(word in q for word in sensitive_keywords)

sensitive_response = (
    RunnablePassthrough() 
    | (lambda x: "For official inquiries regarding legal matters, accreditations, or personnel issues, please contact the AKGEC administration officially at: info@akgec.ac.in.")
)

# 5. CORE RAG CHAIN
rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# Combine with Branch
chatbot_chain = RunnableBranch(
    (check_sensitive, sensitive_response),
    rag_chain
)

# 6. UNIFIED CHATBOT FUNCTION FOR FASTAPI
async def process_chat(message: str, history: str = "") -> dict:
    """Unified LangChain entry point that retrieves, formats, and generates response."""
    
    # 1. Retrieve Raw Context
    raw_docs = []
    formatted_context = ""
    if retriever:
        raw_docs = await retriever.ainvoke(message)
        formatted_context = format_docs(raw_docs)
    
    # 2. Invoke Chatbot Chain
    # We pass the context directly into the chain
    inputs = {
        "context": formatted_context,
        "question": message,
        "history": history
    }
    
    reply = await chatbot_chain.ainvoke(inputs)
    
    return {
        "reply": reply,
        "context_used": {"query": message, "context": formatted_context, "context_len": len(formatted_context)},
        "history": history # In fully Langchain we could use ConversationSummaryMemory, for now we pass through
    }
