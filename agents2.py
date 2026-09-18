from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC as Pinecone

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """You are a dedicated chatbot designed to assist users of 
Ajay Kumar Garg Engineering College (AKGEC).Please ensure that 
responses are strictly relevant to the college context; 
avoid answering questions unrelated or outside the scope 
of the institution.Provide clear, concise, and well-structured information by 
adhering to the following guidelines:Emphasize key points in bold.Use italics to 
highlight important terms or phrases.Organize lengthy responses with appropriate headings. 
Present multiple items using bullet points or numbered lists.
Incorporate tables where they enhance clarity and understanding.
Use the retrive_context tool to fetch relevant information from the vector database when necessary.
If the user query is not related to AKGEC, politely inform them that you can only provide information related to the college and suggest they ask questions within that context.
"""

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PC_API_KEY = os.getenv("PC_API_KEY")

pc = Pinecone(api_key=PC_API_KEY)
index = pc.Index('akgec-data')

# Pinecone Vector db search
@tool
def retrive_context(query: str):
    """Funtion to retrive data from vector database using similarity search with the 
    provided query and return the context along with the query and context length."""
    k = 2
    try:
        query_embedding = pc.inference.embed(
            # model="multilingual-e5-large",
            model="llama-text-embed-v2",
            inputs=[query],
            parameters={
                "input_type": "query"
            }
        )
        results = index.query(
            namespace="doc1",
            # namespace="fit-markdown-data",
            vector=query_embedding[0].values,
            top_k=k,
            include_values=False,
            include_metadata=True
        )
        
        string=[]
        for i in range(k):
            string.append((results["matches"][i]['metadata']['source_text']))
        context=" ".join(string)
        return {"query": query,
                "context": context,
                "context_len": len(context)/4}
    except Exception as e:
        # print(e)
        return {"query": query,
                "context": "Error in Context retrival",
                "context_len": 0}
 

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=GROQ_API_KEY,
    max_tokens=2048,
    timeout=600,
    temperature=0.5,
    streaming=True,
    max_retries=1,
)

# checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[retrive_context],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

# deep_agent = create_deep_agent(
#     model=model,
#     tools=[retrive_context],
#     system_prompt=SYSTEM_PROMPT,
#     checkpointer=checkpointer,
# )

# content = input("Enter your query: ")
content = "who is R k agarwal?"

print("Running create_agent...", flush=True)
agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)
# print("Running create_deep_agent...", flush=True)
# deep_agent_result = deep_agent.invoke(
#     {"messages": [{"role": "user", "content": content}]},
#     config={"configurable": {"thread_id": "great-gatsby-da"}},
# )
print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)
# print("\ncreate_deep_agent:")
# print(deep_agent_result["messages"][-1].content_blocks)
