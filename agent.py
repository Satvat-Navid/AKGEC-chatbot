import os
import json
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from pinecone.grpc import PineconeGRPC as Pinecone

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, SystemMessage
from erp_info import ERPClient, ERPError, ERPData

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PC_API_KEY = os.getenv("PC_API_KEY")

pc = Pinecone(api_key=PC_API_KEY)
index = pc.Index('akgec-data')

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)


class Route(BaseModel):
    step: Literal["student_info", "college_info", "retrival_not_required"] = Field(
        description=(
            "Use student_info only for the user's private ERP/student data, "
            "especially attendance, and only when both credentials are present. "
            "Use college_info for public AKGEC information, including people, "
            "faculty, departments, courses, placements, campus, and admissions. "
            "Use retrival_not_required only for greetings or unrelated requests."
        )
    )
    roll_no: str | None = Field(
        default=None,
        description=(
            "Student roll number copied exactly from the current user message; "
            "required for student_info and never inferred"
        ),
    )
    password: str | None = Field(
        default=None,
        description=(
            "Student ERP password copied exactly from the current user message; "
            "required for student_info and never inferred"
        ),
    )
    search_query: str | None = Field(
        default=None,
        description="Question to use for College_info",
    )

# class Parameters(TypedDict):
#     roll_no: str | None
#     password: str | None
#     query: str | None

# State
class State(TypedDict):
    input: str
    decision: str
    roll_no: str | None
    password: str | None
    search_query: str | None
    context: str | None
    output: str | None

# Student ERP data fetch
# @tool
def student_data(roll_no: str, password: str) -> ERPData:
    """Function to get Student acadamic data and student attandance in a structured way """
    try:
        with ERPClient(roll_no, password) as client:
            data = client.fetch_data()

        # save_json("data.json", data=data) # this line will save the erp data of the student.
        return data

    except ERPError as exc:
        print(f"ERP error: {exc}")


# Pinecone Vector db search
# @tool
def college_info(query: str):
    """Funtion to retrive data from vector database using similarity search with the 
    provided query and return the context along with context length."""
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
        return {"context": context,
                "approx_token": len(context)/4}
    except Exception as e:
        # print(e)
        return {"context": "Error in Context retrival",
                "approx_token": 0}

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

ROUTER_PROMPT = """You are the strict request router for an AKGEC chatbot.

Choose exactly one route:
- college_info: public or general AKGEC information. This includes questions
  about a person, faculty member, department, course, campus, admission,
  placement, contact details, or any other college knowledge-base topic.
- student_info: private ERP data belonging to a student, such as attendance
  or academic records. Choose this route only if the user message contains
  both a roll number and a password. Never invent, infer, or reuse credentials.
- retrival_not_required: greetings, thanks, or questions unrelated to AKGEC.

Important rules:
1. A question about a person or college is college_info, even if the person is
    a student or the user says "my"; it is not student_info without both ERP
    credentials in the current message.
2. An attendance/ERP question without both credentials is still student_info
    with null credentials. The application will refuse it without calling ERP.
3. Copy roll_no and password exactly from the user's message.
4. For college_info, create a focused search_query. Do not put credentials in
    search_query.
"""

# tools = [college_info, student_data]
# tools_by_name = {tool.name: tool for tool in tools}
# model_with_tools = model.bind_tools(tools)


# Augment the LLM with schema for structured output
model_struc_out = model.with_structured_output(Route)

# Nodes
def tool_call_1(state: State):
    """Get student information like academic and attendance."""

    data = student_data(roll_no=state["roll_no"], password=state["password"])
    existing_context = state.get("context") or ""
    data_text = json.dumps(data, ensure_ascii=False, default=str) if isinstance(data, dict) else str(data)
    context = f"{existing_context}\n{data_text}".strip()
    return {"context": context}


def access_denied(state: State):
    """Explain why private ERP data cannot be fetched yet."""

    return {
        "output": (
            "I can check attendance or other private student ERP details only "
            "when you provide both your roll number and ERP password. "
            "I cannot retrieve that information without those credentials."
        )
    }


def tool_call_2(state: State):
    """Get college information like academic, placement, faculty, campus etc."""

    result = college_info(query=state["search_query"] or state["input"])
    existing_context = state.get("context") or ""
    context_text = result.get("context", "") if isinstance(result, dict) else str(result)
    context = f"{existing_context}\n{context_text}".strip()
    return {"context": context}


def llm_call(state: State):
    """Answer the asked query using the gathered context."""

    result = model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{state['input']}\n\n# context: {state.get('context', '')}"),
    ])
    return {"output": getattr(result, "content", str(result))}


# Run the augmented LLM with structured output to serve as tool call logic
def tool_caller(state: State):
    """Route the input to the appropriate node"""

    result = model_struc_out.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=state["input"]),
        ])
    return {"decision": result.step,
            "roll_no": result.roll_no,
            'password': result.password,
            "search_query": result.search_query
            }


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    if state["decision"] == "student_info":
        # Never let an incomplete model extraction reach the ERP client.
        if not state.get("roll_no") or not state.get("password"):
            return "denied"
        return "student"
    elif state["decision"] == "college_info":
        return "college"
    elif state["decision"] == "retrival_not_required":
        return "llm_call"
    else:
        return END


# Build workflow
graph = StateGraph(State)

# Add nodes
graph.add_node("student", tool_call_1)
graph.add_node("college", tool_call_2)
graph.add_node("denied", access_denied)
graph.add_node("tool_caller", tool_caller)
graph.add_node("llm_call", llm_call)


# Add edges to connect nodes
graph.add_edge(START, "tool_caller")
graph.add_conditional_edges(
    "tool_caller",
    route_decision,
    {
        "student": "student",
        "college": "college",
        "denied": "denied",
        "llm_call": "llm_call",
    },
)
graph.add_edge("student", "llm_call")
graph.add_edge("college", "llm_call")
graph.add_edge("denied", END)
# graph.add_edge("llm_call_3", END)

# Compile workflow
chat = graph.compile()


def run_agent(message: str, history: str = "") -> dict:
    """Run the graph for a user message and return the API-friendly payload."""
    state = chat.invoke({
        "input": message,
        "decision": "",
        "roll_no": None,
        "password": None,
        "search_query": None,
        "context": "",
        "output": "",
    })

    context_text = state.get("context") or ""
    context_payload = {
        "query": message,
        "context": context_text,
        "context_len": len(context_text) / 4 if context_text else 0,
    }

    return {
        "reply": state.get("output", ""),
        "context_used": context_payload,
        "history": history,
    }


if __name__ == "__main__":
    example_message = "hi, My name is Satvat Navid"
    print(run_agent(example_message)["reply"])
    