import os
from dotenv import load_dotenv
import asyncio
import json
from groq import AsyncGroq
from pinecone.grpc import PineconeGRPC as Pinecone
import instructor
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PC_API_KEY = os.getenv("PC_API_KEY")

# Init clients
client = AsyncGroq( api_key=GROQ_API_KEY)
client1 = instructor.from_groq(AsyncGroq(api_key=GROQ_API_KEY), mode=instructor.Mode.JSON)
pc = Pinecone(api_key=PC_API_KEY)
index = pc.Index('akgec-data')

# Model selection
# TOOL_MODEL = "moonshotai/kimi-k2-instruct-0905"
TOOL_MODEL = "qwen/qwen3.8-27b"
GENERAL_MODEL = "openai/gpt-oss-120b"

# Response from AI
async def response(user_input, model=GENERAL_MODEL, context="", history=""):

    message = [
            {"role": "system",
             "content": "You are a dedicated chatbot designed to assist users of Ajay Kumar Garg "
             "Engineering College (AKGEC).Please ensure that responses are strictly relevant "
             "to the college context; avoid answering questions unrelated or outside the scope "
             "of the institution.Provide clear, concise, and well-structured information by "
             "adhering to the following guidelines:Emphasize key points in bold.Use italics to "
             "highlight important terms or phrases.Organize lengthy responses with appropriate headings."
             "Present multiple items using bullet points or numbered lists."
             "Incorporate tables where they enhance clarity and understanding."
            },
            {"role": "system", "content": f"conversation history : {history}"},
            {"role": "user", "content": f"context: {context}"},
            {"role": "user", "content": user_input}
        ]

    ai_response = await client.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=2048,
        stream=False
    )
    return ai_response.choices[0].message.content

# This is the summerizing funtion.(need improvement in system prompt and structured output)
async def summerize(chat, model=GENERAL_MODEL, max_tokens=512):
    """It summerize the convversation of user with the chatbot"""
    try:
        summery = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are responsible for summarizing the conversation "
                "history in a clear and concise manner to establish context for subsequent "
                "interactions with the large language model. The summary must preserve all "
                "information without omissions and present it in chronological order.give "
                "in text format with clear distintion in user and reply"},
                {"role": "user", "content": chat}
            ],
            max_tokens=max_tokens,
            stream=False
        )
        return summery.choices[0].message.content
                # "history_len": len(summery.choices[0].message.content)/4
                           
    except Exception as e:
        # print(e)
        return "Error occured while summerizing history."

# Pinecone Vector db search
def retrive_context(query, k=2):
    """Funtion to retrive data from vector database using similarity search with the 
        provided query and return the context along with the query and context length."""
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
 
# Function for specify the query for context generation.
async def context(message, model=TOOL_MODEL, history=""):
    """Use tool call to retrive context form the vector database"""
    tool_schema = {
                "name": "retrive_context",
                "description": "Retrive the data from the vector data base using similarity search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search the relevant data in vector data base. "
                            "query to retrive information about the person, thing or specifically regarding that topic.",
                        }
                    },
                    "required": ["query"],
                    },
                },
    
    # Define the Pydantic model for the tool call
    class ToolCall(BaseModel):
        input_text: str = Field(description="The user's input query")
        tool_name: str = Field(description="The name of the tool to call")
        tool_parameters: str = Field(description="JSON string of tool parameters")

    class ResponseModel(BaseModel):
        tool_calls: list[ToolCall]

    messages = [
        {
            "role": "system",
            "content": f"""You are an assistant that can use tools. 
            You have access to the following tool: {tool_schema}. Only use tool when necessory.
            you use tool only for the final query from the user if the relevent data is not in history.
            Keep in mind that you are using tool for searching in college database"""
        },
        {
            "role": "user", 
            "content": f"**Conversation history: {history}\n\n**Final Query: {message}"}
    ]

    response = await client1.chat.completions.create(
        model=model,
        messages=messages,
        response_model=ResponseModel,
        # max_tokens=100,
        temperature=0.5,
        max_completion_tokens=512,
        # tools=tools,
        tool_choice="auto",
        # tool_choice={"type": "function", "function": {"name": "get_context"}},
        stream=False
    )
    # return response.tool_calls
    
    # Extract the response and any tool call responses
    # response_message = response.choices[0].message
    tool_calls = response.tool_calls
    if tool_calls:
        # # Add the LLM's response to the conversation
        # messages.append(response_message)

        # function_name = tool_calls.function.name
        function_to_call = retrive_context
        function_args = json.loads(tool_calls[0].tool_parameters)

        # Call the tool and get the response
        function_response = function_to_call(
            query=function_args.get("query")
        )
        # function_response = json.loads(function_response)

        # # Add the tool response to the conversation, not required when called externally?
        # messages.append(
        #     {
        #         "role": "tool", # Indicates this message is from tool use
        #         "name": "retrive_context",
        #         "query": tool_calls[0].tool_parameters,
        #     }
        # )
        return function_response
    function_response = {"query": message,
                        "context": "",
                        "context_len": 0}
    return function_response
