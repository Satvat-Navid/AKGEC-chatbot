import json
import streamlit as st
from pinecone.grpc import PineconeGRPC as Pinecone
from groq import Groq

# Groq API
client = Groq( api_key=st.secrets['GROQ_API_KEY'])
pc = Pinecone(api_key=st.secrets['PC_API_KEY'])
index = pc.Index('akgec-data')

# Model selection
TOOL_MODEL = "llama-3.3-70b-versatile"
GENERAL_MODEL = "openai/gpt-oss-120b"

# Similarity Search and context retrival from pinecone.
def get_context(prompt, k=3):
    query_embedding = pc.inference.embed(
        # model="multilingual-e5-large",
        model="llama-text-embed-v2",
        inputs=[prompt],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="doc1",
        vector=query_embedding[0].values,
        top_k=k,
        include_values=False,
        include_metadata=True
    )
    string=[]
    for i in range(k):
        string.append((results["matches"][i]['metadata']['source_text']))
    context=" ".join(string)

    return context

# This is the summerizing funtion.
def summerize(model, chat, max_tokens=500):
    summery = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are responsible for summarizing the conversation history in a clear and concise manner to establish context for subsequent interactions with the large language model. The summary must preserve all information without omissions and present it in chronological order."},
            {"role": "user", "content": chat}
        ],
        max_tokens=max_tokens,
        stream=False
    )
    return summery.choices[0].message.content

# Function for specify the query for context generation.
def context(model, message):

    tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_context",
                    "description": "Retrive the data from the vector data base using similarity search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to search the relevant data in vector data base. Prompt to retrive maximum information and specific regarding that topic. ",
                            }
                        },
                        "required": ["prompt"],
                    },
                },
            }
        ]
    
    ai_response = client.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=100,
        tools=tools,
        tool_choice="auto",
        stream=False
    )

# Extract the response and any tool call responses
    response_message = ai_response.choices[0].message
    tool_calls = response_message.tool_calls
    function_response = ""
    if tool_calls:
        # # Add the LLM's response to the conversation
        # messages.append(response_message)

        # function_name = tool_calls.function.name
        function_to_call = get_context
        function_args = json.loads(tool_calls[0].function.arguments)

        print(function_args.get("prompt"))

        # Call the tool and get the response
        function_response = function_to_call(
            prompt=function_args.get("prompt")
        )
        # Add the tool response to the conversation
        message.append(
            {
                "tool_call_id": tool_calls[0].id, 
                "role": "tool", # Indicates this message is from tool use
                "name": "get_context",
                "content": "context from college data base",
            }
        )    
    if len(function_response) != 0:
        return function_response
    return ""

# Response from AI
def response(model, user_input, context="", history=""):

    message = [
            {"role": "system", "content": """You are a dedicated chatbot designed to assist users of Ajay Kumar Garg Engineering College (AKGEC). Please ensure that responses are strictly relevant to the college context; avoid answering questions unrelated or outside the scope of the institution.

Provide clear, concise, and well-structured information by adhering to the following guidelines:

Emphasize key points in bold.

Use italics to highlight important terms or phrases.

Organize lengthy responses with appropriate headings.

Present multiple items using bullet points or numbered lists.

Incorporate tables where they enhance clarity and understanding."""},
            {"role": "system", "content": f"conversation history : {history}"},
            {"role": "user", "content": f"context: {context}"},
            {"role": "user", "content": user_input}
        ]

    ai_response = client.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=1023,
        stream=False
    )
    return ai_response.choices[0].message.content

# Streamlit chat app
st.title("AKGEC Chatbot")

if "history" not in st.session_state:
    st.session_state.history = ""
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Type your message...")

if user_input:
    # Summarize history if present
    if st.session_state.history:
        st.session_state.history = summerize(GENERAL_MODEL, st.session_state.history, 500)
    # Prepare messages for context tool
    M1 = [
        {"role": "system", "content": """You are a prompt generator designed to facilitate similarity searches within a vector database .
         Use the tool to retrieve relevant context based on the generated prompt .
         Ensure the prompt aligns with the provided user input and conversation history to maintain coherence and accuracy during retrieval .
         Maintain a contextual approach . Do not use tool if user greets or thanks .
         Add all the necessary detail in the PROMPT from the conversation history ."""},
        {"role": "assistant", "content": st.session_state.history},
        {"role": "user", "content": user_input}
    ]
    db_context = context(model=TOOL_MODEL, message=M1)
    temp = f""" PREVIOUS : "{st.session_state.history}" QUERY : "{user_input}" ANSWER : "{db_context}" """
    reply = response(GENERAL_MODEL, user_input, db_context)
    st.session_state.chat.append({"role": "user", "content": user_input})
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.session_state.history = temp

# Display chat history
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

        
