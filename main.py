import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import ChatRequest
from agents import context, summerize, response, GENERAL_MODEL, TOOL_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AKGEC Chatbot API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom exception handler to return JSON instead of plain text on rate limit
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}",
                "error": "Too Many Requests"
        }
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot.mlcoe.tech/", "https://chatbot.satvat.pro/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
@limiter.limit("2/minute; 100/day")
async def chat(fastapi_request: ChatRequest, request: Request):
    logger.info(f"Received query {request.client.host}: {fastapi_request.message}")
    try:
        history = await summerize(fastapi_request.history, model=GENERAL_MODEL, max_tokens=512) if fastapi_request.history else ""
        db_context = await context(message=fastapi_request.message, history=history, model=TOOL_MODEL)
        reply = await response(user_input=fastapi_request.message, model=GENERAL_MODEL, context=db_context.get('context'), history=history)
        logger.info(f"Conversation History Lenght: {len(history)/4}")
        # logger.info(db_context.get("context"))
        return {
            "reply": reply,
            "context_used": db_context,
            "history": history
        }
    except Exception as e:
        logger.error("!!! An unexpected error occurred !!!", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# This is crucial part for serving the frontend
app.mount("/", StaticFiles(directory="public", html = True), name="static")