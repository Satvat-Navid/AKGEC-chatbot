import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import ChatRequest
from agent import run_agent
from agent_async import run_agent  #In case if i want to retrieve the data in parallel
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
@limiter.limit("3/minute; 100/day")
async def chat(fastapi_request: ChatRequest, request: Request):
    logger.info("Received chat query from %s", request.client.host)
    try:
        result = await __import__('asyncio').to_thread(
            run_agent,
            fastapi_request.message,
            fastapi_request.history,
        )
        logger.info(f"Conversation History Length: {len(fastapi_request.history or '') / 4}")
        return result
    except Exception as e:
        logger.error("!!! An unexpected error occurred !!!", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# This is crucial part for serving the frontend
app.mount("/", StaticFiles(directory="public", html = True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")