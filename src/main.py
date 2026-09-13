from fastapi import FastAPI
from pydantic import BaseModel
from agents.pipeline import build_pipeline
from core.logger import logger
import uvicorn

app = FastAPI(title="LangGraph 智能客服服务")
pipeline = build_pipeline()

class ChatRequest(BaseModel):
    conversation_id: str
    user_query: str
    intent: str
    slots: dict
    chat_history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"收到请求: {request.conversation_id} - {request.user_query}")
    initial_state = {
        "conversation_id": request.conversation_id,
        "user_query": request.user_query,
        "intent": request.intent,
        "slots": request.slots,
        "chat_history": request.chat_history,
        "retry_count": 0
    }
    result = pipeline.invoke(initial_state)
    return {"final_reply": result["final_reply"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
