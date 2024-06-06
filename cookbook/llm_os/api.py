from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ChatInput(BaseModel):
    message: str

@app.post("/chat/")
async def chat(input: ChatInput):
    if not input.message:
        raise HTTPException(status_code=400, detail="No message provided")
    
    # Here, process the message using your Streamlit app logic or any other processing
    response_message = "Processed message: " + input.message  # Placeholder for your logic

    return {"response": response_message}
