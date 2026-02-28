from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY

def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",
        temperature=0
    )
