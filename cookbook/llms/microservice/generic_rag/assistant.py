from typing import Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.llm.openai import OpenAIChat
from phi.llm.anthropic import Claude
from phi.llm.mistral import Mistral
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.embedder.voyageai import VoyageAIEmbedder
from phi.embedder.mistral import MistralEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.vectordb.qdrant import Qdrant
from phi.vectordb.pineconedb import PineconeDB
from phi.storage.assistant.postgres import PgAssistantStorage
import os

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "generic-rag-index"
def get_llm(llm_model: str):
    """Factory function to get the appropriate LLM instance based on the model name."""
    if "llama" in llm_model:
        return Groq(model=llm_model)
    elif "gpt" in llm_model or "openai" in llm_model:
        return OpenAIChat(model=llm_model)
    elif "claude" in llm_model:
        return Claude(model=llm_model)
    elif "mistral" in llm_model:
        return Mistral(llm_model=llm_model)
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

def get_embedder(embeddings_model: str):
    """Factory function to get the appropriate embedder instance based on the model name."""
    if "ollama" in embeddings_model or "nomic" in embeddings_model:
        return OllamaEmbedder(model=embeddings_model, dimensions=768)
    elif "ada" in embeddings_model or "small" in embeddings_model:
        return OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    elif "voyage" in embeddings_model:
        # Assuming a hypothetical configuration for text-embedding-v2
        return VoyageAIEmbedder(model=embeddings_model, dimensions=1024)
    elif "mistral" in embeddings_model:
        return MistralEmbedder(model=embeddings_model,dimensions=1024)
    else:
        raise ValueError(f"Unsupported embeddings model: {embeddings_model}")

def get_vector_db(vector_db: str, db_url: str, qdrant_api_key: Optional[str], qdrant_url: Optional[str], pinecone_api_key: Optional[str], pinecone_index_name: Optional[str], embeddings_table: str, embedder):
    """Factory function to get the appropriate vector db instance based on the model name."""
    if vector_db == "Qdrant":
        return Qdrant(
            collection=embeddings_table,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
    elif vector_db == "PgVector":
        return PgVector2(
            db_url=db_url,
            collection=embeddings_table,
            embedder=embedder,
        )
    elif vector_db == "Pinecone":
        return PineconeDB(
            name=pinecone_index_name,
            dimension=1536,  # or another appropriate dimension size
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            api_key=pinecone_api_key,
        )
    else:
        raise ValueError(f"Unsupported vector database type: {vector_db}")


def get_auto_rag_assistant(
    llm_model: str,
    embeddings_model: str,
    vector_db: str,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True
) -> Assistant:
    """Get a configurable RAG Assistant based on the LLM and embeddings model."""
    llm = get_llm(llm_model)
    embedder = get_embedder(embeddings_model)
    embeddings_table = f"auto_rag_documents_{llm_model.lower()}_{embeddings_model.lower()}_{vector_db.lower()}"
    vector_db = get_vector_db(vector_db, db_url, qdrant_api_key, qdrant_url, pinecone_api_key, pinecone_index_name, embeddings_table, embedder)

    return Assistant(
        name=f"auto_rag_assistant_{llm_model.lower()}_{embeddings_model.lower()}",
        run_id=run_id,
        user_id=user_id,
        llm=llm,
        storage=PgAssistantStorage(table_name=f"auto_rag_assistant_{llm_model.lower()}_{embeddings_model.lower()}", db_url=db_url),
         knowledge_base=AssistantKnowledge(
            vector_db = vector_db,
            num_documents=3,
        ),
        description="You are an Assistant called 'Generic RAG' that answers questions by calling functions.",
        instructions=[
            "First get additional information about the users question.",
            "If the user asks to summarize the conversation, use the `get_chat_history` tool to get your chat history with the user.",
            "Carefully process the information you have gathered and provide a clear and concise answer to the user.",
            "Respond directly to the user with your answer, do not say 'here is the answer' or 'this is the answer' or 'According to the information provided'",
            "NEVER mention your knowledge base or say 'According to the search_knowledge_base tool' or 'According to {some_tool} tool'.",
        ],
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
