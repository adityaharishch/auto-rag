import nest_asyncio
from typing import List

import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from assistant import get_auto_rag_assistant  # type: ignore

nest_asyncio.apply()
st.set_page_config(
    page_title="Generic RAG",
)
st.title("Generic RAG")


def restart_assistant():
    logger.debug("---*--- Restarting Assistant ---*---")
    st.session_state["auto_rag_assistant"] = None
    st.session_state["auto_rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


def main() -> None:
    # Get LLM model
    llm_model = st.sidebar.selectbox(
        "Select LLM Model",
    options=[
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-4o",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "mistral-large-latest"
    ]
    )
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    # Get Embeddings model
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["text-embedding-3-small", "text-embedding-ada-002", "nomic-embed-text", "voyage-2", "mistral-embed"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )
    # Set assistant_type in session state
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()


    #Get the Vector DB memory
    vector_db = st.sidebar.selectbox(
        "Select Vector Database",
        options=["PgVector","Qdrant","Pinecone"],
        help="When you change the vector db, the documents will need to be added again.",
    )
     # Set assistant_type in session state
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = vector_db
    # Restart the assistant if assistant_type has changed
    elif st.session_state["vector_db"] != vector_db:
        st.session_state["vector_db"] = vector_db
        st.session_state["vector_db_updated"] = True
        restart_assistant()


    # Get the assistant
    auto_rag_assistant: Assistant
    if "auto_rag_assistant" not in st.session_state or st.session_state["auto_rag_assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        auto_rag_assistant = get_auto_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model, vector_db=vector_db)
        st.session_state["auto_rag_assistant"] = auto_rag_assistant
    else:
        auto_rag_assistant = st.session_state["auto_rag_assistant"]

    # Create assistant run (i.e. log to database) and save run_id in session state
    try:
        st.session_state["auto_rag_assistant_run_id"] = auto_rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    # Load existing messages
    assistant_chat_history = auto_rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Upload a doc and ask me questions..."}]

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            resp_container = st.empty()
            # Streaming is not supported with function calling on Groq atm
            response = auto_rag_assistant.run(question, stream=False)
            resp_container.markdown(response)  # type: ignore
            # Once streaming is supported, the following code can be used
            # response = ""
            # for delta in auto_rag_assistant.run(question):
            #     response += delta  # type: ignore
            #     resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # Load knowledge base
    if auto_rag_assistant.knowledge_base:
        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        auto_rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()
                restart_assistant()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing PDF...", icon="🧠")
            rag_name = uploaded_file.name.split(".")[0]
            if f"{rag_name}_uploaded" not in st.session_state:
                reader = PDFReader()
                rag_documents: List[Document] = reader.read(uploaded_file)
                if rag_documents:
                    auto_rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                else:
                    st.sidebar.error("Could not read PDF")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()
            # restart_assistant()

    if auto_rag_assistant.knowledge_base and auto_rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            auto_rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")
            restart_assistant()

    if auto_rag_assistant.storage:
        auto_rag_assistant_run_ids: List[str] = auto_rag_assistant.storage.get_all_run_ids()
        new_auto_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=auto_rag_assistant_run_ids)
        if st.session_state["auto_rag_assistant_run_id"] != new_auto_rag_assistant_run_id:
            logger.info(f"---*--- Loading {llm_model} run: {new_auto_rag_assistant_run_id} ---*---")
            st.session_state["auto_rag_assistant"] = get_auto_rag_assistant(
                llm_model=llm_model, embeddings_model=embeddings_model, run_id=new_auto_rag_assistant_run_id
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()

    if "embeddings_model_updated" in st.session_state:
        st.sidebar.info("Please add documents again as the embeddings model has changed.")
        st.session_state["embeddings_model_updated"] = False
    if "vector_db_updated" in st.session_state:
        st.sidebar.info("Please add documents again as the vector DB model has changed.")
        st.session_state["vector_db_updated"] = False
main()


