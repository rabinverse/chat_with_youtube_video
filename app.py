import streamlit as st
import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from requests import Session

# Configure page
st.set_page_config(page_title="YouTube Q&A Assistant", page_icon="🎥", layout="wide")


def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query)["v"][0]
        if parsed.path[:7] == "/embed/":
            return parsed.path.split("/")[2]
        if parsed.path[:3] == "/v/":
            return parsed.path.split("/")[2]
    return None


def get_llm(provider, model, api_key, temperature=0.4):
    """Initialize LLM based on provider selection"""
    if provider == "Google Gemini":
        from langchain_google_genai import GoogleGenerativeAI

        os.environ["GOOGLE_API_KEY"] = api_key
        return GoogleGenerativeAI(model=model, temperature=temperature)

    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model=model, temperature=temperature)

    elif provider == "Groq":
        from langchain_groq import ChatGroq

        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(model=model, temperature=temperature)


# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_title" not in st.session_state:
    st.session_state.video_title = None

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("YouTube Q&A")

# Sidebar 
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Provider", ["Groq", "Google Gemini", "OpenAI"])

    model_options = {
        "Groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "Google Gemini": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.5-flash",
        ],
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    }

    selected_model = st.selectbox("Model", model_options[provider])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)

    api_key_label = {
        "Groq": "Groq API Key",
        "Google Gemini": "Google API Key",
        "OpenAI": "OpenAI API Key",
    }
    
    ##
    secret_key = f"{provider.upper().replace(' ', '_')}_API_KEY"
    default_key = ""
    
    try:
        if secret_key in st.secrets:
            default_key = st.secrets[secret_key]
    except:
        pass
    
    if not default_key and secret_key in os.environ:
        default_key = os.environ[secret_key]
    
    if default_key:
        st.success(f"✓ {api_key_label[provider]} configured from secrets")
        api_key = default_key
    else:
        st.caption("Visit providers api/docs then create and paste API key here")
        api_key = st.text_input(
            api_key_label[provider],
            type="password",
            value="",
        )

        if st.button("Save API Key"):
            if api_key:
                os.environ[secret_key] = api_key
                st.success("Saved")
            else:
                st.error("Enter key")
    
    st.caption(
        "Visit https://www.webshare.io/, and find proxy in the free section paste here the link incase Youtube blocks without porxy "
    )
    proxy = st.text_input("Proxy", placeholder="http://user:pass@host:port")

    if st.button("Set Proxy"):
        if proxy:
            st.session_state.proxy = proxy
            st.success("Proxy set")
        else:
            if "proxy" in st.session_state:
                del st.session_state.proxy
            st.info("Proxy cleared")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
api_key_env_var = f"{provider.upper().replace(' ', '_')}_API_KEY"
has_api_key = False

try:
    if api_key_env_var in st.secrets:
        has_api_key = True
except:
    pass

if not has_api_key and api_key_env_var in os.environ and os.environ[api_key_env_var]:
    has_api_key = True

if not has_api_key and api_key:
    has_api_key = True

if not has_api_key:
    st.warning("Set API key in sidebar")
else:
    url = st.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        process_button = st.button("Process")
    with col2:
        if st.session_state.vector_store:
            if st.button("Reset"):
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                st.session_state.video_title = None
                st.rerun()

    if process_button:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid URL")
        else:
            with st.spinner("Processing..."):
                try:
                    proxy = st.session_state.get("proxy", "")
                    if proxy:
                        proxies = {"http": proxy, "https": proxy}
                        http_client = Session()
                        http_client.proxies.update(proxies)
                        api = YouTubeTranscriptApi(http_client=http_client)
                    else:
                        api = YouTubeTranscriptApi()

                    fetched_transcript = api.fetch(video_id, languages=["en"])
                    transcript_list = fetched_transcript.to_raw_data()
                    transcript = " ".join(chunk["text"] for chunk in transcript_list)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = splitter.create_documents([transcript])

                    google_key = ""
                    try:
                        if "GOOGLE_API_KEY" in st.secrets:
                            google_key = st.secrets["GOOGLE_API_KEY"]
                    except:
                        pass
                    
                    if not google_key and "GOOGLE_API_KEY" in os.environ:
                        google_key = os.environ["GOOGLE_API_KEY"]
                    
                    if google_key:
                        os.environ["GOOGLE_API_KEY"] = google_key
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/text-embedding-004"
                        )
                    else:
                        st.error("Google API Key needed for embeddings")
                        st.stop()

                    vector_store = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    st.session_state.video_title = f"Video: {video_id}"
                    st.session_state.chat_history = []
                    st.success("Processed")

                except (TranscriptsDisabled, NoTranscriptFound) as e:
                    st.error(f"Transcript unavailable: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Show status
    if st.session_state.vector_store:
        st.info(f"{st.session_state.video_title} | {provider} - {selected_model}")

# Q&A Section
if st.session_state.vector_store:

    #  chat history
    if st.session_state.chat_history:
        for q, a in st.session_state.chat_history:
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
            st.divider()

    # Question input
    question = st.text_input("Question", placeholder="Ask about the video...")

    if st.button("Ask"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}
                    )

                    llm = get_llm(provider, selected_model, api_key, temperature)

                    prompt = PromptTemplate(
                        template="""You are a helpful assistant analyzing a YouTube video transcript. 
                        
Answer the question based ONLY on the provided transcript content. If the information is not in the transcript, say "I don't have enough information in the transcript to answer that question."

Be concise, accurate, and helpful.

Transcript Context:
{context}

Question: {user_question}

Answer:""",
                        input_variables=["context", "user_question"],
                    )

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "user_question": RunnablePassthrough(),
                        }
                    )

                    chain = parallel_chain | prompt | llm | StrOutputParser()
                    answer = chain.invoke(question)

                    st.session_state.chat_history.append((question, answer))
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Enter a question")