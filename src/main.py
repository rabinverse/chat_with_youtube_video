from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


import os

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API"
parser = StrOutputParser()
####

video_id = "K4Ze-Sp6aUE"

try:
    # 1. Create an instance of the API

    api = YouTubeTranscriptApi()

    # 2. Use .fetch() instead of .get_transcript()
    # 3. Call .to_raw_data() to get the familiar list of dictionaries
    transcript_list = api.fetch(video_id, languages=["en"]).to_raw_data()

    # Join the text parts into one string
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("Transcript fetched successfully!")
    print(transcript[:500])

except Exception as e:
    print(f"An error occurred: {e}")

####


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk = splitter.create_documents([transcript])

####

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_documents(chunk, embeddings)


####

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

####


llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

####

prompt1 = PromptTemplate(
    template="""You are a helpful assistant. Answer only from the provided transcript content.If the content is insufficient just say I dont know. {context}, Question={user_question}""",
    input_variables=["context", "user_question"],
)
# question = (
#     "is the topic of protein discussed in the question if yes then what is discussed"
# )

# retrieved_docs = retriever.invoke(question)
# content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
####


def format_docs(retrieved_docs):
    content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return content_text


####

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "user_question": RunnablePassthrough(),
    }
)


####

chain = parallel_chain | prompt1 | llm | parser


####

answer = chain.invoke("how much protein should a person eat normally")
print(answer)





import streamlit as st
