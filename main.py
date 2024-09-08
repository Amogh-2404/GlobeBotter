import streamlit as st
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks import StreamlitCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(page_title="GlobeBotter", page_icon="🌐")
st.header(
    '🌐 Welcome to Globebotter, your travel assistant with Internet access. What are you planning for your next trip?')

search = SerpAPIWrapper(serpapi_api_key=os.getenv('SERPAPI_API_KEY'))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)
llm = ChatOpenAI()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(),
        "italy_travel",
        "Searches and returns documents regarding Italy."
    )
]
agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True)

user_query = st.text_input(
    "**Where are you planning your next vacation?**",
    placeholder="Ask me anything!"
)

if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", 'content': 'How can I help you?'}]
if "memory" not in st.session_state:
    st.session_state["memory"] = memory

for msg in st.session_state['messages']:
    st.chat_message(msg["role"]).write(msg['content'])

if user_query:
    st.session_state.messages.append({'role': 'user', "content": user_query})
    st.chat_message('user').write(user_query)
    with st.chat_message('assistant'):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append((ChatMessage(role="assistant", content=response.content)))

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []