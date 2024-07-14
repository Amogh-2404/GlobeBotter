from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains import LLMChain, ConversationChain
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

chat = ChatOpenAI()
messages = [
    SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
    HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")]
output = chat.invoke(messages)
print(output.content)
