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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


load_dotenv()

chat = ChatOpenAI()
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=chat, verbose=True, memory=memory)

#
# messages = [
#     SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
#     HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")]
# output = chat.invoke(messages)
# print(output.content)

while True:
    query = input("you: ")
    if query == 'q':
        break
    output = conversation({"input": query})
    print("User: ", query)
    print('AI system: ', output['response'])


