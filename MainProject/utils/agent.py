"""
This file handles all logic for agents, including creating an agent, giving it tools, memory, etc.
create_agent    Creates an agent that has access to the tools in 'scripts', and returns it.
"""

# Imports
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

from . import scripts

def create_agent(model_name: str = "gpt-4o", max_msgs: int = 10):
    # Define the prompt. Includes system prompt, chat history, human input, and placeholder for LLM response.
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are given tools for loading data and building LDA, QDA, and PCA."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Define the model to be used.
    model = ChatOpenAI(model=model_name, temperature=0)

    # Create the agent. Attach the list of tools included in 'scripts'.
    agent = create_tool_calling_agent(model, scripts.tz, prompt)

    # Create the memory storage for the conversation.
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=max_msgs)

    # Create and return the agent executor, using these objects.
    return AgentExecutor(agent=agent, tools=scripts.tz, memory=memory, verbose=True)