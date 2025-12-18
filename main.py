from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Docstring for search
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(temperature=0, model="gpt-5")
# llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
tools = [search]
agent = create_agent(
    llm,
    tools,
)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Tìm 3 công việc phù hợp với kỹ năng xây dựng AIagent ở Hà Nội?")
            ]
        }
    )
    messages = response.get("messages", [])
    last_ai = next((m for m in reversed(messages) if hasattr(m, "content")), None)
    print("Agent response:", getattr(last_ai, "content", response))


if __name__ == "__main__":
    main()
