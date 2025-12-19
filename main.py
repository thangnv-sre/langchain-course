from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    result = chain.invoke(
        {"input": "Tìm 2 job devops cho tôi link tuyển dụng ở Việt Nam có mức lương trên 40 triệu đồng/tháng."}
    )
    print(result)


if __name__ == "__main__":
    main()
