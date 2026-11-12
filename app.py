import os
import certifi
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub
from langchain.agents import create_agent
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
search_tool = TavilySearchResults(max_results=2)
@tool
def get_weather_data(city : str) -> str:
    """
    Fetch the current weather information for a city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"city : {city}\n"
        f"Temperature: {data['current']['temperature']}*C \n"
    )
search_tool.invoke("What is the capital of India?")
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)
from google import genai
import os

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(model.name)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
response = llm.invoke("Tell me a joke about AI")
print(response.content)

tools = [search_tool, get_weather_data]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use tools when necessary."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find the capital if India and then find its current weather"}
    ]
})

print(response["messages"][-1].content)