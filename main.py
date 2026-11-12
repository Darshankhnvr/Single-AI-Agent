
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent

load_dotenv()

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Assistant")
st.caption("Powered by Gemini, LangChain and Tavily")

# -------------------------------
# Initialize Agent
# -------------------------------

@st.cache_resource
def initialize_agent():

    weather_api_key = os.getenv("WEATHERSTACK_API_KEY")

    @tool
    def get_weather_data(city: str) -> str:
        """Fetch current weather information for a city."""

        url = "https://api.weatherstack.com/current"

        try:
            response = requests.get(
                url,
                params={
                    "access_key": weather_api_key,
                    "query": city
                },
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            if "current" not in data:
                return f"Could not fetch weather for {city}"

            return (
                f"City: {city}\n"
                f"Temperature: {data['current']['temperature']}°C\n"
                f"Weather: {data['current']['weather_descriptions'][0]}\n"
                f"Humidity: {data['current']['humidity']}%"
            )

        except requests.RequestException as e:
            return f"Weather API error: {e}"

    search_tool = TavilySearchResults(max_results=2)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    tools = [search_tool, get_weather_data]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant. "
            "Use web search for current information. "
            "Use the weather tool when users ask about weather. "
            "Provide clear and concise answers."
        )
    )

    return agent


# -------------------------------
# Initialize Session
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    agent = initialize_agent()
except Exception as e:
    st.error(f"Agent initialization failed: {e}")
    st.stop()


# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:
    st.header("AI Assistant")

    st.write("Available capabilities:")

    st.markdown("""
    🔎 **Web Search**

    🌤️ **Live Weather**

    💬 **General AI Chat**
    """)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# -------------------------------
# Display Chat History
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------
# Chat Input
# -------------------------------

user_input = st.chat_input("Ask me anything...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = agent.invoke({
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": msg["content"]
                        }
                        for msg in st.session_state.messages
                    ]
                })

                answer = response["messages"][-1].content

                if isinstance(answer, list):
                    answer = "\n".join(
                        item["text"]
                        for item in answer
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
