from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient
from langgraph.prebuilt import create_react_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="City Agent", page_icon="🏙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background: #0a0a0f; color: #e8e4dc; }

.city-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 2.4rem; letter-spacing: -0.03em; color: #f0e9d6;
}
.city-subtitle {
    font-size: 0.72rem; color: #6b7280;
    letter-spacing: 0.14em; text-transform: uppercase; margin-top: 2px;
}
.msg-user {
    background: #161620; border: 1px solid #2a2a38;
    border-radius: 12px 12px 2px 12px; padding: 12px 16px;
    margin: 6px 0; font-size: 0.86rem; color: #e8e4dc;
    max-width: 80%; margin-left: auto;
}
.msg-bot {
    background: #111118; border: 1px solid #1e3a2f;
    border-left: 3px solid #22c55e; border-radius: 2px 12px 12px 12px;
    padding: 12px 16px; margin: 6px 0; font-size: 0.86rem;
    color: #d1fae5; max-width: 80%; white-space: pre-wrap;
}
.msg-label {
    font-size: 0.62rem; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 4px; opacity: 0.45;
}
.approval-box {
    background: #1a1400; border: 1px solid #f59e0b;
    border-radius: 8px; padding: 14px 16px; margin: 10px 0;
    font-size: 0.84rem; color: #fde68a;
}
.approval-label {
    font-size: 0.62rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #f59e0b; margin-bottom: 6px;
}
.tool-badge {
    display: inline-block; background: #292300;
    border: 1px solid #f59e0b; border-radius: 4px;
    padding: 1px 8px; font-size: 0.78rem; color: #fbbf24;
    font-family: 'DM Mono', monospace; margin: 2px 0;
}
div[data-testid="stTextInput"] input {
    background: #111118 !important; border: 1px solid #2a2a38 !important;
    border-radius: 8px !important; color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.15) !important;
}
.stButton > button {
    background: #0f1f17 !important; border: 1px solid #22c55e !important;
    color: #22c55e !important; border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important;
}
.stButton > button:hover { background: #22c55e !important; color: #0a0a0f !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="city-title">🏙 City Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="city-subtitle">weather · news · powered by mistral</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],        # list of {role, content} for display
    "pending_tool_calls": [],  # tool calls waiting for approval
    "agent_messages": [],      # full langgraph message history
    "awaiting_approval": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get current weather of city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    data = requests.get(url).json()
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = tavily_client.search(query=f"latest news in {city}", search_depth="basic", max_results=3)
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url   = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

TOOLS = {"get_weather": get_weather, "get_news": get_news}

# ── Agent (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatMistralAI(model="mistral-small-2506")
    return create_react_agent(
        llm,
        tools=list(TOOLS.values()),
        prompt="You are a helpful city assistant.",
    )

agent = build_agent()

# ── Helper: run tool calls and return ToolMessages ────────────────────────────
def execute_tool_calls(tool_calls):
    results = []
    for tc in tool_calls:
        fn = TOOLS.get(tc["name"])
        if fn:
            output = fn.invoke(tc["args"])
        else:
            output = f"Unknown tool: {tc['name']}"
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return results

# ── Helper: stream one agent step ─────────────────────────────────────────────
def run_agent_step(messages):
    """
    Run the agent with the given messages.
    Returns (final_reply: str | None, pending_tool_calls: list).
    If the agent wants to call tools → returns (None, [tool_calls]).
    If the agent gives a final answer → returns (reply_text, []).
    """
    result = agent.invoke({"messages": messages})
    last = result["messages"][-1]

    # Agent made tool calls
    if isinstance(last, AIMessage) and last.tool_calls:
        return None, last.tool_calls, result["messages"]

    # Final text answer
    return last.content, [], result["messages"]

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-label">you</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-bot"><div class="msg-label">agent</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# ── Approval UI ───────────────────────────────────────────────────────────────
if st.session_state.awaiting_approval and st.session_state.pending_tool_calls:
    tcs = st.session_state.pending_tool_calls
    names = ", ".join(f'<span class="tool-badge">{tc["name"]}({tc["args"]})</span>' for tc in tcs)
    st.markdown(
        f'<div class="approval-box">'
        f'<div class="approval-label">⚠ tool call approval required</div>'
        f'Agent wants to run: {names}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", key="approve"):
            # Execute tools, append results, continue agent
            tool_results = execute_tool_calls(st.session_state.pending_tool_calls)
            st.session_state.agent_messages.extend(tool_results)
            st.session_state.pending_tool_calls = []
            st.session_state.awaiting_approval = False

            # Continue agent with tool results
            with st.spinner("Running..."):
                reply, more_tcs, updated_msgs = run_agent_step(st.session_state.agent_messages)

            st.session_state.agent_messages = updated_msgs

            if more_tcs:
                # Another round of tool calls needs approval
                st.session_state.pending_tool_calls = more_tcs
                st.session_state.awaiting_approval = True
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            st.rerun()

    with col2:
        if st.button("❌ Deny", key="deny"):
            # Inject denied ToolMessages for each pending call
            for tc in st.session_state.pending_tool_calls:
                st.session_state.agent_messages.append(
                    ToolMessage(content="Tool call denied by user.", tool_call_id=tc["id"])
                )
            st.session_state.pending_tool_calls = []
            st.session_state.awaiting_approval = False
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I wasn't allowed to run those tools. Let me know how else I can help.",
            })
            st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Message",
        placeholder="Ask about weather or news in any city...",
        label_visibility="collapsed",
        disabled=st.session_state.awaiting_approval,
    )
    submitted = st.form_submit_button("Send →", disabled=st.session_state.awaiting_approval)

if submitted and user_input.strip():
    user_text = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    # Build message list: prior agent messages + new user message
    messages = st.session_state.agent_messages + [HumanMessage(content=user_text)]

    with st.spinner("Thinking..."):
        try:
            reply, tool_calls, updated_msgs = run_agent_step(messages)
            st.session_state.agent_messages = updated_msgs

            if tool_calls:
                st.session_state.pending_tool_calls = tool_calls
                st.session_state.awaiting_approval = True
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"⚠ Error: {e}"})

    st.rerun()
