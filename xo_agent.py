#!/usr/bin/env python3
import os
import time
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from uagents_adapter import LangchainRegisterTool, cleanup_uagent
import logging
from rich.markdown import Markdown
from rich.console import Console
from flask import Flask, request, jsonify

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")
SEED_PHRASE = os.getenv("SEED_PHRASE")
if not AGENTVERSE_API_KEY:
    raise ValueError("Missing AGENTVERSE_API_KEY")

# Parameterized URLs
SOW_WEBHOOK_URL = os.getenv("SOW_WEBHOOK_URL")
SOW_FETCH_URL = os.getenv("SOW_FETCH_URL")
XO_CODER_API_URL = os.getenv("XO_CODER_API_URL")

# -----------------------------
# State Definition
# -----------------------------
class SOWState(TypedDict, total=False):
    user_id: str
    input: str
    sow_draft: str
    status: str
    next_node: str
    output: str
    history: List[BaseMessage]

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatGroq(model="gemma2-9b-it", temperature=0)

# -----------------------------
# Orchestrator Node
# -----------------------------
with open("orchestrator_prompt.txt", "r") as f:
    orchestrator_system_prompt = f.read()
orchestrator_prompt = ChatPromptTemplate.from_messages([
    ("system", orchestrator_system_prompt),
    ("human",
     "User message: {input}\n\n"
     "Current status: {status}\n\n"
     "Current SOW Draft:\n{sow_draft}\n\n"
     "Conversation History:\n{history}")
])

def orchestrator_node(state: SOWState) -> SOWState:
    history_msgs = state.get("history", [])
    formatted_history = "\n".join([m.content for m in history_msgs[-5:] if isinstance(m, HumanMessage)])
    decision = (orchestrator_prompt | llm).invoke({
        "input": state["input"],
        "status": state["status"],
        "sow_draft": state["sow_draft"],
        "history": formatted_history
    }).content.strip().lower()

    if decision not in ["sow_node", "approval_check", "general_query_node","xo_coder"]:
        decision = "general_query_node"

    return {**state, "next_node": decision}

# -----------------------------
# SOW Drafting Node
# -----------------------------
with open("sow_prompt.txt", "r") as f:
    sow_system_prompt = f.read()
sow_prompt = ChatPromptTemplate.from_messages([
    ("system", sow_system_prompt),
    ("human", "{history}\n\nUser input: {input}\n\nCurrent SOW: {sow_draft}")
])

def sow_node(state: SOWState) -> SOWState:
    history_msgs = state.get("history", [])
    formatted_history = "\n".join([f"User: {m.content}" for m in history_msgs if isinstance(m, HumanMessage)])

    new_sow = (sow_prompt | llm).invoke({
        "input": state["input"],
        "sow_draft": state["sow_draft"],
        "history": formatted_history
    }).content.strip()
    
    try:
        response = requests.post(SOW_WEBHOOK_URL, json={
            "sow_draft": new_sow
        })
        response.raise_for_status()  # Raises an exception for 4xx/5xx responses
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Webhook send failed: {e}")

    state["history"] = state.get("history", []) + [
        make_user_message(state["input"]),
        make_assistant_message(f"SOW updated:\n{new_sow}")
    ]

    return {
        **state,
        "sow_draft": new_sow,
        "status": "modifying",
        "output": new_sow,
        "next_node": "orchestrator"
    }

# -----------------------------
# Approval Check Node
# -----------------------------
with open("approval_prompt.txt", "r") as f:
    approval_system_prompt = f.read()
approval_prompt = ChatPromptTemplate.from_messages([
    ("system", approval_system_prompt),
    ("human",
     "User message: {input}\n\nRecent history:\n{history}")])

def approval_check_node(state: SOWState) -> SOWState:
    history_msgs = state.get("history", [])
    formatted_history = "\n".join([m.content for m in history_msgs[-5:] if isinstance(m, HumanMessage)])
    decision = (approval_prompt | llm).invoke({
        "input": state["input"],
        "history": formatted_history
    }).content.strip().lower()

    state["history"] = state.get("history", []) + [make_user_message(state["input"])]

    if decision.startswith("yes"):
        state["history"].append(make_assistant_message("Agent: Approved."))
        return {
            **state,
            "status": "approved",
            "output": f"✅ Final SOW:\n{state['sow_draft']}",
            "next_node": "collect_email_node"
        }

    state["history"].append(make_assistant_message("Agent: Not approved."))
    return {
        **state,
        "status": "modifying",
        "output": "❌ Not approved. Please share your changes.",
        "next_node": "orchestrator"
    }

# -----------------------------
# Claude code Node
# -----------------------------
with open("claude_prompt.txt", "r") as f:
    claude_system_prompt = f.read()
claude_prompt_template = ChatPromptTemplate.from_messages([
    ("system", claude_system_prompt),
    ("human", "Scope of Work: {input}")
])

def xo_coder_node(state: SOWState) -> SOWState:
    state["history"] = state.get("history", []) + [make_user_message(state["input"])]
    
    # Step 1: Fetch latest SOW from webhook
    try:
        sow_response = requests.get(SOW_FETCH_URL)
        sow_response.raise_for_status()
        sow_data = sow_response.json()
        sow_text = sow_data.get("sow", "").strip()
        if not sow_text:
            raise ValueError("Empty SOW received from store-agent webhook.")
    except Exception as e:
        error_msg = f"❌ Failed to fetch SOW from webhook: {e}"
        state["history"].append(make_assistant_message(error_msg))
        return {
            **state,
            "output": error_msg,
            "next_node": "orchestrator"
        }

    # Step 2: Generate Claude-ready prompt using the fetched SOW
    generated_prompt = (claude_prompt_template | llm).invoke({
        "input": sow_text+"\n"+state['input']
    }).content.strip()

    # Step 3: Send to Claude code generation API
    payload = {
        "prompt": generated_prompt,
        "user_id": state["user_id"]
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(XO_CODER_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = f"✅ XO Code Generation Successful:\n{result}"
    except requests.exceptions.RequestException as e:
        response_text = f"❌ XO Coder API request failed: {e}"

    state["history"].append(make_assistant_message(f"Agent: {response_text}"))

    return {
        **state,
        "output": f"XO is working on it: {generated_prompt}",
        "next_node": "orchestrator"
    }

# -----------------------------
# General Query Node
# -----------------------------
with open("xo_context_prompt.txt", "r") as f:
    xo_context_system_prompt = f.read()
xo_context_prompt = ChatPromptTemplate.from_messages([
    ("system", xo_context_system_prompt),
    ("human","{history}\n\n{input}")])

def general_query_node(state: SOWState) -> SOWState:
    history_msgs = state.get("history", [])
    formatted_history = "\n".join([f"User: {m.content}" for m in history_msgs if isinstance(m, HumanMessage)])

    response = (xo_context_prompt | llm).invoke({
        "input": state["input"],
        "history": formatted_history
    }).content.strip()

    state["history"] = state.get("history", []) + [
        make_user_message(state["input"]),
        make_assistant_message(f"Agent: {response}")
    ]

    return {
        **state,
        "output": response,
        "next_node": "orchestrator"
    }

# -----------------------------
# Graph Build
# -----------------------------
builder = StateGraph(SOWState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("sow_node", sow_node)
builder.add_node("approval_check", approval_check_node)
builder.add_node("general_query_node", general_query_node)
builder.add_node("xo_coder", xo_coder_node)

builder.set_entry_point("orchestrator")

builder.add_conditional_edges("orchestrator", lambda s: s["next_node"], {
    "sow_node": "sow_node",
    "approval_check": "approval_check",
    "general_query_node": "general_query_node",
    "xo_coder": "xo_coder"
})

builder.add_edge("xo_coder", END)
builder.add_edge("sow_node", END)
builder.add_edge("approval_check", END)
builder.add_edge("general_query_node", END)

graph = builder.compile()

console = Console()

app = Flask(__name__)

# In-memory state store for demo (keyed by user_id)
user_states = {}

def get_initial_state(user_id):
    return {
        "user_id": user_id,
        "input": "",
        "sow_draft": "",
        "status": "drafting",
        "history": [],
        "next_node": "",
        "output": ""
    }

def make_user_message(text):
    return HumanMessage(content=text, role="user")

def make_assistant_message(text):
    return HumanMessage(content=text, role="assistant")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id", "user_xyz")
    user_input = data.get("input", "")
    if not user_input:
        return jsonify({"error": "Missing input"}), 400

    # Get or initialize state
    state = user_states.get(user_id) or get_initial_state(user_id)
    state["input"] = user_input

    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = graph.invoke(state, config={"recursion_limit": 100})
    output = result.get("output", "⚠️ No output generated by this step.")
    status = result.get("status", "")

    # Update state for next turn
    user_states[user_id] = result

    return jsonify({
        "output": output,
        "status": status,
        "sow_draft": result.get("sow_draft", ""),
        "history": [
            {"role": getattr(m, 'role', 'user'), "content": m.content}
            for m in result.get("history", []) if hasattr(m, 'content')
        ],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True) 