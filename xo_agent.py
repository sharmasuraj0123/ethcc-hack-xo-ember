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
orchestrator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent routing assistant inside a Scope of Work (SOW) generation agent. "
     "Your job is to decide which node should handle the user's next message.\n\n"

     "The graph has 4 main nodes:\n"
     "- 'sow_node': Use this when the user is clearly describing a project, listing features, suggesting revisions, or providing new technical requirements for a Scope of Work.\n"
     "- 'approval_check': Use this when the user indicates approval, satisfaction, or readiness to proceed. This can be explicit (e.g., 'I approve', 'Looks good', 'This works', 'Great', 'Proceed') "
     "or implicit (e.g., 'Alright', 'Perfect', 'Cool', 'Nice'). If a valid SOW draft already exists, treat these phrases as approval signals.\n"
     "- 'general_query_node': Use this when the user is asking general knowledge questions (e.g., 'What is an SOW?', 'What are typical timelines?'), or when the input is vague, off-topic, or a greeting (e.g., 'Hello', 'Hey', 'Can you help?', 'Are you there?').\n\n"
     "- 'xo_coder': Use this when the user mentions code generation, Coder, creating or deploying websites/apps, or similar tasks like 'create a dashboard', 'build a site', 'generate code'.\n\n"

     "Only respond with EXACTLY ONE of:\n"
     "- sow_node\n"
     "- approval_check\n"
     "- general_query_node\n"
     "- xo_coder"
     
     "IMPORTANT:\n"
     "- DO NOT route to 'sow_node' if the message is vague or unclear.\n"
     "- DO NOT guess technical intent from short inputs like 'Ok', 'Yup', or 'Hey'. Use context and history.\n"
     "- If you're unsure or the message looks like a greeting or generic question, default to 'general_query_node'.\n\n"
    ),
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
sow_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that writes or updates a professional SOW."),
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
approval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an approval-checking assistant for a Scope of Work (SOW) generator.\n"
     "Your job is to decide if the user has approved the final version of the SOW.\n"
     "Approval might be expressed with phrases like: 'Looks good', 'Perfect', 'Finalize it', 'Ready', 'this works', etc.\n"
     "If the user is still asking for changes or expressing doubts, it's not approved.\n"
     "You must respond with only 'yes' or 'no'."),
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
claude_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a developer assistant. Your task is to generate a clear, natural-sounding prompt to send to Claude for code generation.\n\n"
     "Base this prompt entirely on the provided Scope of Work and any additional user instructions.\n\n"
     "The final prompt should:\n"
     "- Be written naturally, like a human request.\n"
     "- Clearly define what needs to be built or done.\n"
     "- Include goals, frameworks, or components if mentioned.\n"
     "- Avoid unnecessary repetition or verbose explanations.\n"
     "- Stay within 100 to 160 words total.\n\n"
     "Focus on clarity, specificity, and brevity to help Claude generate precise code."),
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
xo_context_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant inside XO, an AI-powered software platform based in New York.\n\n"
     "About XO:\n"
     "- Mission: Build smarter. Launch faster. Stay ahead.\n"
     "- XO eliminates the chaos of AI/Web3 software development by orchestrating a swarm of agents.\n"
     "- Agents handle UI/UX, backend, DevOps, blockchain, etc., all from a single prompt.\n"
     "- Products include MVP builders, no-code tools, founder agents, and blockchain modules.\n"
     "- Typical users: founders, enterprises, developers.\n"
     "- Access: xo.builders | demo.xo.builders | LinkedIn demos\n"
     "- Team: Suraj Sharma (ex-Microsoft), Yash Sanghvi (ex-BlackRock)\n"
     "- Competitive Edge: Full-stack agentic workflows, not snippet-based responses.\n"
     "- Example claims: MVPs live in hours, 70–80% workflow acceleration, $30K–90K/year savings.\n\n"
     "Answer general questions about XO or related topics clearly and concisely. If the question is not about XO, still respond helpfully."
    ),
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