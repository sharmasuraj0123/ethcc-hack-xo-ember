from datetime import datetime
from uuid import uuid4
from uagents import Agent, Protocol, Context
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

#import the necessary components from the chat protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
# Initialise agent2
agent2 = Agent(
    name="test-agent-uagent",
    seed="test-agent-uagent1234",
    mailbox=True,
    publish_agent_details=True,
)

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)
XO_AGENT_URL = os.getenv("XO_AGENT_URL")
user_id = "test-agent-uagent-12345"

# Startup Handler - Print agent details
@agent2.on_event("startup")
async def startup_handler(ctx: Context):
    # Print agent details
    ctx.logger.info(f"My name is {ctx.agent.name} and my address is {ctx.agent.address}")

# Message Handler - Process received messages and send acknowledgements
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            # Log received message
            ctx.logger.info(f"Received message from {sender}: {item.text}")
            
            # Send acknowledgment
            ack = ChatAcknowledgement(
                timestamp=datetime.utcnow(),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, ack)

        
        payload_text = " ".join([item.text for item in msg.content if isinstance(item, TextContent)]).strip()
        if not payload_text:
            # Optionally, you can log or handle the empty input case here, but do not send a fallback error message
            ctx.logger.info(f"Received empty input from {sender}, skipping XO API call.")
            return

        payload = {
            "input": payload_text,
            "user_id": str(ctx.session)
        }
        headers = {
            "Content-Type": "application/json"
        }

        print("\n--- XO API Request Payload ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("--- End Payload ---\n")

        try:
            response = requests.post(XO_AGENT_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            # Extract the 'output' field from the response
            xo_output = result.get("output")
            if xo_output:
                response_text = f"✅ XO Code Generation Successful:\n{xo_output}"
                print(response_text)
                # Send XO output as the response message
                response_msg = ChatMessage(
                    timestamp=datetime.utcnow(),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=xo_output)]
                )
                await ctx.send(sender, response_msg)
                return
            else:
                response_text = f"❌ XO API response missing 'output': {result}"
                print(response_text)
        except requests.exceptions.RequestException as e:
            response_text = f"❌ XO API request failed: {e}"
            print(response_text)
            
        # Send fallback response message
        response = ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"Something went wrong")]
        )
        await ctx.send(sender, response)

# Acknowledgement Handler - Process received acknowledgements
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# Include the protocol in the agent to enable the chat functionality
# This allows the agent to send/receive messages and handle acknowledgements using the chat protocol
agent2.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    agent2.run()