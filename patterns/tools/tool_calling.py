"""
Tool calling 101: teach the model to call our Python functions.

Flow:
1. We describe our tools (name, description, JSON schema for args) to the model.
2. The model decides whether it needs a tool, and replies with a "tool_calls" request
   instead of (or before) a final answer.
3. We actually run the requested Python function ourselves and send the result back.
4. The model reads the result and writes the final answer.
"""

import json
import os
import requests

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------------------
# Step 1: Define real Python functions the model can call
# --------------------------------------------------------------

def add(a: float, b: float) -> float:
    return a + b



def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 2: Describe those functions to the model as "tools"
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature and wind speed for a location, given its latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude of the location"},
                    "longitude": {"type": "number", "description": "Longitude of the location"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# Map tool name -> actual Python function to run
available_functions = {
    "add": add,
    "get_weather": get_weather,
}


# --------------------------------------------------------------
# Step 3: The agent loop
# --------------------------------------------------------------

def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # If the model didn't ask for a tool, we're done.
        if not msg.tool_calls:
            return msg.content

        # Otherwise, run each requested tool and feed the result back.
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"[tool call] {fn_name}({fn_args})")

            result = available_functions[fn_name](**fn_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    answer = run_agent(
        "What's the weather in Lahore (latitude 31.5497, longitude 74.3436), "
        "and what's 12 plus 30?"
    )
    print("\nFinal answer:", answer)
