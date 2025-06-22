
# %%
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import Field, BaseModel
import requests
from typing import Final
from openai import OpenAI

model_name: Final[str] = "llama3.2:3b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------

def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

## these tool copied from openai website about function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

user_prompt = "What's the weather like in Paris today?"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
)

# %%

json = completion.model_dump_json()
print(json)

# %%
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)

# %%
import json
def extract_function_call(tool_call: ChatCompletionMessageToolCall) -> tuple[str, dict]:
    name = tool_call.function.name
    args_json = tool_call.function.arguments
    args = json.loads(args_json)
    return name, args

    
# %%
tool_calls = completion.choices[0].message.tool_calls
if tool_calls is not None:
    for tool_call in tool_calls:
        name, args = extract_function_call(tool_call)

        result = call_function(name, args)
        print("result", result)

        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )
# %%
print("messages", messages)
# %%
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )
 # %%   
completion_2 = client.beta.chat.completions.parse(
    model=model_name,
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)
# %%
final_response = completion_2.choices[0].message.parsed
# %%
print("json",final_response.model_dump_json())
print()
temp = final_response.temperature
print("temp", temp)
print("")
res = final_response.response
print("res", res)
# %%