# %%

from typing import Final
from openai import OpenAI

model_name: Final[str] = "llama3.2:3b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
# %%
def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)
# %%
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
# %%
system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
)
# %%
res_1 = completion.model_dump()
# %%
print("res1", res_1)
# %%
def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)
# %%
import json
from openai.types.chat import ChatCompletionMessageToolCall

def extract_function_call(tool_call: ChatCompletionMessageToolCall) -> tuple[str, dict]:
    name = tool_call.function.name
    args_json = tool_call.function.arguments
    args = json.loads(args_json)
    return name, args

# %%
for tool_call in completion.choices[0].message.tool_calls:
    name, args = extract_function_call(tool_call)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    print("result", result)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )
# %%
print("messages", messages)
    
# %%
from pydantic import BaseModel, Field

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")
# %%
completion_2 = client.beta.chat.completions.parse(
    model=model_name,
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)
# %%
res_2 = completion_2.model_dump_json()
print("res_2", res_2)
# %%

res_2_parsed = completion_2.choices[0].message.parsed
print("res 2 parsed:", res_2_parsed)
answer = res_2_parsed.answer 
print("answer:", answer)
source = res_2_parsed.source
print("source:", source)
# %%
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
)
# %%
res_3 = completion_3.choices[0]
print("res3", res_3)
# %%
