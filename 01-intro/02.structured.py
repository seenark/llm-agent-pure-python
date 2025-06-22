# %%
from typing import Final
from openai import OpenAI
from pydantic import BaseModel

model_name: Final[str] = "llama3.2:3b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model=model_name,
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed
print("event", event)
event.name
event.date
event.participants

# %%
