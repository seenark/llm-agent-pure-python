# %%
# setup loggin
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# %%
from openai import OpenAI
from typing import Final

model_name: Final[str] = "llama3.2:3b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
# %%
from pydantic import BaseModel, Field
# from user question let LLM extract the event information from that question
class EventExtractionFromUserPrompt(BaseModel):
    """First LLM call: Extract basic event information from user prompt"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")

from datetime import datetime
def extract_event_info_from_user_prompt(user_input: str) -> EventExtractionFromUserPrompt:
    """First LLM call to determine if input is a calendar event"""
    
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event. think step by step",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtractionFromUserPrompt,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result
# %%
class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected integer duration in minutes without unit")
    participants: list[str] = Field(description="List of participants")

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result
    
# %%
from typing import Optional
class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign of with your name; Susie",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result
# %%

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info_from_user_prompt(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation

# %%
model_name: Final[str] = "deepseek-r1:8b"
deepseek_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

today = datetime.now()
date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

expand_input_completion = deepseek_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": f"""
                {date_context} Analyze if the text describes a calendar event.
                then expand the event description about event details to be let small LLM understand intension of user input.
                your result will pass to llama3.2:3b to generate confirmation message.
                try to create event name.
                try to get participants, if you cloud not get it please do not specify any paticipants.
                try to get date and time for the event, if you cloud not get it please do not specify date and time.
                try to get how long in minutes for the event, if you cloud not get it please do not specify how long.
                Do not add other information.
                """,
            },
            {"role": "user", "content": user_input},
        ]
    )

expanded_content = expand_input_completion.choices[0].message.content
print("expanded_content", expanded_content)

# %%
import re
def clean_think(input: str) -> str:
    return re.sub(r"<think>.*?</think>", "", input, flags=re.DOTALL)

expanded_content_cleaned = clean_think(expanded_content)
print("expanded_content_cleaned", expanded_content_cleaned)
# %%
result = process_calendar_request(expanded_content_cleaned)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
# %%
# --------------------------------------------------------------
# Test the chain with an invalid input
# --------------------------------------------------------------

user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")
# %%
