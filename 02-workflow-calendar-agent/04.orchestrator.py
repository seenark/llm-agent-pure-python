# %%
# ╞╡ setup logging ╞═══════════════════════════════════════════════════╡
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# %%
# ╞╡ setup llm ╞═══════════════════════════════════════════════════════╡
import os
from openai import OpenAI
from typing import Final

openrouter_token = os.getenv("OPENROUTER_API_KEY")
openrouter_url = "https://openrouter.ai/api/v1"

ollama_url = "http://localhost:11434/v1"
ollama_token = "ollama"

openrouter_deepseek_model_name: Final[str] = "deepseek/deepseek-r1-0528-qwen3-8b:free"
openrouter_model_name: Final[str] = "mistralai/mistral-small-3.2-24b-instruct:free"

ollama_model_name: Final[str] = "llama3.2:3b"
ollama_deepseek_model_name: Final[str] = "deepseek-r1:8b"

model_name = openrouter_model_name
deepseek_model_name = openrouter_deepseek_model_name

client = OpenAI(base_url=openrouter_url, api_key=openrouter_token)

# %%
from pydantic import BaseModel, Field
from typing import List

ORCHESTRATOR_PROMPT = """
Analyze this blog topic and break it down into logical sections.

Topic: {topic}
Target Length: {target_length} words
Style: {style}

Return your response in this format:

# Analysis
Analyze the topic and explain how it should be structured.
Consider the narrative flow and how sections will work together.

# Target Audience
Define the target audience and their interests/needs.

# Sections
## Section 1
- Type: section_type
- Description: what this section should cover
- Style: writing style guidelines

[Additional sections as needed...]
"""


class SubTask(BaseModel):
    """Blog section task defined by orchestrator"""

    section_type: str = Field(description="Type of blog section to write")
    description: str = Field(description="What this section should cover")
    style_guide: str = Field(description="Writing style for this section")
    target_length: int = Field(description="Target word count for this section")


# ╞╡ Orchestrator Plan ╞═══════════════════════════════════════════════╡
class OrchestratorPlan(BaseModel):
    """Orchestrator's blog structure and tasks"""

    topic_analysis: str = Field(description="Analysis of the blog topic")
    target_audience: str = Field(description="Intended audience for the blog")
    sections: List[SubTask] = Field(description="List of sections to write")


def get_plan(topic: str, target_length: int, style: str) -> OrchestratorPlan:
    """Get orchestrator's blog structure plan"""
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": ORCHESTRATOR_PROMPT.format(
                    topic=topic, target_length=target_length, style=style
                ),
            }
        ],
        response_format=OrchestratorPlan,
    )
    return completion.choices[0].message.parsed


# %%
# ╞╡ generate section content ╞════════════════════════════════════════╡

WORKER_PROMPT = """
Write a blog section based on:
Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}

Return your response in this format:

# Content
[Your section content here, following the style guide]

# Key Points
- Main point 1
- Main point 2
[Additional points as needed...]
"""


class SectionContent(BaseModel):
    """Content written by a worker"""

    content: str = Field(description="Written content for the section")
    key_points: List[str] = Field(description="Main points covered")


def write_section(
    topic: str, section: SubTask, sections_content: dict
) -> SectionContent:
    """Worker: Write a specific blog section with context from previous sections.

    Args:
        topic: The main blog topic
        section: SubTask containing section details

    Returns:
        SectionContent: The written content and key points
    """
    # Create context from previously written sections
    previous_sections = "\n\n".join(
        [
            f"=== {section_type} ===\n{content.content}"
            for section_type, content in sections_content.items()
        ]
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": WORKER_PROMPT.format(
                    topic=topic,
                    section_type=section.section_type,
                    description=section.description,
                    style_guide=section.style_guide,
                    target_length=section.target_length,
                    previous_sections=previous_sections
                    if previous_sections
                    else "This is the first section.",
                ),
            }
        ],
        response_format=SectionContent,
    )
    return completion.choices[0].message.parsed


# %%
# ╞╡ Review content ╞══════════════════════════════════════════════════╡

REVIEWER_PROMPT = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {audience}

Sections:
{sections}

Provide a cohesion score between 0.0 and 1.0, suggested edits for each section if needed, and a final polished version of the complete post.

The cohesion score should reflect how well the sections flow together, with 1.0 being perfect cohesion.
For suggested edits, focus on improving transitions and maintaining consistent tone across sections.
The final version should incorporate your suggested improvements into a polished, cohesive blog post.
"""


class SuggestedEdits(BaseModel):
    """Suggested edits for a section"""

    section_name: str = Field(description="Name of the section")
    suggested_edit: str = Field(description="Suggested edit")


class ReviewFeedback(BaseModel):
    """Final review and suggestions"""

    cohesion_score: float = Field(description="How well sections flow together (0-1)")
    suggested_edits: List[SuggestedEdits] = Field(
        description="Suggested edits by section"
    )
    final_version: str = Field(description="Complete, polished blog post")


def review_post(
    topic: str, plan: OrchestratorPlan, sections_content: dict
) -> ReviewFeedback:
    """Reviewer: Analyze and improve overall cohesion"""
    sections_text = "\n\n".join(
        [
            f"=== {section_type} ===\n{content.content}"
            for section_type, content in sections_content.items()
        ]
    )

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": REVIEWER_PROMPT.format(
                    topic=topic,
                    audience=plan.target_audience,
                    sections=sections_text,
                ),
            }
        ],
        response_format=ReviewFeedback,
    )
    return completion.choices[0].message.parsed


# %%
from typing import Dict

# ╞╡ write post ╞══════════════════════════════════════════════════════╡
def write_blog(
    topic: str,
    sections_content: dict,
    target_length: int = 200,
    style: str = "informative",
) -> Dict:
    """Process the entire blog writing task"""
    logger.info(f"Starting blog writing process for: {topic}")

    # Get blog structure plan
    plan = get_plan(topic, target_length, style)
    logger.info(f"Blog structure planned: {len(plan.sections)} sections")
    logger.info(f"Blog structure planned: {plan.model_dump_json(indent=2)}")

    # Write each section
    for section in plan.sections:
        logger.info(f"Writing section: {section.section_type}")
        content = write_section(topic, section, sections_content=sections_content)
        sections_content[section.section_type] = content

    # # Review and polish
    logger.info("Reviewing full blog post")
    review = review_post(topic, plan, sections_content=sections_content)

    # return {"structure": plan, "sections": sections_content, "review": ""}
    return {"structure": plan, "sections": sections_content, "review": review}

# %%
# ╞╡ create blog ╞═════════════════════════════════════════════════════╡
topic = "Different between primitive types and reference types in Typescript"
sections_content = {}

result = write_blog(topic, sections_content=sections_content)
# %%
print("result", result)
# %%
print("sections",result["sections"])

print()

print("review", result["review"])

# %%
