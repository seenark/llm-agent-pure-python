# Pure Python LLM Agent

This project demonstrates how to create a Language Learning Model (LLM) Agent using pure Python, leveraging only the OpenAI library. It showcases the implementation of AI agents that can understand, process, and respond to natural language commands.

## Project Structure

```
├── 01-intro/
│   ├── 01-basic.py       # Basic implementation concepts
│   ├── 02.structured.py  # Structured approach to agent development
│   ├── 03.tools.py       # Tool integration examples
│   ├── 04.retrieval.py   # Information retrieval capabilities
│   └── kb.json           # Knowledge base file
├── 02-workflow/
│   ├── 01-prompt-chaining.py  # Prompt chaining implementation
│   └── 02.routing.py          # Request routing logic
└── main.py              # Main application entry point
```

## Features

- Pure Python implementation
- OpenAI API integration
- Modular architecture
- Example implementations for:
  - Basic agent interactions
  - Structured responses
  - Tool usage
  - Information retrieval
  - Prompt chaining
  - Request routing

## Prerequisites

- Python 3.x
- OpenAI API key
- uv (Python package installer)

## Setup

1. Clone the repository
2. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install dependencies using uv:
   ```
   uv sync
   ```

## Usage

Explore the different modules in the project to understand various aspects of LLM Agent implementation:

1. Start with the basic concepts in `01-intro/01-basic.py`
2. Progress through the structured approach in `02.structured.py`
3. Learn about tool integration in `03.tools.py`
4. Explore information retrieval in `04.retrieval.py`
5. Study advanced workflows in the `02-workflow/` directory

## License

MIT