# LangChain Agents

A versatile AI assistant platform built with LangChain and Streamlit, featuring specialized agents for different domains.



## Overview

This project demonstrates the implementation of multiple AI agents using the LangChain framework. Each agent is specialized for a specific domain and can maintain contextual conversations with users. The application is built with a Streamlit interface for easy interaction.

## Features

### Conversational Q&A Agent
- General-purpose conversational AI assistant
- Powered by OpenAI's GPT-3.5 Turbo model
- Maintains conversation history for contextual responses
- Simple and straightforward interface for general questions

### Gym Trainer Agent
- Specialized fitness coach with domain expertise
- Powered by OpenAI's GPT-4o model
- Uses Retrieval Augmented Generation (RAG) with a fitness knowledge base
- Provides personalized advice based on fitness level and goals
- Features include:
  - Workout plan generation
  - Nutrition recommendations
  - Form guidance
  - Personalized fitness advice
- Formats responses for better readability (workout plans, nutrition advice)

## Architecture

The project follows a modular architecture with adapter pattern implementation:

- **Base Adapter Interface**: Defines the common interface for all agents
- **Agent-Specific Implementations**: Each agent has its own implementation with specialized features
- **Streamlit UI**: Provides a user-friendly interface for interacting with the agents

### Key Components

- **Adapter Pattern**: Standardizes agent interactions through a common interface
- **RAG System**: The Gym Trainer Agent uses a vector database with fitness knowledge
- **Chat History**: Both agents maintain conversation history for contextual responses
- **Response Formatting**: Custom formatting for domain-specific outputs (e.g., workout plans)

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **Streamlit**: Web interface for user interaction
- **OpenAI Models**: GPT-3.5 Turbo and GPT-4o
- **ChromaDB**: Vector database for the RAG system
- **Python 3.10+**: Core programming language
- **Poetry**: Dependency management

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (dependency management)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jdev-king/langchain-agents.git
cd langchain-agents
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Activate the Poetry virtual environment:
```bash
poetry shell
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit application:

```bash
poetry run streamlit run src/langchain_agents/app.py
```

Alternatively, if you're already in the Poetry shell:

```bash
streamlit run src/langchain_agents/app.py
```

The application will open in your default web browser. From there, you can:

1. Select an agent type from the sidebar (Conversational Q&A or Gym Trainer)
2. For the Gym Trainer, set your fitness level and goals
3. Ask questions in the chat interface
4. Reset the conversation using the button in the sidebar

## Examples

### Conversational Q&A Agent

- "What are the benefits of learning Python?"
- "Can you explain how neural networks work?"
- "Tell me about the history of artificial intelligence."

### Gym Trainer Agent

- "Can you create a workout plan for building muscle?"
- "What should I eat before a workout?"
- "How do I improve my squat form?"
- "Give me a nutrition plan for weight loss."

## Project Structure

```
langchain-agents/
├── src/
│   └── langchain_agents/
│       ├── adapters/            # Adapter pattern implementation
│       ├── conversational_qa_agent/  # General Q&A agent
│       ├── gym_trainer_agent/   # Specialized fitness agent
│       │   ├── data/            # Fitness knowledge base
│       │   └── vectorstore/     # Vector database for RAG
│       └── app.py              # Streamlit application
├── tests/                      # Test suite
└── pyproject.toml             # Poetry configuration
```

## Author

Jesus Diez - [jesussebastiandiezplasencia@gmail.com](mailto:jesussebastiandiezplasencia@gmail.com)

---

Built with ❤️ using LangChain and OpenAI
