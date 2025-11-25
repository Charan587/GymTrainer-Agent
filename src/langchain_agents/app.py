import streamlit as st
from dotenv import load_dotenv

# Import the adapters instead of direct agent imports
from adapters import (
    ConversationalQAAgentAdapter,
    GymTrainerAgentAdapter,
)

load_dotenv()


# Modify the agent loading functions
@st.cache_resource
def load_qa_agent():
    return ConversationalQAAgentAdapter()


@st.cache_resource
def load_gym_agent():
    return GymTrainerAgentAdapter()


# Set page config
st.set_page_config(page_title="LangChain Agents", page_icon="🤖", layout="wide")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for agent selection
st.sidebar.title("Agent Selection")
agent_type = st.sidebar.radio(
    "Select an agent:", ["Conversational Q&A Agent", "Gym Trainer Agent"]
)

# Reset chat when changing agents
if "current_agent" not in st.session_state:
    st.session_state.current_agent = agent_type
elif st.session_state.current_agent != agent_type:
    st.session_state.messages = []
    st.session_state.current_agent = agent_type

# Display agent description based on selection
if agent_type == "Conversational Q&A Agent":
    st.sidebar.markdown("""
    **Conversational Q&A Agent**
    
    A simple agent that can answer general questions using GPT-3.5-turbo.
    This agent does not use RAG and relies solely on the model's knowledge.
    """)

    # Load the Q&A agent
    qa_agent = load_qa_agent()

    # Set the active agent
    active_agent = qa_agent

else:  # Gym Trainer Agent
    st.sidebar.markdown("""
    **Gym Trainer Agent**
    
    A specialized fitness coach that provides workout plans, nutrition advice,
    and exercise guidance using a RAG system with a fitness knowledge base.
    
    Features:
    - Workout plan generation
    - Nutrition recommendations
    - Form guidance
    - Personalized fitness advice
    """)

    # Add fitness level selector for gym trainer
    fitness_level = st.sidebar.select_slider(
        "Your fitness level:",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Intermediate",
    )

    fitness_goals = st.sidebar.multiselect(
        "Your fitness goals:",
        [
            "Weight loss",
            "Muscle gain",
            "Endurance",
            "Strength",
            "Flexibility",
            "General fitness",
        ],
        default=["General fitness"],
    )

    # Load the gym trainer agent
    gym_agent = load_gym_agent()

    # Set the active agent
    active_agent = gym_agent

# Main content area
st.title(f"🤖 {agent_type}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response based on selected agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if agent_type == "Conversational Q&A Agent":
                # Call with just the prompt
                response = active_agent.get_answer(prompt)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            else:  # Gym Trainer Agent
                user_info = {
                    "fitness_level": fitness_level,  # type: ignore
                    "goals": fitness_goals,  # type: ignore
                }
                # Call with both prompt and user_info
                result = active_agent.get_answer(prompt, user_info)
                st.markdown(result["formatted_answer"])  # type: ignore
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["formatted_answer"]}  # type: ignore
                )

# Add a reset button to clear chat history
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    if agent_type == "Gym Trainer Agent":
        active_agent.reset_chat_history()
    st.rerun()
