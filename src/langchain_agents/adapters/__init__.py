# Import all adapters for easy access
from .base import AgentAdapter
from .conversational_qa import ConversationalQAAgentAdapter
from .gym_trainer import GymTrainerAgentAdapter

# For backward compatibility
__all__ = ["AgentAdapter", "ConversationalQAAgentAdapter", "GymTrainerAgentAdapter"]
