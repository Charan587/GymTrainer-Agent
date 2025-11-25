from adapters import AgentAdapter
from gym_trainer_agent import GymTrainerAgent


class GymTrainerAgentAdapter(AgentAdapter):
    """Adapter for the GymTrainerAgent."""

    def __init__(self):
        self.agent = GymTrainerAgent()

    def get_answer(self, prompt, user_info=None):
        return self.agent.get_answer(prompt, user_info)  # type: ignore

    def reset_chat_history(self):
        self.agent.reset_chat_history()
