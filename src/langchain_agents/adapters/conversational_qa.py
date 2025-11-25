from adapters import AgentAdapter
from conversational_qa_agent import ConversationalQAAgent


class ConversationalQAAgentAdapter(AgentAdapter):
    """Adapter for the ConversationalQAAgent."""

    def __init__(self):
        self.agent = ConversationalQAAgent()

    def get_answer(self, prompt, user_info=None):
        # Ignore user_info for this agent
        return self.agent.get_answer(prompt)

    def reset_chat_history(self):
        self.agent.reset_chat_history()
