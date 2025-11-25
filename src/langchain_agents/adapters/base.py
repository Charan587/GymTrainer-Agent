class AgentAdapter:
    """Base adapter interface for all agent implementations."""
    
    def get_answer(self, prompt, user_info=None):
        """Process a prompt and return an answer.
        
        Args:
            prompt (str): The user's prompt or question
            user_info (dict, optional): Additional user information
            
        Returns:
            The agent's response (format depends on the specific adapter)
        """
        raise NotImplementedError("Subclasses must implement get_answer()")

    def reset_chat_history(self):
        """Reset the agent's chat history."""
        raise NotImplementedError("Subclasses must implement reset_chat_history()")