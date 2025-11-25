from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()


# Define the agent class for better encapsulation
class ConversationalQAAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        template = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
        human_template = "{text}"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", human_template),
            ]
        )
        self.model = ChatOpenAI(model=model_name)
        self.chain = self.prompt | self.model | StrOutputParser()
        self.chat_history = []

    def get_answer(self, question):
        """Process a question and return the answer

        Args:
            question (str): The user's question
            user_info (dict, optional): Additional user information (ignored for this agent)
        """
        # Ignore user_info for this agent
        answer = self.chain.invoke(
            {"text": question, "chat_history": self.chat_history}
        )

        # Update chat history
        self.chat_history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )

        return answer

    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = []


# Create a default instance for easy importing
default_agent = ConversationalQAAgent()


# For backward compatibility
def get_answer(question):
    return default_agent.get_answer(question)


def reset_chat_history():
    default_agent.reset_chat_history()
