import os
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class GymTrainerAgent:
    def __init__(self, model_name="gpt-4o"):
        load_dotenv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, "vectorstore", "chroma_gym_db")

        # Initialize embeddings and vector store
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.db = Chroma(
            persist_directory=persist_directory, embedding_function=self.embedding
        )

        # Configure MMR retriever
        self.retriever = self.db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 7,  # Retrieve more documents initially
                "fetch_k": 15,  # Consider top 15 documents for diversity calculation
                "lambda_mult": 0.7,  # Balance between relevance (1.0) and diversity (0.0)
            },
        )

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name)

        # Set up the contextual question reformulation
        contextualize_q_system_prompt = """
            Given a chat history and the latest user question
            which might reference context in the chat history,
            formulate a standalone question which can be understood
            without the chat history. Do NOT answer the question, just
            reformulate it if needed and otherwise return it as is.
        """

        # Create a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # This uses the LLM to help reformulate the question based on chat history
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # Define the QA system prompt
        qa_system_prompt = """
           You are a certified personal trainer and fitness coach with expertise in exercise science, nutrition, and wellness. Use the provided context to deliver clear, professional fitness advice tailored to individual goals and questions.

            USER PROFILE:
            {user_profile}
           
            GENERAL GUIDELINES:
            - Prioritize proper form, injury prevention, and evidence-based practices.
            - Provide specific, actionable recommendations tailored to the user's fitness level and objectives.
            - Explain both what to do and why it matters, including the rationale behind each recommendation.
            - Adapt all advice for different fitness levels (beginner, intermediate, advanced).
            - Be honest about limitations—if the context lacks sufficient information, say so and suggest consulting a qualified professional.
            - NEVER offer medical advice, diagnose injuries, or recommend treatment protocols.
            - If the user does not explicitly ask for a workout plan, don't provide one.

            RESPONSE FORMATS:

            For workout plans, use this structure:
            - OBJECTIVE: [Clearly state the primary goal, e.g., hypertrophy, fat loss, endurance, strength]
            - WORKOUT STRUCTURE: [Outline the training format, e.g., full body, push/pull/legs, upper/lower split]
            - EXERCISES: [List each exercise with proper form cues and equipment needed]
            - SETS/REPS: [Provide the number of sets and reps for each exercise]
            - REST PERIODS: [Recommend rest times between sets and exercises]
            - NOTES: [Additional tips on progression, warm-up, cool-down, frequency, or modifications]

            For nutrition guidance, use this structure:
            - RECOMMENDATION: [Summary of dietary advice based on the user's goals]
            - MACROS: [Guidance on protein, carbohydrates, and fat intake]
            - TIMING: [Recommendations on when to eat relative to activity]
            - ALTERNATIVES: [Suggested food swaps or dietary options]

            For form corrections, use clear anatomical descriptions to explain proper movement and posture.

            Always emphasize progression, safety, and realistic goal-setting based on user input. If information is missing or unclear, respond truthfully and suggest consulting a qualified fitness professional.

            
            {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

        # Initialize chat history
        self.chat_history = []

    def format_response(self, response_text):
        """Format the response for better readability, especially for workout plans."""
        # Check if this is a workout plan response
        if "OBJECTIVE:" in response_text and "EXERCISES:" in response_text:
            # Split the response into sections
            sections = [
                "OBJECTIVE:",
                "WORKOUT STRUCTURE:",
                "EXERCISES:",
                "SETS/REPS:",
                "REST PERIODS:",
                "NOTES:",
            ]

            formatted_text = response_text
            for section in sections:
                if section in formatted_text:
                    # Add proper formatting to section headers
                    formatted_text = formatted_text.replace(
                        section, f"\n\n**{section.strip(':')}\**"
                    )

            # Format exercise lists if present
            if "**EXERCISES**" in formatted_text:
                exercise_section = formatted_text.split("**EXERCISES**")[1].split("**")[
                    0
                ]
                exercises = exercise_section.strip().split("\n")
                formatted_exercises = "\n\n**EXERCISES**\n"

                for exercise in exercises:
                    if exercise.strip() and ":" in exercise:
                        ex_name, ex_desc = exercise.split(":", 1)
                        formatted_exercises += (
                            f"- **{ex_name.strip()}**: {ex_desc.strip()}\n"
                        )
                    elif exercise.strip():
                        formatted_exercises += f"- {exercise.strip()}\n"

                formatted_text = formatted_text.replace(
                    exercise_section, formatted_exercises
                )

            return formatted_text

        # Check if this is a nutrition advice response
        elif "RECOMMENDATION:" in response_text and "MACROS:" in response_text:
            # Split the response into sections
            sections = ["RECOMMENDATION:", "MACROS:", "TIMING:", "ALTERNATIVES:"]

            formatted_text = response_text
            for section in sections:
                if section in formatted_text:
                    # Add proper formatting to section headers
                    formatted_text = formatted_text.replace(
                        section, f"\n\n**{section.strip(':')}\**"
                    )

            return formatted_text

        # Return the original text if no special formatting is needed
        return response_text

    def get_answer(self, query: str, user_info: dict[str, str | List[str]]):
        """Process a query and return the answer

        Args:
            query (str): The user's question
            user_info (dict, optional): Additional user information like fitness level, goals, etc.

        Returns:
            dict: A dictionary containing both raw and formatted answers
        """
        # Add user info to query if provided
        if user_info:
            user_profile = (
                f"Fitness level: {user_info.get('fitness_level', 'Intermediate')}. "
            )
            user_profile += (
                f"Goals: {', '.join(user_info.get('goals', ['General fitness']))}. "
            )
        else:
            user_profile = "No specific fitness information provided."

        # Process the query through the RAG chain
        result = self.rag_chain.invoke(
            {
                "input": query,  # Just the query without fitness info
                "user_profile": user_profile,  # Separate placeholder for user info
                "chat_history": self.chat_history,
            }
        )

        # Format the response
        formatted_answer = self.format_response(result["answer"])

        # Update chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(SystemMessage(content=result["answer"]))

        return {"raw_answer": result["answer"], "formatted_answer": formatted_answer}

    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = []


# Create a default instance for easy importing
default_agent = GymTrainerAgent()


# For backward compatibility and simple usage
def get_answer(query, user_info):
    return default_agent.get_answer(query, user_info)


def reset_chat_history():
    default_agent.reset_chat_history()
