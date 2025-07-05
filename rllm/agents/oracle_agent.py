from rllm.agents.agent import BaseAgent


class OracleAgent(BaseAgent):
    """
    Oracle agent that handles structured messages (system + user) properly
    for single-turn oracle environments with custom reward functions.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    def update_from_env(self, observation, reward, done, info):
        """
        Process environment observation into chat messages.
        
        Handles both structured messages and fallback question format.
        
        Args:
            observation: Environment observation containing messages or question
            reward: Reward from environment 
            done: Whether episode is done
            info: Additional environment info
        """
        
        if "messages" in observation:
            # Use the original system/user messages directly
            messages = observation["messages"]
            
            # Reset messages and add the structured messages
            self.messages = []
            for msg in messages:
                self.messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        else:
            # Fallback to question-based approach for compatibility
            question = observation.get("question", str(observation))
            self.messages.append({"role": "user", "content": question})
        
        self.update_trajectory_step(observation, reward, done, info)

    def reset(self):
        """Reset the agent state for a new episode."""
        super().reset()
        self.messages = []