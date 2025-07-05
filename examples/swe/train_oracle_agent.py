import hydra

from rllm.agents.oracle_agent import OracleAgent
from rllm.data import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.code_reward import swe_rl_reward_fn


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    Train oracle agent using SingleTurnEnvironment with custom SWE RL reward function.
    
    This training script:
    1. Loads oracle datasets with proper message structure
    2. Uses SingleTurnEnvironment for single-turn interactions
    3. Applies custom swe_rl_reward_fn to compare responses with ground truth patches
    4. Trains with PPO using the standard rLLM training pipeline
    """
    
    # Load Oracle datasets
    train_dataset = DatasetRegistry.load_dataset("Oracle_SWE", "train")
    val_dataset = DatasetRegistry.load_dataset("Oracle_SWE", "train")  # Use same for validation
    
    # Create environment factory with custom reward function
    def create_env_with_reward(**kwargs):
        return SingleTurnEnvironment(reward_fn=swe_rl_reward_fn, **kwargs)
    
    trainer = AgentTrainer(
        agent_class=OracleAgent,
        env_class=create_env_with_reward,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()