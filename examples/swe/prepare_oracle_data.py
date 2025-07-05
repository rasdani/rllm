from datasets import Dataset
from rllm.data.dataset import DatasetRegistry


def prepare_oracle_data():
    """
    Prepare and register oracle SWE datasets for training.
    
    Converts dataset with messages (system, user) and patch to SingleTurnEnvironment format.
    Maintains proper message structure instead of concatenating.
    """
    
    # Load your oracle dataset
    oracle_dataset = load_oracle_dataset()
    
    train_data = []
    
    for item in oracle_dataset:
        # Keep original messages structure - don't concatenate!
        formatted_item = {
            "messages": item["messages"],  # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            "patch": item["patch"],  # Ground truth for reward function
            "data_source": "oracle_swe",  # Custom dataset identifier
        }
        
        train_data.append(formatted_item)
    
    # Register dataset
    DatasetRegistry.register_dataset("Oracle_SWE", train_data, "train")
    print(f"Registered Oracle_SWE dataset with {len(train_data)} examples")
    
    return train_data


def load_oracle_dataset():
    """
    PLACEHOLDER: Load your oracle dataset.
    
    Returns:
        List of dictionaries with structure:
        {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"}
            ],
            "patch": "ground truth patch string"
        }
    """
    # TODO: Implement your dataset loading logic
    # Example placeholder data:
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks."},
                {"role": "user", "content": "Consider the following github issue:\n\n<github_issue>\nFix the bug in this function where addition is implemented as subtraction:\n\ndef add(a, b):\n    return a - b\n</github_issue>\n\nCan you help me implement the necessary changes to fix this issue?"}
            ],
            "patch": "return a + b"
        },
        {
            "messages": [
                {"role": "system", "content": "You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks."},
                {"role": "user", "content": "Consider the following github issue:\n\n<github_issue>\nThe multiply function always returns 0:\n\ndef multiply(x, y):\n    return x * 0\n</github_issue>\n\nCan you help me implement the necessary changes to fix this issue?"}
            ],
            "patch": "return x * y"
        }
    ]


if __name__ == "__main__":
    prepare_oracle_data()