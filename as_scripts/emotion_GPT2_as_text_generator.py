# Import required libraries
import json  # For parsing JSON data
import random  # For setting seeds and shuffling data
import gzip  # For decompressing dataset
from typing import Any, Tuple
from dataclasses import dataclass

import requests  # For downloading dataset from URL
import torch  # Main PyTorch library
from torch.utils.data import Dataset, DataLoader  # For dataset handling
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast  # Hugging Face model components
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW  # Optimizer for training
from tqdm import tqdm  # Progress bar utilities
import re  # For text normalization


@dataclass
class DataEntry:
    prompt: str
    completion: str


@dataclass
class TrainingExample:
    input_ids: list[int]
    labels: list[int]
    prompt: str
    expected_completion: str


def set_seed(seed):
    """
    Sets random seeds for reproducibility across different libraries.

    Args:
        seed (int): Seed value for random number generation
    """
    # Set Python's built-in random seed
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set seed for all available GPUs
    # torch.mps.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    # Request cuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN's auto-tuner for consistent behavior
    torch.backends.cudnn.benchmark = False


def build_prompt(text):
    """
    Creates a standardized prompt for emotion classification.

    Args:
        text (str): Input text to classify

    Returns:
        str: Formatted prompt for the model
    """
    # Format the input text into a consistent prompt structure
    # Include explicit task instruction and expected output format
    return f"Predict the emotion for the following text: {text}\nEmotion:"


def encode_text(tokenizer: GPT2TokenizerFast, text: str, return_tensor: object = False) -> list[int]:
    """
    Encodes text using the provided tokenizer.

    Args:
        tokenizer: Hugging Face tokenizer
        text (str): Text to encode
        return_tensor (bool): Whether to return PyTorch tensor

    Returns:
        List or tensor of token IDs
    """
    # If tensor output is requested, encode with PyTorch tensors
    if return_tensor:
        return tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )
    # Otherwise return list of token IDs
    else:
        return tokenizer.encode(text, add_special_tokens=False)


def decode_text(tokenizer: GPT2TokenizerFast, token_ids: list[int]):
    """
    Decodes token IDs back to text.

    Args:
        tokenizer: Hugging Face tokenizer
        token_ids: List or tensor of token IDs

    Returns:
        str: Decoded text
    """
    # Convert token IDs back to text, skipping special tokens
    return tokenizer.decode(token_ids, skip_special_tokens=True)


class PromptCompletionDataset(Dataset[TrainingExample]):
    """
    PyTorch Dataset for prompt-completion pairs.
    Handles the conversion of text data into model-ready format.

    Args:
        data (list): List of dictionaries containing prompts and completions
        tokenizer: Hugging Face tokenizer
    """

    def __init__(self, data: list[DataEntry], tokenizer: GPT2TokenizerFast):
        # Store the raw data and tokenizer for later use
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the total number of examples in the dataset
        return len(self.data)

    def __getitem__(self, idx: int) -> TrainingExample:
        """
            Returns a single training example.

            Args:
                idx (int): Index of the example to fetch

            Returns:
                dict: Contains input_ids, labels, prompt, and expected completion
            """
        # Get the specific example from our dataset
        item = self.data[idx]
        prompt: str = item.prompt
        completion: str = item.completion

        # Convert text to token IDs for both prompt and completion
        encoded_prompt = encode_text(self.tokenizer, prompt)
        encoded_completion = encode_text(self.tokenizer, completion)
        # Get the end-of-sequence token ID
        eos_token = self.tokenizer.eos_token_id

        # Combine prompt and completion tokens with EOS token
        input_ids = encoded_prompt + encoded_completion + [eos_token]
        # Create labels: -100 for prompt (ignored in loss), completion tokens for learning
        labels = [-100] * len(encoded_prompt) + encoded_completion + [eos_token]

        return TrainingExample(
            input_ids=input_ids,
            labels=labels,
            prompt=prompt,
            expected_completion=completion
        )


def create_collate_fn(tokenizer: GPT2TokenizerFast):
    """
    Creates a collate function with access to the tokenizer.
    
    Args:
        tokenizer: The tokenizer to use for padding
    
    Returns:
        A collate function that can be used with DataLoader
    """
    def collate_fn(batch: list[TrainingExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
        """
        Collates batch of examples into training-ready format.
        Handles padding and conversion to tensors.

        Args:
            batch: List of examples from Dataset

        Returns:
            tuple: (input_ids, attention_mask, labels, prompts, expected_completions)
        """
        # Find the longest sequence in the batch for padding
        max_length = max(len(item.input_ids) for item in batch)

        # Pad input sequences to max_length with pad token
        input_ids = [
            item.input_ids +
            [tokenizer.pad_token_id] * (max_length - len(item.input_ids))
            for item in batch
        ]

        # Pad label sequences with -100 (ignored in loss calculation)
        labels = [
            item.labels +
            [-100] * (max_length - len(item.labels))
            for item in batch
        ]

        # Create attention masks: 1 for real tokens, 0 for padding
        attention_mask = [
            [1] * len(item.input_ids) +
            [0] * (max_length - len(item.input_ids))
            for item in batch
        ]

        # Keep original prompts and completions for evaluation
        prompts = [item.prompt for item in batch]
        expected_completions = [item.expected_completion for item in batch]

        # Convert everything to PyTorch tensors except text
        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
            prompts,
            expected_completions
        )
    
    return collate_fn


def normalize_text(text):
    """
    Normalizes text for consistent comparison.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    # Remove leading/trailing whitespace and convert to lowercase
    text = text.strip().lower()
    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    return text


def calculate_accuracy(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, loader: DataLoader[TrainingExample]):
    """
    Calculates prediction accuracy on a dataset.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        loader: DataLoader containing evaluation examples

    Returns:
        float: Accuracy score
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    # Initialize counters for accuracy calculation
    correct = 0
    total = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate through batches
        for input_ids, attention_mask, labels, prompts, expected_completions in loader:
            # Process each example in the batch
            for prompt, expected_completion in zip(prompts, expected_completions):
                # Generate model's prediction for this prompt
                generated_text = generate_text(model, tokenizer, prompt)
                # Compare normalized versions of prediction and expected completion
                if normalize_text(generated_text) == normalize_text(expected_completion):
                    correct += 1
                total += 1

    # Calculate accuracy, handling empty dataset case
    accuracy = correct / total if total > 0 else 0
    # Reset model to training mode
    model.train()
    return accuracy


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, prompt: str, max_new_tokens: int = 50) -> str:
    """
    Generates text completion for a given prompt.

    Args:
        model: Fine-tuned model
        tokenizer: Associated tokenizer
        prompt (str): Input prompt
        max_new_tokens (int): Maximum number of tokens to generate

    Returns:
        str: Generated completion
    """
    # Encode prompt and move to model's device
    encoding: BatchEncoding = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate completion using model
    output_ids = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # Use KV cache for faster generation
        num_beams=1,  # Use greedy decoding
        do_sample=False,  # Don't use sampling
    )[0]

    # Extract and decode only the generated part (excluding prompt)
    generated_text = decode_text(tokenizer, output_ids[encoding["input_ids"].shape[1]:])
    return generated_text.strip()


def test_model(model_path: str, test_input: str):
    """
    Tests a saved model on a single input.

    Args:
        model_path (str): Path to saved model
        test_input (str): Text to classify
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load saved model and move to appropriate device
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_path)

    # Ensure model has proper padding token configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create prompt and generate prediction
    prompt = build_prompt(test_input)
    generated_text = generate_text(model, tokenizer, prompt)

    # Display results
    print(f"Input: {test_input}")
    print(f"Generated emotion: {generated_text}")


def download_and_prepare_data(data_url: str, tokenizer: GPT2TokenizerFast, batch_size: int, test_ratio: float = 0.1) \
        -> tuple[DataLoader[TrainingExample], DataLoader[TrainingExample]]:
    """
    Downloads and prepares dataset for training.

    Args:
        data_url (str): URL of the dataset
        tokenizer: Tokenizer for text processing
        batch_size (int): Batch size for DataLoader
        test_ratio (float): Proportion of data for testing

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Download compressed dataset
    response = requests.get(data_url)
    # Decompress and decode the content
    content = gzip.decompress(response.content).decode()

    # Parse each line as JSON and format into prompt-completion pairs
    dataset: list[DataEntry] = []
    for entry in map(json.loads, content.splitlines()):
        dataset.append(
            DataEntry(
                prompt=build_prompt(entry['text']),
                completion=entry["label"].strip()
            )
        )

    # Randomly shuffle dataset for better split
    random.shuffle(dataset)
    # Calculate split index based on test ratio
    split_index = int(len(dataset) * (1 - test_ratio))
    # Split into train and test sets
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    # Create dataset objects
    train_dataset = PromptCompletionDataset(train_data, tokenizer)
    test_dataset = PromptCompletionDataset(test_data, tokenizer)

    # Create collate function with tokenizer access
    collate_fn = create_collate_fn(tokenizer)
    
    # Create data loaders with appropriate settings
    train_loader = DataLoader[TrainingExample](
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn  # Custom collation for padding
    )
    test_loader = DataLoader[TrainingExample](
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        collate_fn=collate_fn
    )

    return train_loader, test_loader


def get_hyperparameters() -> tuple[int, int, float]:
    """
    Returns training hyperparameters.

    Returns:
        tuple: (num_epochs, batch_size, learning_rate)
    """
    # Training for 2 epochs as a balance of learning and efficiency
    num_epochs = 2
    # Batch size of 16 works well with most GPU memory sizes
    batch_size = 16
    # Standard learning rate for fine-tuning transformers
    learning_rate = 5e-5

    return num_epochs, batch_size, learning_rate


# Main training script
def main():
    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    data_url = "https://www.thelmbook.com/data/emotions"
    model_name = "openai-community/gpt2"
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and configure padding token

    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained model and move to appropriate device
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Get training hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Load and prepare training data
    train_loader, test_loader = download_and_prepare_data(data_url, tokenizer, batch_size)

    # Initialize optimizer with learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Initialize epoch metrics
        total_loss = 0
        num_batches = 0
        # Create progress bar for this epoch
        progress_bar: tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # Process each batch
        for input_ids, attention_mask, labels, _, _ in progress_bar:
            # Move batch data to appropriate device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass with loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update progress metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            progress_bar.set_postfix({"Loss": total_loss / num_batches})

        # Calculate and display epoch metrics
        avg_loss = total_loss / num_batches
        test_acc = calculate_accuracy(model, tokenizer, test_loader)
        print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Calculate final model performance
    train_acc = calculate_accuracy(model, tokenizer, train_loader)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the trained model and tokenizer
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    # Test model with a sample input
    test_input = "I'm so happy to be able to finetune an LLM!"
    test_model("./finetuned_model", test_input)


if __name__ == "__main__":
    main()
    # test_model("./finetuned_model", "What time is it?")
