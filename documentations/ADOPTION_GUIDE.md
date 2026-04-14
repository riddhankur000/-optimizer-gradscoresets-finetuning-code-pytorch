# How to Adopt RiemanianFinetune Strategy for Your Math Training

## 🔧 **IMPLEMENTATION GUIDE**

### **Step 1: Create Multi-Source Dataset Loader**

Create file: `colm/data/load_math_datasets.py`

```python
import datasets
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
import functools

def _get_MetaMath_instructions(example, tokenizer):
    """MetaMathQA dataset processor"""
    instructions = [
        {
            "role": "system",
            "content": "You are a math expert. Solve the problem step by step."
        },
        {"role": "user", "content": example['query']},
    ]
    
    start_idx = example['response'].find("The answer is:")
    if start_idx == -1:
        correct_answer = example['response']
    else:
        start_idx += len("The answer is: ")
        correct_answer = example['response'][start_idx:]
    
    instructions_ans = [
        {"role": "assistant", "content": example['response']}
    ]
    
    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )
    
    return {
        'text': text,
        'correct_answer': correct_answer,
        'task': 'MetaMathQA',
        'original_index': example.get('original_index', -1),
        'source': 'MetaMathQA'
    }


def _get_GSM8K_instructions(example, tokenizer):
    """GSM8K dataset processor"""
    instructions = [
        {
            "role": "system",
            "content": "You are a math expert. Solve the problem step by step."
        },
        {"role": "user", "content": example['question']},
    ]
    
    start_idx = example['answer'].find("#### ")
    if start_idx != -1:
        correct_answer = example['answer'][start_idx + len("#### "):]
        response = example['answer'][:start_idx] + f"The answer is: {correct_answer}"
    else:
        response = example['answer']
        correct_answer = example['answer']
    
    instructions_ans = [
        {"role": "assistant", "content": response}
    ]
    
    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )
    
    return {
        'text': text,
        'correct_answer': correct_answer,
        'task': 'GSM8K',
        'original_index': -1,
        'source': 'GSM8K'
    }


def _get_MATH_instructions(example, tokenizer):
    """MATH dataset processor (if available separately)"""
    instructions = [
        {
            "role": "system",
            "content": "You are a math expert. Solve the problem step by step."
        },
        {"role": "user", "content": example['problem']},
    ]
    
    instructions_ans = [
        {"role": "assistant", "content": example['solution']}
    ]
    
    text = tokenizer.apply_chat_template(
        instructions + instructions_ans,
        tokenize=False
    )
    
    return {
        'text': text,
        'correct_answer': example['answer'],
        'task': 'MATH',
        'original_index': -1,
        'source': 'MATH'
    }


def load_multiple_math_datasets(dataset_names=['MetaMathQA', 'GSM8K']):
    """
    Load multiple math datasets from HF Hub
    
    Args:
        dataset_names: List of dataset names to load
    
    Returns:
        List of loaded datasets
    """
    datasets_dict = {
        'MetaMathQA': lambda: load_dataset('meta-math/MetaMathQA'),
        'GSM8K': lambda: load_dataset('openai/gsm8k', 'main'),
        'MATH': lambda: load_dataset('competition_math'),  # if available
    }
    
    loaded_datasets = []
    for name in dataset_names:
        if name in datasets_dict:
            print(f"Loading {name}...")
            loaded_datasets.append(datasets_dict[name]())
        else:
            print(f"Warning: {name} not available")
    
    return loaded_datasets


def process_math_datasets(dataset_list, tokenizer, num_proc=8):
    """
    Apply formatters to each math dataset
    
    Args:
        dataset_list: List of datasets to process
        tokenizer: Tokenizer for chat template
        num_proc: Number of processes for parallel processing
    
    Returns:
        List of processed datasets with task/source metadata
    """
    processors = {
        'MetaMathQA': _get_MetaMath_instructions,
        'GSM8K': _get_GSM8K_instructions,
        'MATH': _get_MATH_instructions,
    }
    
    processed_datasets = []
    
    for i, dataset in enumerate(dataset_list):
        dataset_name = list(dataset.keys())[0]  # 'train', 'validation' or 'test'
        print(f"Processing dataset {i}: {dataset_name}")
        
        processor = processors.get(dataset_name, _get_MetaMath_instructions)
        processor_fn = functools.partial(processor, tokenizer=tokenizer)
        
        # Apply formatter to all splits
        processed = dataset.map(
            processor_fn,
            batched=False,
            num_proc=num_proc,
            remove_columns=dataset[dataset_name].column_names
        )
        
        processed_datasets.append(processed)
    
    return processed_datasets


def combine_math_datasets(processed_datasets, train_ratio=0.9):
    """
    Combine multiple datasets into train/val/test splits
    
    Args:
        processed_datasets: List of processed dataset dicts
        train_ratio: Train/val split ratio
    
    Returns:
        Combined DatasetDict
    """
    all_train = []
    all_val = []
    all_test = []
    
    for dataset in processed_datasets:
        # Split train into train/val
        if 'train' in dataset:
            train_split = dataset['train'].train_test_split(
                test_size=1-train_ratio,
                shuffle=True,
                seed=42
            )
            all_train.append(train_split['train'])
            all_val.append(train_split['test'])
        
        # Use validation as test
        if 'validation' in dataset:
            all_test.append(dataset['validation'])
    
    # Concatenate all
    combined = DatasetDict({
        'train': datasets.concatenate_datasets(all_train) if all_train else None,
        'validation': datasets.concatenate_datasets(all_val) if all_val else None,
        'test': datasets.concatenate_datasets(all_test) if all_test else None,
    })
    
    print(f"Combined dataset sizes:")
    print(f"  Train: {len(combined['train'])} examples")
    print(f"  Validation: {len(combined['validation'])} examples")
    print(f"  Test: {len(combined['test'])} examples")
    
    return combined


def main():
    # Example usage
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    
    # Load datasets
    dataset_list = load_multiple_math_datasets(['MetaMathQA', 'GSM8K'])
    
    # Process with formatters
    processed = process_math_datasets(dataset_list, tokenizer)
    
    # Combine
    combined = combine_math_datasets(processed)
    
    # Save
    combined.save_to_disk('./colm_math_combined_dataset')
    
    print("Dataset saved!")


if __name__ == '__main__':
    main()
```

---

### **Step 2: Update Training Script to Load Combined Dataset**

Modify: `colm/train/train.py`

```python
# Add after imports
from colm.data.load_math_datasets import (
    load_multiple_math_datasets,
    process_math_datasets,
    combine_math_datasets
)

# In main() function, replace dataset loading section:

# OLD CODE (COMMENT OUT):
# train_dataset = get_training_dataset(
#     data_args.train_files,
#     tokenizer=tokenizer,
#     ...
# )

# NEW CODE:
if training_args.use_combined_math:
    print("Loading multiple math datasets...")
    
    # Load datasets
    dataset_list = load_multiple_math_datasets([
        'MetaMathQA',
        'GSM8K'
        # Add more as needed
    ])
    
    # Process with formatters
    processed_datasets = process_math_datasets(
        dataset_list,
        tokenizer,
        num_proc=training_args.num_proc
    )
    
    # Combine
    combined_dataset_dict = combine_math_datasets(
        processed_datasets,
        train_ratio=0.85  # 85% train, 15% val
    )
    
    train_dataset = combined_dataset_dict['train']
    analysis_dataset = combined_dataset_dict['validation']
    
    logger.info(f"Loaded combined math datasets")
    logger.info(f"Train split: {len(train_dataset)} examples")
    logger.info(f"Val split: {len(analysis_dataset)} examples")
else:
    # Fall back to original method
    train_dataset = get_training_dataset(...)
```

---

### **Step 3: Add Per-Task Evaluation Metrics**

Create file: `colm/train/per_task_metrics.py`

```python
import numpy as np
from collections import defaultdict
import torch

class PerTaskMetricsTracker:
    """
    Track metrics per task/source for detailed analysis
    """
    
    def __init__(self):
        self.task_losses = defaultdict(list)
        self.task_perplexities = defaultdict(list)
        self.task_accuracies = defaultdict(list)
        self.step_count = 0
    
    def record_loss(self, loss, task_name):
        """Record loss for a specific task"""
        self.task_losses[task_name].append(loss)
    
    def record_accuracy(self, accuracy, task_name):
        """Record accuracy for a specific task"""
        self.task_accuracies[task_name].append(accuracy)
    
    def get_task_loss(self, task_name):
        """Get average loss for a task"""
        if task_name in self.task_losses:
            return np.mean(self.task_losses[task_name])
        return None
    
    def get_all_metrics(self):
        """Get all task metrics"""
        metrics = {}
        for task_name in self.task_losses:
            metrics[f'{task_name}/loss'] = np.mean(self.task_losses[task_name])
            metrics[f'{task_name}/loss_std'] = np.std(self.task_losses[task_name])
        
        return metrics
    
    def reset(self):
        """Reset for new epoch"""
        self.task_losses = defaultdict(list)
        self.task_accuracies = defaultdict(list)
        self.step_count = 0
```

Usage in custom trainer:

```python
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_metrics = PerTaskMetricsTracker()
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # Call original
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        # Add per-task metrics to wandb
        task_metrics = self.task_metrics.get_all_metrics()
        if task_metrics:
            self.log(task_metrics)
```

---

### **Step 4: Modify Data Collator to Track Task**

Update: `colm/data/get_training_dataset.py`

```python
class DataCollatorForMultiTaskWithSource(object):
    """
    Collate examples while tracking task/source
    """
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract task information
        task_names = []
        sources = []
        
        for instance in instances:
            task_names.append(instance.get('task', 'unknown'))
            sources.append(instance.get('source', 'unknown'))
        
        # Get text and labels
        sources_text = [inst['text'] for inst in instances]
        
        # Tokenize
        data_dict = preprocess(sources_text, ['']*len(instances), self.tokenizer)
        input_ids = data_dict['input_ids']
        
        # Pad
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            task_names=task_names,  # ← NEW
            sources=sources,       # ← NEW
        )
```

---

### **Step 5: Update Training Arguments**

Modify: `colm/train/training_arguments.py`

```python
@dataclass
class TrainingArguments(TA):
    # ... existing fields ...
    
    use_combined_math: bool = field(
        default=False,
        metadata={"help": "Use combined multiple math datasets"}
    )
    
    num_proc: int = field(
        default=8,
        metadata={"help": "Number of processes for data loading"}
    )
    
    eval_per_task: bool = field(
        default=True,
        metadata={"help": "Evaluate per task separately"}
    )
    
    track_task_loss: bool = field(
        default=True,
        metadata={"help": "Track loss per task"}
    )
```

---

### **Step 6: Create Per-Task Evaluation Script**

Create file: `colm/train/eval_per_task.py`

```python
def evaluate_per_task(model, eval_dataset, tokenizer, task_names=None):
    """
    Evaluate model performance per task
    
    Args:
        model: Trained model
        eval_dataset: Evaluation dataset with 'task' column
        tokenizer: Tokenizer
        task_names: List of task names to evaluate
    
    Returns:
        Dict of per-task metrics
    """
    
    metrics_per_task = {}
    
    if task_names is None:
        task_names = set(eval_dataset['task'])
    
    for task_name in task_names:
        print(f"\nEvaluating task: {task_name}")
        
        # Filter to this task
        task_dataset = eval_dataset.filter(
            lambda x: x['task'] == task_name
        )
        
        if len(task_dataset) == 0:
            print(f"  Skipping: No examples for {task_name}")
            continue
        
        # Evaluate
        task_metrics = evaluate_single_task(
            model,
            task_dataset,
            tokenizer
        )
        
        metrics_per_task[task_name] = task_metrics
        
        print(f"  {task_name} Loss: {task_metrics['loss']:.4f}")
        print(f"  {task_name} Accuracy: {task_metrics.get('accuracy', 'N/A')}")
    
    return metrics_per_task


def evaluate_single_task(model, dataset, tokenizer):
    """Evaluate single task"""
    model.eval()
    
    total_loss = 0
    total_examples = 0
    
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
            total_examples += 1
    
    return {
        'loss': total_loss / total_examples if total_examples > 0 else 0,
    }
```

---

### **Step 7: Create Training Script with Multiple Datasets**

Create file: `scripts/train_multi_math_datasets.sh`

```bash
#!/bin/bash

export MODEL_NAME="microsoft/phi-2"
export DATA_DIR="./data"
export OUTPUT_DIR="./out/multi_math_lora"

python -m colm.train.train \
    --model_name_or_path $MODEL_NAME \
    --use_combined_math True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --lora True \
    --lora_r 128 \
    --lora_alpha 512 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10 \
    --eval_per_task True \
    --track_task_loss True \
    --fp16 True \
    --report_to wandb \
    --run_name multi_math_lora \
    --seed 0
```

---

## 📊 **EXPECTED OUTPUT WITH NEW STRUCTURE**

### Training Logs:
```
Loading multiple math datasets...
Loading MetaMathQA...
Loading GSM8K...
Processing dataset 0: MetaMathQA
Processing dataset 1: GSM8K
Combining datasets...

Combined dataset sizes:
  Train: 625,000 examples
  Validation: 93,750 examples
  Test: 15,000 examples

Device: cuda:0, Device Memory: 47.5 GB
[27%|██▋       | 163/600 [00:45<02:15, 3.25it/s]
04/13/2026 23:45:23 - INFO - BoolQ/loss: 0.4523
04/13/2026 23:45:23 - INFO - MetaMathQA/loss: 0.3821
04/13/2026 23:45:23 - INFO - GSM8K/loss: 0.4156
...
```

### Evaluation Output:
```
Per-Task Metrics at Step 100:

MetaMathQA:
  Loss: 0.3821
  Accuracy: 0.72

GSM8K:
  Loss: 0.4156
  Accuracy: 0.68

Average:
  Loss: 0.3989
  Accuracy: 0.70
```

### W&B Dashboard:
```
- training/loss (global)
- training/MetaMathQA_loss
- training/GSM8K_loss
- eval/MetaMathQA_loss
- eval/GSM8K_loss
```

---

## ✅ **BENEFITS OF THIS APPROACH**

✓ **Per-task analysis** - Understand which math domains improve
✓ **Better evaluation** - More comprehensive assessment
✓ **Flexible scaling** - Add new datasets easily
✓ **Detailed logging** - Track model behavior per source
✓ **Task-specific improvements** - Iterate on weak areas
✓ **Reproducibility** - Standardized formatting per dataset
