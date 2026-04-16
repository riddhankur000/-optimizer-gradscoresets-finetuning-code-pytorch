# Riemannian Training Data Format Analysis

## Quick Reference: Data Format at Each Stage

### Stage 1: Task Loading (colm/data/tasks.py)
**Output**: Sample objects

```python
@dataclass
class Sample:
    id: int = None
    data: dict = None  # Original HF dataset example
    correct_candidate: Union[str, List[str]] = None  # The label/answer
    candidates: List[str] = None  # Possible options
```

**Example for SST2**:
```python
Sample(
    id=0,
    data={
        'sentence': 'This movie is great',
        'idx': 0,
        'label': 1
    },
    correct_candidate=1,  # Binary: 0 or 1
    candidates=[0, 1]
)
```

### Stage 2: Convert to HuggingFace Dataset (convert_task_samples_to_hf_dataset)
**Input**: TaskDataset with Sample objects
**Output**: HFDataset with fields:

```python
{
    "id": ["0", "1", "2", ...],                    # String IDs
    "text": ["This movie is great", ...],          # Concatenated text fields
    "label": [1, 0, 1, ...],                       # Scalar int32 labels
}
```

**Field Extraction Logic**:
- `id`: Extracted from sample.id or sample_data['id']
- `text`: Concatenated from available fields (text, sentence, sentence1, question, context)
- `label`: From correct_candidate if exists, else sample_data['label'], then cast to int

### Stage 3: Tokenization (train_sequential_from_config.py line 540-550)
**Input**: HFDataset with text and label fields
**Output**: HFDataset with tokenized format

```python
def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding=False,  # Data collator handles padding
    )
    # KEY: Convert scalar labels to sequences for seq2seq collator
    result['labels'] = [[int(label)] for label in examples['label']]
    return result
```

**Output Fields**:
```python
{
    "input_ids": [
        [101, 2054, 2003, 2062, ...],  # Tokenized input
        ...
    ],
    "attention_mask": [
        [1, 1, 1, 1, ...],              # Attention mask (1 for real tokens)
        ...
    ],
    "labels": [
        [1],                            # Label as 1-element sequence
        [0],
        [1],
        ...
    ]
}
```

### Stage 4: Data Collator (DataCollatorForSeq2Seq)
**Input**: List of sample dicts
**Output**: Batched tensors

```python
{
    "input_ids": torch.Tensor([[101, ..., 0, 0], ...]),      # Padded to max length
    "attention_mask": torch.Tensor([[1, 1, ..., 0, 0], ...]), # Padded to match
    "labels": torch.Tensor([[1], [0], [1], ...])             # Labels (padded with -100)
}
```

**Collator Config**: `label_pad_token_id=-100` masks loss for padding positions

### Stage 5: SubsetTrainerEfficient
**Expects in batch**: `input_ids`, `attention_mask`, `labels`

Batches flow through:
```
Batch → Model → Logits
Batch['input_ids'] → Embedding → Hidden States → LM Head → Logits
Batch['labels'] → Loss Computation (ignoring -100 positions)
```

---

## Answer to Your Questions

### 1. How datasets are loaded and prepared before training

**Path**: `colm/data/sequential_task_loader.py` + `colm/data/tasks.py`

```python
# Step 1: Load task
task = get_task(task_name)  # Returns Dataset with samples attribute
train_samples = task.samples["train"]  # List of Sample objects
val_samples = task.samples["valid"]

# Step 2: Create TaskDataset wrapper
train_dataset = TaskDataset(train_samples, task_name, split="train")

# Step 3: Convert to HF Dataset
train_dataset_hf = convert_task_samples_to_hf_dataset(train_dataset)
# Output: HFDataset with "id", "text", "label" fields

# Step 4: Apply tokenization
train_dataset_tokenized = train_dataset_hf.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['text', 'id']
)
# Output: HFDataset with "input_ids", "attention_mask", "labels"

# Step 5: Pass to trainer
trainer = SubsetTrainerEfficient(
    train_dataset=train_dataset_tokenized,
    ...
)
```

### 2. Dataset format passed to SubsetTrainerEfficient

**Exact Format**:
```python
Training Dataset:
- Type: HuggingFace Dataset with Examples
- Each example contains:
  {
      "input_ids": [101, 2054, 2003, ...],      # int list, tokenized input
      "attention_mask": [1, 1, 1, ...],         # int list, attention mask
      "labels": [1],                            # int list, 1-element sequence
  }

- No padding at dataset level (data collator handles padding during batching)
- Length varies per sample (will be padded during collation)
```

### 3. How labels/candidates are handled for classification tasks

**For SST2 (Binary Classification)**:
```python
class SST2Dataset(Dataset):
    def build_sample(self, example):
        label = int(example["label"])  # 0 or 1
        return Sample(
            id=example["idx"],
            data=example,
            correct_candidate=label,    # Scalar int
            candidates=[0, 1]           # Binary options
        )

# In convert_task_samples_to_hf_dataset:
label = sample.correct_candidate  # Extract scalar label
data_dict["label"].append(int(label))  # Append as int

# In tokenize_function:
result['labels'] = [[int(label)] for label in examples['label']]  # Wrap as sequence
```

**For RTE (Binary)**:
```python
class RTEDataset(Dataset):
    def build_sample(self, example):
        return Sample(
            data=example,
            correct_candidate=example['label'],  # 0 or 1
            candidates=[0, 1]
        )
# Same flow: scalar → extracted → wrapped as [label]
```

**For BoolQ (Binary with string labels)**:
```python
class BoolQDataset(Dataset):
    def build_sample(self, example):
        return Sample(
            data=example,
            correct_candidate="Yes" if example["answer"] else "No",  # String
            candidates=["Yes", "No"]
        )

# In convert_task_samples_to_hf_dataset:
# String labels are converted to int (likely treated as indices)
```

**Pattern**: All classification tasks follow the same flow:
1. Extract scalar label or string candidate
2. Convert to int in HF Dataset
3. Wrap in list during tokenization: `[[label]]`
4. DataCollator pads with -100 for loss masking

### 4. Whether input_ids are pre-tokenized or tokenized on-the-fly

**Answer**: **Tokenized ON-THE-FLY**

**Current Approach (train_sequential_from_config.py)**:
1. `get_task()` returns raw data with original text fields
2. `convert_task_samples_to_hf_dataset()` extracts and concatenates text → "text" field
3. `.map(tokenize_function)` applies tokenization during dataset preparation
4. Trainer receives pre-tokenized data

**Code Evidence**:
```python
# Line 540-550: Tokenization happens BEFORE trainer receives data
train_subset = train_subset.map(tokenize_function, batched=True)

# Then passed to trainer
trainer = SubsetTrainerEfficient(
    train_dataset=train_subset,  # Already tokenized
    ...
)
```

**NOT lazily tokenized in data collator** - tokenization is eager (happens at `.map()` call)

### 5. Field names expected by the trainer

**SubsetTrainerEfficient Expects**:

```
Required fields in each example:
├── input_ids: List[int]          # Tokenized input (variable length)
├── attention_mask: List[int]     # Binary mask (1 for real, 0 for padding)
└── labels: List[int]             # Label sequence (1 element for classification)

Optional fields (removed by tokenize_function):
├── text (removed after tokenization)
└── id (removed after tokenization)
```

**At Batch Level (from data collator)**:
```python
batch = {
    "input_ids": torch.Tensor([...]),        # Shape: (batch_size, seq_length)
    "attention_mask": torch.Tensor([...]),   # Shape: (batch_size, seq_length)
    "labels": torch.Tensor([...]),           # Shape: (batch_size, 1)
}
```

**Critical Details**:
- `input_ids` and `attention_mask` are padded to same length within batch
- `labels` is kept as-is (single element per sample after padding with -100)
- `label_pad_token_id=-100` is used to ignore padding in loss computation
- No other special columns needed (e.g., token_type_ids is optional)

---

## Data Flow Summary: Raw → SubsetTrainerEfficient

```
Classification Task (SST2, RTE, BoolQ)
    ↓
Sample objects with:
- id, data (dict), correct_candidate (scalar), candidates (list)
    ↓
HF Dataset conversion:
- id → string, text → concatenated text, label → int
    ↓
Tokenization:
- text → tokenized to input_ids + attention_mask
- label → wrapped as [label] sequence
    ↓
Dataset columns:
- input_ids: List[int], attention_mask: List[int], labels: List[int]
    ↓
DataCollatorForSeq2Seq:
- Pads input_ids and attention_mask to max length in batch
- Pads labels with -100 to match sequence length
    ↓
SubsetTrainerEfficient batch:
- input_ids: Tensor, attention_mask: Tensor, labels: Tensor
- Ready for forward pass and loss computation
```

---

## Key Preprocessing Steps Before SubsetTrainerEfficient

1. **Task Load** (get_task) → Sample objects
2. **Convert** (convert_task_samples_to_hf_dataset) → HFDataset with text + label
3. **Tokenize** (tokenize_function) → input_ids + attention_mask + labels as sequences
4. **Collate** (DataCollatorForSeq2Seq) → Batched padded tensors
5. **Train** (SubsetTrainerEfficient) → Forward pass and loss computation
