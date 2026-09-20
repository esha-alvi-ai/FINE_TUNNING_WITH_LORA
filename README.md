# LoRA Fine-Tuning

## Overview

**LoRA (Low-Rank Adaptation)** is a Parameter-Efficient Fine-Tuning
(PEFT) technique for adapting a pretrained language model without
updating all of its original parameters.

Instead of changing the complete pretrained weight matrix `W`, LoRA
freezes the original weights and learns a small low-rank update:

[ W' = W + `\frac{\alpha}{r}`{=tex}BA \]

Where:

-   `W` = original pretrained weight matrix
-   `A` = trainable low-rank matrix
-   `B` = trainable low-rank matrix
-   `r` = LoRA rank
-   `α` = LoRA scaling factor

------------------------------------------------------------------------

## Why LoRA?

Full fine-tuning updates every model parameter.

``` text
Full Fine-Tuning

Pretrained LLM
      |
      v
All weights trainable
      |
      v
Huge memory + optimizer requirements
      |
      v
Updated complete model
```

LoRA changes this to:

``` text
LoRA

Pretrained LLM
      |
      +--------------------+
      |                    |
      v                    v
Frozen Base Model       LoRA Adapter
                         A + B
                           |
                           v
                      Trainable
                           |
                           v
                    Adapted behavior
```

The base model stays frozen and only a small number of adapter
parameters are trained.

------------------------------------------------------------------------

## LoRA Architecture

``` mermaid
flowchart TD
    A[Input Tokens] --> B[Transformer Layer]
    B --> C[Original Weight W]
    B --> D[LoRA Branch]
    D --> E[Matrix A<br/>d_in x r]
    E --> F[Matrix B<br/>r x d_out]
    F --> G[Scaling alpha/r]
    C --> H[Base Output]
    G --> I[LoRA Update]
    H --> J[Add Base + LoRA]
    I --> J
    J --> K[Next Transformer Layer]
```

The LoRA branch learns:

\[ `\Delta `{=tex}W = `\frac{\alpha}{r}`{=tex}BA \]

and the effective weight becomes:

\[ W' = W + `\Delta `{=tex}W \]

------------------------------------------------------------------------

## Example of Parameter Reduction

Suppose a model has a weight matrix:

``` text
W = 4096 x 4096
```

Full matrix parameters:

``` text
4096 x 4096 = 16,777,216
```

With LoRA rank `r = 8`:

``` text
A = 8 x 4096
B = 4096 x 8
```

Trainable parameters:

``` text
(8 x 4096) + (4096 x 8)
= 65,536
```

So the LoRA update is much smaller than the original matrix.

------------------------------------------------------------------------

## Important LoRA Hyperparameters

### `r`

Rank controls the capacity of the adapter.

``` python
r=8
```

Small rank:

-   fewer trainable parameters
-   lower memory
-   smaller adapter
-   lower adaptation capacity

Larger rank:

-   more trainable parameters
-   more memory
-   potentially more task capacity

------------------------------------------------------------------------

### `lora_alpha`

Scaling factor:

``` python
lora_alpha=16
```

The common scaling is:

\[ `\frac{\alpha}{r}`{=tex} \]

------------------------------------------------------------------------

### `lora_dropout`

Example:

``` python
lora_dropout=0.05
```

Dropout can help reduce overfitting.

------------------------------------------------------------------------

### `target_modules`

These specify where LoRA adapters are inserted.

For Transformer attention, common targets include:

``` python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

Some models use different layer names, so target modules should be
checked for the specific architecture.

------------------------------------------------------------------------

## LoRA Training Flow

``` mermaid
flowchart LR
    A[Dataset] --> B[Tokenizer]
    B --> C[Pretrained LLM]
    C --> D[Freeze Base Weights]
    D --> E[Add LoRA Adapters]
    E --> F[Forward Pass]
    F --> G[Calculate Loss]
    G --> H[Backpropagation]
    H --> I[Update LoRA Only]
    I --> J[Save Adapter]
```

------------------------------------------------------------------------

## What Is Frozen?

``` text
Base Model:
    Frozen

LoRA A:
    Trainable

LoRA B:
    Trainable

Optimizer:
    Updates LoRA parameters
```

The optimizer does not need to update the full pretrained model.

------------------------------------------------------------------------

## LoRA Code

``` python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
```

------------------------------------------------------------------------

## LoRA With Hugging Face

Typical workflow:

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

------------------------------------------------------------------------

## Saving the Adapter

``` python
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
```

The adapter can be stored separately from the base model.

------------------------------------------------------------------------

## Loading the Adapter

``` python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name
)

model = PeftModel.from_pretrained(
    base_model,
    "./lora_adapter"
)
```

Architecture:

``` text
Base Model
    +
LoRA Adapter
    =
Fine-Tuned Model
```

------------------------------------------------------------------------

## Merging LoRA

If required, the adapter can be merged into the base model:

``` python
merged_model = model.merge_and_unload()
```

After merging, the LoRA update is incorporated into the model weights.

------------------------------------------------------------------------

## LoRA vs Full Fine-Tuning

  Feature                  Full Fine-Tuning   LoRA
  ------------------------ ------------------ ---------------
  Base weights             Trainable          Frozen
  Adapter                  No                 Yes
  Trainable parameters     Very large         Small
  GPU memory               High               Lower
  Storage                  Large model copy   Small adapter
  Multiple task adapters   Difficult          Easy
  Parameter-efficient      No                 Yes

------------------------------------------------------------------------

## LoRA vs RAG

These solve different problems.

### RAG

RAG retrieves external information at inference time.

``` text
User
 ↓
Retriever
 ↓
Documents
 ↓
LLM
 ↓
Answer
```

### LoRA

LoRA changes learned model behavior through fine-tuning.

``` text
Base LLM
 ↓
LoRA Training
 ↓
Adapted LLM
```

A system can use both:

``` text
                 Application
                     |
          +----------+----------+
          |                     |
         RAG                   LoRA
          |                     |
 Product/current data     Behavior/style/task
          |                     |
          +----------+----------+
                     |
                    LLM
```

------------------------------------------------------------------------

## LoRA vs QLoRA

``` text
LoRA:
Normal/standard-precision base model
+
LoRA adapters

QLoRA:
4-bit quantized base model
+
LoRA adapters
```

QLoRA reduces base-model memory further.

------------------------------------------------------------------------

## LoRA vs Prompt Tuning

**Prompt tuning** learns trainable prompt embeddings.

**LoRA** learns low-rank updates to selected model layers.

``` text
Prompt Tuning:
Input → Trainable Prompt → Model

LoRA:
Input → Model + Trainable Low-Rank Updates
```

------------------------------------------------------------------------
## Common Mistakes

1.  Using the wrong `target_modules`.
2.  Accidentally making the base model trainable.
3.  Using an unsuitable learning rate.
4.  Training on poor-quality data.
5.  Using too few examples and assuming the model has learned a general
    capability.
6.  Evaluating only on the training data.
7.  Forgetting to save the adapter.
8.  Ignoring the model's chat template for chat/instruction models.

------------------------------------------------------------------------

## Best Practices

-   Start with a small rank such as `r=8`.
-   Verify trainable parameter percentage.
-   Use a validation set.
-   Keep training data high quality.
-   Use a consistent instruction format.
-   Match the tokenizer/chat template to the base model.
-   Compare the base model and adapted model on the same evaluation set.
-   Increase rank only when the task needs more adapter capacity.
-   Keep the adapter separate when you need multiple task-specific
    versions.

------------------------------------------------------------------------

## Key Interview Definition
> LoRA is a parameter-efficient fine-tuning method that freezes a
> pretrained model and learns small low-rank matrices that represent
> weight updates for selected layers.
------------------------------------------------------------------------

## Key Formula

[ W' = W + `\frac{\alpha}{r}`{=tex}BA \]

Remember:

``` text
W = frozen original weight
A = trainable
B = trainable
r = rank
alpha = scaling
```
