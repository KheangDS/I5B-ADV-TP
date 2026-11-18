# Supervised Fine-Tuning LLMs — Summary


Supervised fine-tuning (SFT) adapts a pretrained model to a specific task by continuing training on a smaller dataset of input–target examples. It builds on knowledge acquired during pretraining rather than training a model from scratch.


## 1. Model and Preprocessor Setup

### Model selection

Choose a model by considering:

- **Task alignment:** suitability of its pretraining and capabilities for the target task.
- **Architecture:** compatibility with the required inputs, outputs, and adaptation method.
- **Size and complexity:** balance capability against available memory and compute.
- **Existing performance:** benchmarks and evidence of effective transfer learning.
- **Ecosystem support:** documentation, community resources, and framework compatibility.

### Model loaders

A model loader brings a pretrained model’s architecture and weights into memory for inference or further training.

| Loader approach | Purpose | Examples |
| --- | --- | --- |
| Model-specific | Explicitly select a known architecture | `BertForSequenceClassification`, `GPT2LMHeadModel`, `T5ForConditionalGeneration` |
| AutoModel | Infer the architecture from the model identifier while selecting the task-specific head | `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForCausalLM` |

Other task families include masked language modeling, sequence-to-sequence generation, question answering, speech recognition, and image segmentation. Choose the loader that matches the task, not just the model name.

### Preprocessors

`AutoProcessor` loads the appropriate preprocessing components for a supported model. Depending on the input modality, preprocessing can involve a tokenizer, image processor, audio feature extractor, or a combination of these.

**Key point:** the model, processor, and input format must be compatible.

### Quantization

Quantization reduces the precision used to represent model weights and/or activations, primarily to reduce memory and storage requirements. Lower precision may also improve efficiency, depending on hardware and implementation.

The slides introduce `BitsAndBytesConfig`, low-bit model loading, compute dtypes, device placement, and optional attention implementations. These settings must match the model and hardware.

> **Technical clarification:** slide 13 mixes 8-bit loading with names resembling 4-bit options. NF4, double quantization, and the corresponding configurable compute dtype belong to the `bnb_4bit_*` settings, not the displayed `bnb_8bit_*` names. FP16 is reduced-precision floating point, distinct from integer quantization. Treat the slide snippets as illustrations rather than copy-ready code.

## 2. Data Preparation

**Pipeline:**

```text
Raw data → Conversation formatting → Chat template
         → Tokenization / preprocessing → Batch collation
```

### Raw data

Start with structured JSON, CSV, or text. Use accessible fields such as `prompt`, `answer`, and, for multimodal tasks, `image`.

The examples load JSON files using `datasets.load_dataset()` and include image-based tasks such as object detection and robot-arm instructions.

### Conversation formatting

Convert each example into messages with roles and content:

- **User:** the instruction or question, optionally accompanied by an image.
- **Assistant:** the desired response used as the supervised target.

Multimodal content can contain separate typed items for text and images. The demonstrated conversion returns a `messages` list; its image preprocessing opens the image, resizes it, and converts it to RGB.

Determine the expected message schema from the model documentation, chat template, or example training data.

### Chat templates

`apply_chat_template()` serializes structured messages into the model’s expected conversational format, including role delimiters and special tokens. Depending on its options, it can return formatted text or tokenized inputs.

The slides show an inference-style example with `tokenize=False` and `add_generation_prompt=True`, which appends the start of an assistant response. For supervised training, formatting must include the target answer and use options appropriate to the training pipeline.

### Data collation

A data collator prepares examples into batches. Depending on the implementation, it may:

- Perform tokenization or multimodal preprocessing.
- Pad sequences to compatible lengths.
- Construct masks and training labels.
- Apply chat templates.

Pass the collator to the trainer. The vision example uses `UnslothVisionDataCollator(model, tokenizer)`.

## 3. Parameter-Efficient Fine-Tuning

### PEFT

**Parameter-Efficient Fine-Tuning (PEFT)** trains a small subset of parameters rather than updating the entire model, reducing training memory and computational requirements.

### LoRA

**Low-Rank Adaptation (LoRA)** freezes the original weights and learns a low-rank update. Using the slides’ simplified notation:

```text
W′ = W + AB
```

For a `512 × 512` weight matrix with rank `r = 8`:

- Full matrix: `512 × 512 = 262,144` parameters.
- LoRA matrices: `512 × 8 + 8 × 512 = 8,192` parameters.
- About **96.9% fewer trainable parameters for that matrix**.

The product `AB` still has shape `512 × 512`, but only its smaller factors are trained. Actual LoRA implementations also scale the update.

### LoRA configuration

The Unsloth example uses:

| Setting | Example | Purpose |
| --- | --- | --- |
| `r` | `16` | Rank of the learned update |
| `lora_alpha` | `16` | Controls update scaling |
| `lora_dropout` | `0.05` | Regularization |
| `bias` | `"none"` | Leaves bias parameters untrained |
| `random_state` | `3407` | Reproducible initialization |
| `use_rslora` | `True` | Rank-stabilized LoRA |

It also enables adaptation of vision, language, attention, and MLP components. Inspect `print(model)` to identify valid target module names, and use `model.print_trainable_parameters()` to verify the trainable fraction.

Increasing rank increases adaptation capacity and resource use; it does **not** guarantee higher accuracy.

## 4. Supervised Trainer Setup

### Example training settings

| Setting | Slide value | Meaning |
| --- | --- | --- |
| Per-device batch size | `2` | Examples processed per device per microbatch |
| Gradient accumulation | `4` | Accumulate gradients before an optimizer update |
| Warmup steps | `5` | Gradually increase the learning rate initially |
| Training epochs | `1` | One full pass through the dataset |
| Learning rate | `2e-4` | Initial optimization rate |
| Precision | BF16 if supported, otherwise FP16 | Reduced-precision training |
| Logging interval | `5` steps | Metric reporting frequency |
| Checkpoint interval | `50` steps | Saving frequency |
| Optimizer | `adamw_8bit` | Memory-efficient AdamW |
| Weight decay | `0.01` | Regularization |
| Scheduler | `linear` | Learning-rate schedule |
| Seed | `3407` | Reproducibility |
| Maximum sequence length | `2048` | Input sequence-length limit |
| Reporting | `wandb` | Weights & Biases integration |

On one device, batch size `2` with accumulation `4` gives an effective batch size of **8 examples per optimizer update**.

For its vision pipeline, the example also sets `remove_unused_columns=False`, `dataset_text_field=""`, and `dataset_kwargs={"skip_prepare_dataset": True}` so the custom collator can handle the multimodal data.

### Starting training

Provide the trainer with the model, processor/tokenizer, data collator, configuration, training dataset, and optionally an evaluation dataset. The Unsloth example first enables training with `FastVisionModel.for_training(model)`, then starts optimization with:

```python
trainer.train()
```

These are example settings, not universal defaults. Constructor arguments and supported options depend on library versions.

## 5. Training Diagnosis

### Training loss versus validation loss

| Pattern | Likely interpretation |
| --- | --- |
| Training and validation loss decrease | Learning with improving generalization |
| Training loss decreases while validation loss rises | Possible overfitting |
| Both losses remain high | Possible underfitting or a data/configuration problem |

Training loss measures fit to the training examples. Validation loss measures performance on held-out data. Very low training loss alone does not prove either success or overfitting; compare it with validation behavior.

### Weights & Biases

Set `report_to="wandb"` and initialize/authenticate W&B before training as needed. Its dashboards support comparing runs and tracking loss, learning rate, gradient norm, training steps, and epochs.

## 6. Checkpoints

Checkpoints preserve progress so you can:

- Resume interrupted training.
- Avoid restarting from scratch.
- Evaluate intermediate versions and select a good model.
- Reload trained weights for inference.

A resumable training checkpoint can contain model or adapter weights, optimizer state, and training metadata. An adapter-only save is not necessarily sufficient to resume the complete optimizer state.

The slides show two loading approaches:

1. Load a checkpoint directly through a compatible loader.
2. Load the original base model and attach the saved adapter with `model.load_adapter(...)`.

**Important:** a LoRA adapter depends on the compatible base model; it is not a standalone replacement for the full model.

