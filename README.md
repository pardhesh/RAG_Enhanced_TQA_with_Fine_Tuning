# RAG_Enhanced_TQA_with_Fine_Tuning

Question answering system combining ColBERT retrieval with fine-tuned Llama 2 7B for the TQA dataset. Achieved 83.72% test accuracy.

## Overview

The TQA (Textbook Question Answering) dataset contains 7,000+ scientific passages that are often too long for language models to process effectively. This project uses retrieval-augmented generation (RAG) to solve that problem.

**Key Results:**
- Test Accuracy: 83.72% (vs 82.01% baseline)
- Validation Accuracy: 83.86% (vs 79.39% baseline)
- Memory Usage: 7GB (vs 28GB full fine-tuning)

## Problem

When fine-tuning directly on full passages:
- Passages exceed token limits and get truncated
- Important context gets cut off
- Model learns from noisy, irrelevant information

## Solution

1. Index 7K passages using ColBERT
2. For each question, retrieve top-2 most relevant passages
3. Create dataset with focused contexts
4. Fine-tune Llama 2 7B with QLoRA on this dataset


**Requirements:**
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM)
- Preferably A100 GPU from cloud like Lightning AI
- ~20GB disk space


## Results

| Metric | Baseline | RAG + Fine-tuning | Improvement |
|--------|----------|-------------------|-------------|
| Validation | 79.39% | 83.86% | +4.47% |
| Test | 82.01% | 83.72% | +1.71% |
| Memory | 28GB | 7GB | -75% |

## Architecture
```
Training:
7K Passages → ColBERT Index → Retrieve Top-2 → Create Dataset → Fine-tune Llama 2

Inference:
User Question → ColBERT Retrieval → Format Prompt → Generate Answer
```

## Technical Details

**Model:**
- Base: Llama 2 7B
- Fine-tuning: QLoRA (4-bit quantization)
- LoRA rank: 64
- Learning rate: 2e-4
- Epochs: 2

**Retrieval:**
- Model: ColBERT v2
- Top-k: 2 passages
- Max doc length: 512 tokens
- Index size: ~1.2GB


## Dependencies
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
ragatouille>=0.0.8
datasets>=2.14.0
tqdm
accelerate>=0.24.0
```

## Acknowledgments

- TQA Dataset: [Allen AI]([https://allenai.org/](https://prior.allenai.org/projects/tqa))
- Llama 2: Meta AI
- ColBERT: Stanford NLP
- Built with Hugging Face ecosystem

## Further Improvements in Process
- Query Agumentation
- Knowledge Refiner
- Testing on other LLM. Right now Llama 3.1 is in my mind
