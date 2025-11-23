# RAG_Enhanced_TQA_with_Fine_Tuning

Modular Retrieval-Augmented Question Answering system for the TQA dataset, combining ColBERT retrieval, query augmentation, knowledge refinement, and fine-tuned Llama models (Llama-2-7B and Llama-3.1-8B).

## Overview

The TQA (Textbook Question Answering) dataset contains long, concept-dense science lessons from middle-school textbooks. These passages are often too long and noisy for standard LLMs to process. This project applies a modular RAG framework—retrieval, query rewriting, refinement, and fine-tuning—to overcome long-context limitations.

**Key Results:**
- Llama-2 Baseline Test Accuracy: 82.01%
- RAG + Fine-tuning Test Accuracy: 83.72%
- RAG + Llama-3.1-8B Test Accuracy: 89.01%
- Memory Reduction via QLoRA: 28GB → 7GB

## Problem

Directly fine-tuning on full textbook passages fails due to:
- Extremely long passages exceeding model token limits  
- Truncation leading to missing key concepts  
- Large amounts of irrelevant text  
- Inconsistency between informal student questions and formal textbook style

## Solution

A modular and improved RAG pipeline:

1. **Index 7K+ textbook passages using ColBERT**  
   Token-level, relevance-guided retrieval instead of BM25/DPR.

2. **Query Augmentation**  
   Student queries are rewritten into textbook-style versions using a fine-tuned FLAN-T5 rewriter.

3. **Dual Retrieval (Original + Augmented Queries)**  
   Retrieve highly aligned passages from textbooks.

4. **Knowledge Refinement**  
   Extract essential sentences to build a concise, noise-free context.

5. **Fine-tune Llama models using QLoRA**  
   Adaptation using 4-bit training to fit GPU memory constraints.

## Requirements
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM; A100 recommended)
- Disk space: ~20GB
- Hugging Face Transformers + ColBERT + RAGatouille stack

## Results

| Model Configuration                            | Validation (%) | Test (%) |
|------------------------------------------------|----------------|----------|
| Baseline Llama-2-7B                            | 79.39          | 82.01    |
| RAG + Llama-2-7B                               | 83.86          | 83.72    |
| Query Augmenter + RAG + Llama-2-7B             | 82.63          | 83.08    |
| RAG + Llama-3.1-8B                             | 91.38          | 89.41    |
| Query Augmenter + RAG + Llama-3.1-8B           | 89.91          | 88.69    |

**Memory Usage Comparison**
- Full fine-tuning: 28GB  
- QLoRA fine-tuning: 7GB  

## Architecture
Training:
Passages → Query Augmentation → ColBERT Index → Dual Retrieval → Knowledge Refinement → Fine-tune Llama

Inference:
User Question → Rewrite → ColBERT Retrieval → Refined Context → Llama Answer Generation

## Technical Details

**Models:**
- Llama-2-7B and Llama-3.1-8B
- Fine-tuning: QLoRA (4-bit)
- LoRA rank: 64
- Learning rate: 2e-4
- Epochs: 2

**Retrieval:**
- Retriever: ColBERT v2
- Dual-query retrieval (original + rewritten)
- Top-k: 2
- Max document length: 512 tokens

**Dataset (TQA):**
- Source: CK-12 middle-school science texts
- 1,076 lessons, 78k sentences  
- 13,693 text MCQs, 12,567 diagram questions  
- Train/Val/Test: 8665 / 2528 / 2512 questions  

## Dependencies
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
ragatouille>=0.0.8
datasets>=2.14.0
tqdm
accelerate>=0.24.0
## Novelty
- **Student-aware Query Augmentation:** Rewrites student questions into textbook-style queries to reduce semantic mismatch.
- **First use of ColBERT in Textbook QA:** Token-level retrieval significantly improves relevance over BM25+DPR.
- **Modular Knowledge Refinement:** Extracts essential sentences to remove noise.
- **Educational Domain Adaptation:** Applies biomedical QA concepts (LLM-AMT) to textbook QA for the first time.

## Acknowledgments
- TQA Dataset — Allen AI  
- Llama-2 — Meta AI  
- Llama-3.1 — Meta AI  
- ColBERT — Stanford NLP  
- Built using Hugging Face and RAGatouille

## Further Improvements (In Progress)
- More robust query augmentation  
- Scaling experiments with Llama-3 models  
- Cross-domain testing on other QA benchmarks 
