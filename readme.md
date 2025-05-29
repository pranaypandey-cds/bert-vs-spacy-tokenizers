# BERT vs SpaCy Tokenizer Comparison on IMDB Dataset

This repository contains a comparative analysis of the `bert-base-uncased` tokenizer and the SpaCy tokenizer (`spacy.blank('en')`) using the Large Movie Review (IMDB) dataset.

## 📌 Project Overview

The goal of this project is to evaluate how two fundamentally different tokenization methods — BERT's subword tokenizer and SpaCy's rule-based tokenizer — process movie reviews for sentiment classification. The comparison includes an in-depth look at vocabulary size, token informativeness, and preprocessing complexity.

## 📚 Dataset

The [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) is used in this analysis. Specifically:
- **Subset**: Training set
- **Size**: 25,000 movie reviews
  - 12,500 Positive
  - 12,500 Negative

## 🧠 Why BERT Tokenizer?

- **Subword Tokenization (WordPiece)**: Breaks down words into smaller, meaningful units. For example, “unpredictable” → `["un", "##predict", "##able"]`, allowing robust handling of rare or unseen words (OOV).
- **Fixed Vocabulary (~30k tokens)**: Reduces model complexity and prevents vocabulary explosion.
- **Preprocessing Efficiency**: Automatically lowercases text and handles punctuation effectively.
- **Model Compatibility**: Designed specifically for transformer-based models like BERT, enabling more compact and efficient input representations.

## 🔍 About SpaCy's Tokenizer

- **Rule-Based Tokenization**: SpaCy’s tokenizer relies on predefined linguistic rules and token boundary patterns, rather than subword strategies.
- **No Subword Handling**: Words are not broken into smaller units, so OOV words are kept as-is or split based on punctuation and whitespace.
- **Dynamic Vocabulary**: Every new word contributes to vocabulary growth, which can lead to sparsity in downstream models.
- **Configuration Used**: `spacy.blank('en')` was used to isolate and test the base English tokenizer, without the influence of spacy's pre-trained models.


## 🧪 Approach

1. **Dataset Loading**  
   Load and preprocess the IMDB dataset from Hugging Face.

2. **Tokenization**  
   Apply both the BERT and SpaCy tokenizers on the same dataset.

3. **Entropy Calculation**  
   Compute Information Value (IV) of each token to assess informativeness.

4. **Top Tokens Extraction**  
   Extract the top 1000 tokens from each tokenizer's output based on IV scores.

5. **Comparison**  
   Analyze token overlap, sparsity, entropy distribution, and processing behavior.

6. **Conclusion**  
   Evaluate which tokenizer provides more efficient and informative representations for sentiment classification.

## 📊 Results Summary

- **BERT tokenizer** produced a smaller, denser vocabulary thanks to subword decomposition.
- **SpaCy tokenizer** resulted in more unique and sparse tokens due to its word-level rule-based strategy.
- BERT tokens showed higher **information value concentration**, meaning fewer tokens carried more useful information.

## 🛠️ Requirements

- Python 3.7+
- `transformers`
- `spacy`
- `datasets`
- `numpy`, `pandas`, `matplotlib`
