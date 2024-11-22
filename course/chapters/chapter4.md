---
layout: default
title: "Chapter 4: Fundamental Methods for Representing Text"
---

# Chapter 4: Fundamental Methods for Representing Text

## Introduction

In this chapter, we will explore how to represent textual data in a numerical form that machine learning algorithms can process. Representing text numerically is a crucial step in NLP because most algorithms require numerical input. We will start with traditional methods and then move on to modern approaches that capture semantic relationships between words.

**What to expect in this chapter:**

- Understanding traditional text representation techniques like Bag of Words and TF-IDF.
- Learning about word embeddings and how they capture semantic meaning.
- An overview of advanced models like BERT, GPT, and T5.
- Practical implementation of BoW and TF-IDF in Python.
- Visualizing text data for better understanding.

---

## 4. Fundamental Methods for Representing Text

### 4.1 Traditional Approaches

#### 4.1.1 Bag of Words (BoW)

**Definition:**

The **Bag of Words (BoW)** model is one of the simplest methods to convert text into numerical features. In BoW, a text is represented as a bag (multiset) of its words, disregarding grammar and word order but keeping multiplicity.

**How It Works:**

1. **Vocabulary Creation:**
    
    - Collect all unique words from the corpus to form a vocabulary.
2. **Vectorization:**
    
    - For each document (text sample), create a vector of the same length as the vocabulary.
    - Each position in the vector corresponds to a word in the vocabulary.
    - The value is the count of how many times the word appears in the document.

**Example:**

Consider the following two sentences:

- Document 1: "I love natural language processing"
- Document 2: "Language processing is fun"

**Vocabulary:**

`['I', 'love', 'natural', 'language', 'processing', 'is', 'fun']`

**Vectors:**

- Document 1: `[1, 1, 1, 1, 1, 0, 0]`
- Document 2: `[0, 0, 0, 1, 1, 1, 1]`

**Advantages:**

- Simple to implement.
- Works well for text classification tasks where word frequency matters.

**Limitations:**

- **High Dimensionality:** Vocabulary size can be very large.
- **Sparsity:** Most documents will have zero counts for most words.
- **Ignores Context:** Does not capture semantics or word order.

---

#### 4.1.2 Term Frequency–Inverse Document Frequency (TF-IDF)

**Definition:**

**TF-IDF** is a statistical measure that evaluates how relevant a word is to a document in a collection of documents (corpus). It aims to reduce the weight of common words and increase the weight of words that are not very frequent across documents but are frequent in a particular document.

**How It Works:**

1. **Term Frequency (TF):**
    
    - Measures how frequently a term occurs in a document.
    - **Formula:** `TF = (Number of times term t appears in a document) / (Total number of terms in the document)`
2. **Inverse Document Frequency (IDF):**
    
    - Measures how important a term is.
    - **Formula:** `IDF = log_e(Total number of documents / Number of documents with term t in it)`
3. **TF-IDF Score:**
    
    - Multiply TF and IDF scores.
    - **Formula:** `TF-IDF = TF * IDF`

**Example:**

Assume we have a corpus of 5 documents.

- The word "language" appears in 3 documents.
- The total number of documents is 5.
- The IDF for "language" is `log_e(5 / 3) ≈ 0.51`

If "language" appears 2 times in a document of 100 words, the TF is `2 / 100 = 0.02`

The TF-IDF score is `0.02 * 0.51 ≈ 0.0102`

**Advantages:**

- Reduces the impact of common words that are less informative.
- Improves performance over simple BoW in many cases.

**Limitations:**

- Still results in high-dimensional, sparse vectors.
- Does not capture semantics or context between words.

---

### 4.2 Modern Approaches

#### 4.2.1 Introduction to Word Embeddings (Word2Vec, GloVe)

**Word Embeddings:**

Word embeddings are dense vector representations of words where semantically similar words have similar representations. Unlike BoW and TF-IDF, embeddings capture context and semantic relationships.

**Word2Vec:**

- Developed by Google.
- Uses neural networks to learn word associations.
- Two architectures:
    - **Continuous Bag of Words (CBOW):** Predicts a word based on its context.
    - **Skip-Gram:** Predicts the context based on a word.

**GloVe (Global Vectors for Word Representation):**

- Developed by Stanford.
- Combines global matrix factorization and local context window methods.
- Trains on word-word co-occurrence statistics from a corpus.

**Properties:**

- **Semantic Relationships:**
    
    - Captures analogies, e.g., `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`
- **Dimensionality Reduction:**
    
    - Embeddings typically have dimensions ranging from 50 to 300, reducing computational complexity.

**Advantages:**

- Captures semantic meaning and relationships between words.
- Dense representations are more efficient.

**Limitations:**

- Requires a large corpus for training.
- Out-of-vocabulary words (unseen words) cannot be represented unless using techniques like subword embeddings.

---

#### 4.2.2 Overview of State-of-the-Art Models (BERT, GPT, T5)

**Transformer Architecture:**

Introduced by Vaswani et al. in 2017, transformers rely entirely on self-attention mechanisms to model relationships in sequential data, allowing for better handling of long-range dependencies.

**BERT (Bidirectional Encoder Representations from Transformers):**

- Developed by Google.
- **Bidirectional:** Considers context from both left and right of a word.
- **Pre-training Tasks:**
    - **Masked Language Modeling (MLM):** Predicting masked words in a sentence.
    - **Next Sentence Prediction (NSP):** Predicting whether one sentence follows another.

**GPT (Generative Pre-trained Transformer):**

- Developed by OpenAI.
- **Unidirectional:** Processes text from left to right.
- **Focus:** Language generation tasks.
- **GPT-2 and GPT-3:** Larger versions with significant improvements in language understanding and generation.

**T5 (Text-To-Text Transfer Transformer):**

- Developed by Google.
- Treats every NLP problem as a text-to-text task.
- **Versatility:** Can handle translation, summarization, question answering, etc.

**Advantages:**

- **Contextualized Representations:** Words are represented differently depending on context.
- **Fine-tuning:** Pre-trained on large corpora and can be fine-tuned for specific tasks with less data.

**Limitations:**

- **Computationally Intensive:** Requires significant resources to train and run.
- **Data Hungry:** Pre-training requires massive datasets.

---

### Practice: Implementing BoW and TF-IDF in Python; Visualizing Text Data

#### Practice Task 1: Implementing Bag of Words

**Objective:**

- Learn how to convert text documents into BoW vectors.
- Understand how to use scikit-learn's `CountVectorizer`.

**Instructions:**

1. **Create a small corpus of text documents.** For example, you can use a list of sentences or short paragraphs relevant to a topic of your choice.
        
2. **Initialize a `CountVectorizer`** from scikit-learn.
        
3. **Fit the `CountVectorizer` to your corpus** and transform the text data into BoW vectors.
        
4. **Examine the resulting vocabulary** and the BoW representation of your documents.
        
5. **Reflect on the BoW model:**
        
    - How does the BoW representation handle word order?
    - What are the limitations of this representation?
    
**Hints:**

- Refer to the scikit-learn documentation for `CountVectorizer` for guidance on its usage.
- Consider printing out the vocabulary and the transformed vectors to better understand the output.

---

#### Practice Task 2: Implementing TF-IDF

**Objective:**

- Learn how to compute TF-IDF scores for your text corpus.
- Understand the impact of TF-IDF weighting compared to simple term counts.

**Instructions:**

1. **Use the same corpus from Practice Task 1** or create a new one.
        
2. **Initialize a `TfidfVectorizer`** from scikit-learn.
        
3. **Fit the `TfidfVectorizer` to your corpus** and transform the text data into TF-IDF vectors.
        
4. **Analyze the TF-IDF scores:**
        
    - Which words have higher TF-IDF scores?
    - How do the TF-IDF scores compare to the term frequencies in the BoW model?
    
5. **Consider how TF-IDF weighting addresses some limitations of the BoW model.**
        
    
**Hints:**

- The scikit-learn documentation for `TfidfVectorizer` provides examples and explanations.
- You can compare the output of `CountVectorizer` and `TfidfVectorizer` to see the differences.

---

#### Practice Task 3: Visualizing Text Data

**Objective:**

- Visualize the most frequent words in your corpus.
- Use visualization tools to gain insights into your text data.

**Instructions:**

1. **Aggregate your text data** into a single string or a suitable format for visualization.
        
2. **Choose a visualization method:**
        
    - **Word Cloud:** Generate a word cloud to display the most frequent words visually.
    - **Bar Plot:** Create a bar chart showing the frequencies or TF-IDF scores of the top N words.
        
3. **Use appropriate Python libraries** for visualization, such as `matplotlib`, `seaborn`, or `wordcloud`.
        
4. **Interpret your visualizations:**
        
    - What are the most prominent words?
    - Does the visualization align with your expectations about the corpus?
    
**Hints:**

- Ensure you preprocess your text data appropriately (e.g., removing stopwords) before visualization.
- For word clouds, the `wordcloud` library is useful.
- For bar plots, you might need to extract word frequencies from your BoW or TF-IDF vectors.

---

**Note to Students:**

These practice tasks are designed to reinforce your understanding of text representation methods. By implementing these tasks on your own, you will gain hands-on experience with transforming and analyzing text data, which is fundamental for NLP tasks.

_Check [Sample Answers](answers/chapter4_practice.md) for insights and examples._

---

### Additional Resources

- **scikit-learn Documentation:**
    
    - CountVectorizer
    - TfidfVectorizer
- **Gensim Library:**
    
    - For advanced implementations of word embeddings like Word2Vec and GloVe.
- **Reading Materials:**
    
    - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) by Mikolov et al. (Word2Vec)
    - [Glove: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) by Pennington et al.

---

### Conclusion

In this chapter, we've explored both traditional and modern methods for representing text data numerically. Understanding these methods is essential for selecting the right approach for your NLP tasks. While BoW and TF-IDF are suitable for simpler models and tasks, word embeddings and transformer-based models offer deeper insights into language semantics and are powerful tools for more complex applications.

---

**[Next](chapter5.md):** In the next chapter, we will dive into core NLP tasks, including text classification, information extraction, and text generation. You'll apply the knowledge from this chapter to build and evaluate models for these tasks.
