---
layout: default
title: "Chapter 5: Core NLP Tasks"
---

# Chapter 5: Core NLP Tasks

## Introduction

In this chapter, we will delve into some of the core tasks in Natural Language Processing (NLP) that are fundamental to many applications. These tasks include text classification, information extraction, text generation, and machine translation. By exploring these areas, you will gain hands-on experience in building models that can understand and generate human language.

**What to expect in this chapter:**

- Understanding text classification and building a basic sentiment analysis model.
- Learning about information extraction with a focus on Named Entity Recognition (NER).
- Exploring text generation using language models, specifically N-gram models.
- Gaining insights into machine translation and the various approaches used.

---

## 5. Core NLP Tasks

### 5.1 Text Classification

#### 5.1.1 Introduction to Text Classification

**Text Classification** is the process of assigning predefined categories or labels to textual data. It is one of the most common tasks in NLP and is essential for organizing and structuring information.

**Applications:**

- **Spam Detection:** Classifying emails as spam or not spam.
- **Sentiment Analysis:** Determining the sentiment (positive, negative, neutral) of reviews or social media posts.
- **Topic Classification:** Categorizing news articles into topics like sports, politics, technology, etc.

#### 5.1.2 Sentiment Analysis

**Sentiment Analysis** is a specific type of text classification that focuses on identifying the emotional tone behind a body of text. It is widely used in marketing, customer service, and social media monitoring to understand public opinion.

**Approaches:**

- **Rule-Based Methods:** Use predefined lexicons to assign sentiment scores.
- **Machine Learning Methods:** Train classifiers using labeled datasets.
- **Deep Learning Methods:** Use neural networks for feature extraction and classification.

#### 5.1.3 Building a Basic Text Classification Model

We will build a simple sentiment analysis model using a machine learning approach.

**Dataset:**

- We'll use the **Movie Review** dataset from NLTK, which contains positive and negative movie reviews.

**Steps:**

1. **Data Loading and Preprocessing**
2. **Feature Extraction using TF-IDF**
3. **Model Training with Naive Bayes Classifier**
4. **Model Evaluation**

**Example Code:**

```python
import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Download the movie_reviews corpus
nltk.download('movie_reviews')

# Load the reviews and labels
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Prepare the data
reviews = [" ".join(words) for words, category in documents]
labels = [category for words, category in documents]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Model training using Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train_features, y_train)

# Predictions
y_pred = classifier.predict(X_test_features)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

**Output:** 

Accuracy: 0.81  
Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| neg   | 0.80      | 0.83   | 0.81     | 198     |
| pos   | 0.83      | 0.79   | 0.81     | 202     |
| accuracy | -      | -      | 0.81     | 400     |
| macro avg | 0.81  | 0.81   | 0.81     | 400     |
| weighted avg | 0.81 | 0.81 | 0.81     | 400     |

**Note:** The accuracy may vary each time you run the code due to random shuffling.

---

#### Practice: Building a Basic Text Classification Model

**Objective:**

- Build your own sentiment analysis model using a different dataset or tweaking the parameters.

**Instructions:**

1. **Choose a Dataset:**
    - Use the **IMDb Movie Reviews** dataset or any other dataset of your choice.
    - Ensure the dataset has text data and corresponding labels.

2. **Data Preprocessing:**
    - Clean the text data (lowercasing, removing punctuation, stopwords, etc.).
    - Split the data into training and test sets.

3. **Feature Extraction:**
    - Experiment with different feature extraction methods (e.g., CountVectorizer, TF-IDF).
    - Adjust parameters like max_features, ngram_range, etc.

4. **Model Training:**
    - Try different classifiers (e.g., Logistic Regression, Support Vector Machines).
    - Use cross-validation to tune hyperparameters.

5. **Evaluation:**
    - Evaluate your model using accuracy, precision, recall, and F1-score.
    - Analyze misclassified examples to understand model limitations.

6. **Report:**
    - Summarize your findings.
    - Discuss how different preprocessing steps or models affected performance.

**Hints:**

- Use scikit-learn's train_test_split, TfidfVectorizer, and classifiers.
- For datasets, you can use nltk.corpus or download datasets from sources like Kaggle.

---

### 5.2 Information Extraction

#### 5.2.1 Introduction to Information Extraction

**Information Extraction (IE)** involves automatically extracting structured information from unstructured text data. It aims to identify and classify key elements in text.

**Applications:**

- **Named Entity Recognition (NER):** Identifying entities like people, organizations, locations.
- **Relation Extraction:** Discovering relationships between entities.
- **Event Extraction:** Identifying events and their attributes.

#### 5.2.2 Named Entity Recognition (NER)

**Named Entity Recognition** is the process of locating and classifying named entities in text into predefined categories.

**Common Categories:**

- **PERSON:** People’s names.
- **ORG:** Organizations.
- **GPE:** Geopolitical entities (countries, cities).
- **DATE:** Dates and times.
- **MONEY:** Monetary values.

**Approaches:**

- **Rule-Based Methods:** Using regular expressions and pattern matching.
- **Statistical Methods:** Using machine learning algorithms trained on annotated data.
- **Neural Network Models:** Utilizing architectures like LSTMs and transformers.

#### 5.2.3 Annotating Text and Extracting Entities

We will use the spaCy library to perform NER.

**Example Code:**

```python
import spacy  

# Load the pre-trained model
nlp = spacy.load('en_core_web_sm')  

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook said in a conference in San Francisco."

# Process the text
doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Output:**

```python
Apple ORG U.K. GPE $1 billion MONEY Tim Cook PERSON San Francisco GPE
```

---

#### Practice: Annotating Text and Extracting Entities

**Objective:**

- Practice extracting named entities from text and understand their categories.

**Instructions:**

1. **Choose or Create a Text Sample:**
    - Use a news article, a Wikipedia page, or write your own paragraph.
2. **Load the spaCy Model:**
    - Install spaCy and download the English model if you haven't already:
    
    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm
    ```
3. **Process the Text:**
    - Use spaCy to process the text and extract entities.
4. **Analyze the Entities:**
    - Print out the entities and their labels.
    - Are all entities correctly identified?
    - Note any misclassifications or missing entities.
5. **Visualize the Entities (Optional):**
    
    ```python
    from spacy import displacy
    displacy.render(doc, style='ent', jupyter=True)
    ```

**Hints:**

- Try using different models (e.g., `en_core_web_md`) for potentially better performance.
- For more advanced analysis, consider training a custom NER model with your own annotations.

---

### 5.3 Text Generation

#### 5.3.1 Basics of Language Models

**Language Models** are statistical models that calculate the probability of a sequence of words. They are fundamental for tasks like text generation, speech recognition, and machine translation.

**N-gram Models:**

- An **N-gram** is a contiguous sequence of N items from a given text.
  - **Unigram:** Single words.
  - **Bigram:** Sequences of two words.
  - **Trigram:** Sequences of three words.

**Markov Assumption:**

- The probability of a word depends only on the previous N-1 words.

**Formula:**

- For a trigram model: \( P(w_n | w_{n-2}, w_{n-1}) \)

#### 5.3.2 Generating Text Using a Simple N-gram Model

We will build a basic bigram model to generate text.

**Example Code:**

```python
import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import random
import nltk.data

# Load sample text (e.g., Gutenberg corpus)
nltk.download('gutenberg')
from nltk.corpus import gutenberg

# Choose a text
text = gutenberg.raw('melville-moby_dick.txt')

# Tokenize sentences
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sent_detector.tokenize(text)

# Tokenize words and create bigrams
model = defaultdict(lambda: defaultdict(lambda: 0))

for sentence in sentences:
    words = nltk.word_tokenize(sentence.lower())
    for w1, w2 in ngrams(words, 2):
        model[w1][w2] += 1

# Convert counts to probabilities
for w1 in model:
    total_count = float(sum(model[w1].values()))
    for w2 in model[w1]:
        model[w1][w2] /= total_count

# Generate text
def generate_text(start_word, length=50):
    text = [start_word]
    for _ in range(length):
        w1 = text[-1]
        if w1 in model:
            choices = list(model[w1].keys())
            probs = list(model[w1].values())
            next_word = random.choices(choices, probs)[0]
            text.append(next_word)
        else:
            break
    return ' '.join(text)

# Generate a text starting with 'whale'
generated_text = generate_text('whale')
print(generated_text)
```

**Output:**

`whale . but it is not , and the whale 's spout ? '' '' i have seen him . '' '' and i am not a whale , and the whale 's spout ? '' '' i have seen him .`

**Note:** The generated text may not be coherent due to the simplicity of the model.

---

#### Practice: Generating Text Using a Simple N-gram Model

**Objective:**

- Build and experiment with N-gram language models for text generation.

**Instructions:**

1. **Select a Text Corpus:**
    - Use a text corpus from NLTK (e.g., Shakespeare, Austen) or your own text data.

2. **Build N-gram Models:**
    - Experiment with different values of N (e.g., unigram, bigram, trigram).
    - Implement smoothing techniques if necessary (e.g., Laplace smoothing).

3. **Generate Text:**
    - Generate sentences or paragraphs starting with a chosen word or sequence.
    - Compare the coherence of text generated with different N-gram sizes.

4. **Analyze Results:**
    - How does increasing N affect the quality of the generated text?
    - Discuss the limitations of N-gram models.

**Hints:**

- For larger N, you may need more data to get meaningful results.
- Consider using nltk's `ConditionalFreqDist` and `ConditionalProbDist` for managing probabilities.

---

### 5.4 Machine Translation

#### 5.4.1 Key Approaches to Machine Translation

Machine Translation (MT) involves automatically translating text from one language to another.

**Approaches:**

1. **Rule-Based Machine Translation (RBMT):**
    - Uses linguistic rules and dictionaries.
    - Relies on morphological, syntactic, and semantic analysis.
    - **Advantages:** Explainable translations.
    - **Limitations:** Requires extensive manual work to create rules.

2. **Statistical Machine Translation (SMT):**
    - Uses statistical models derived from bilingual text corpora.
    - **Phrase-Based SMT:** Translates sequences of words (phrases).
    - **Advantages:** Learns from data, no need for handcrafted rules.
    - **Limitations:** May produce grammatically incorrect translations.

3. **Neural Machine Translation (NMT):**
    - Uses deep neural networks to model the translation process.
    - **Sequence-to-Sequence Models:** Encoder-decoder architectures with attention mechanisms.
    - **Advantages:** Improved fluency and handling of long sentences.
    - **Limitations:** Requires large amounts of data and computational resources.

#### 5.4.2 Translating Text Using Available Libraries

We will use the `transformers` library by Hugging Face to perform machine translation using a pre-trained model.

**Example Code:**

```python
from transformers import MarianMTModel, MarianTokenizer

# Choose model for English to French translation
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Text to translate
text = "Hello, how are you?"

# Tokenize text
tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

# Perform translation and decode
translated = model.generate(**tokens)
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print("Original Text:", text)
print("Translated Text:", translated_text)
```

**Output:**

```python
Original Text: Hello, how are you?
Translated Text: Bonjour, comment ça va ?
```

---

#### Practice: Translating Text Using Available Libraries

**Objective:**

- Use pre-trained models to translate text between different languages.

**Instructions:**

1. **Install Required Libraries:**

    ```python
    pip install transformers sentencepiece
    ```

2. **Choose Language Pairs:**

    - Visit the Helsinki-NLP models to find available language pairs.
    - Select a model for your desired translation direction (e.g., English to German).

3. **Write Code to Translate Text:**

    - Load the appropriate tokenizer and model.
    - Prepare your text and perform translation.

4. **Experiment:**

    - Translate different sentences.
    - Try both directions (e.g., English to German and German to English).
    - Evaluate the quality of the translations.

5. **Discuss:**

    - How does the translation quality compare to online translators?
    - What limitations did you observe?

**Hints:**

- Make sure to use the correct model name for the language pair.
- For better results, use models trained on high-resource language pairs.

---

### Conclusion

In this chapter, we've explored some of the core tasks in NLP, including text classification, information extraction, text generation, and machine translation. By working through these tasks, you have gained practical experience in building models that can analyze and generate human language. These skills are foundational for more advanced NLP topics and applications.

---

**[Next](chapter6.md):** In the next chapter, we will cover the basics of machine learning for NLP, including model evaluation and common algorithms used in NLP tasks.
