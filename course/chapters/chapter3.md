---
layout: default
title: "Chapter 1: Introduction to the Course"
---

# **Chapter 3: Working with Text Data**

---

### **Overview**

In this chapter, we will delve deeper into the essential steps for handling and preprocessing text data in Natural Language Processing (NLP). Proper handling of text data is crucial because it forms the foundation upon which all subsequent NLP tasks are built. This chapter is divided into two main sections:

1. **Basics of Working with Text**
    
    - Understanding text formats and encodings.
    - Reading and writing text data in Python.
    - Handling common issues with text data.
2. **Text Preprocessing**
    
    - Detailed steps in preprocessing text.
    - Advanced preprocessing techniques.
    - Practical examples with code snippets.

---

### **3.1 Basics of Working with Text**

#### **3.1.1 Text Formats**

Text data comes in various formats, each suited for different types of data representation. Understanding these formats is essential for effectively parsing and processing text.

**a. Plain Text (TXT):**

- **Description:** Unstructured text files containing raw text data.
- **Use Cases:** Storing simple text data, logs, or any data where structure is not required.
- **Example Content:**
    
    ```python
    This is a sample text file. It contains multiple lines of text.
    ```

**b. Comma-Separated Values (CSV):**

- **Description:** Tabular data where each row represents a record, and columns are separated by commas.
- **Use Cases:** Datasets where each record has multiple attributes, such as sentiment datasets with text and labels.
- **Example Content:**
    
    ```python
    review,rating
    "The product is great!",5
    "Not satisfied with the quality.",2
    ```

**c. JavaScript Object Notation (JSON):**

- **Description:** A lightweight data-interchange format that's easy for humans to read and write, and easy for machines to parse and generate.
- **Use Cases:** APIs, complex datasets with hierarchical structures.
- **Example Content:**
    
    ```python
    {
        "reviews": [
            {
                "review_id": 1,
                "text": "The product is great!",
                "rating": 5
            },
            {
                "review_id": 2,
                "text": "Not satisfied with the quality.",
                "rating": 2
            }
        ]
    }
    ```

**d. Extensible Markup Language (XML):**

- **Description:** A markup language that defines a set of rules for encoding documents in a format readable for both humans and machines.
- **Use Cases:** Configuration files, data exchange between systems.
- **Example Content:**
    
    ```python
    <reviews>
        <review id="1">
            <text>The product is great!</text>
            <rating>5</rating>
        </review>
        <review id="2">
            <text>Not satisfied with the quality.</text>
            <rating>2</rating>
        </review>
    </reviews>
    ```

---

#### **3.1.2 Text Encoding**

Text encoding is a method of converting text data into a form that can be easily stored and transmitted by computers.

**Common Encodings:**

- **UTF-8 (Unicode Transformation Format - 8-bit):**
    
    - **Description:** Supports all Unicode characters, making it suitable for multilingual text.
    - **Advantages:** Backward compatible with ASCII, efficient for English text, and standard on the web.
    - **Usage Tip:** Recommended for most applications to avoid encoding issues.
- **ASCII (American Standard Code for Information Interchange):**
    
    - **Description:** Encodes 128 specified characters into seven-bit integers.
    - **Limitations:** Only supports basic English characters, no support for accents or other languages.
- **ISO-8859-1 (Latin-1):**
    
    - **Description:** Supports Western European languages.
    - **Limitations:** Insufficient for texts containing characters from multiple languages.

**Common Encoding Issues:**

- **UnicodeDecodeError:** Occurs when a byte sequence does not match the expected encoding.
- **Mojibake:** Garbled text resulting from decoding text with the wrong encoding.

**Practical Tips:**

- Always specify the encoding when reading or writing files in Python.
- Use `encoding='utf-8'` unless you have a specific reason not to.

**Example: Reading a File with Specified Encoding**
    
    ```python
    file_path = 'data/sample.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    ```

---

#### **3.1.3 Reading and Writing Text in Python**

**Reading Text Files:**

- **Reading the Entire File:**
    
    ```python
    with open('data/sample.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        print(content)
    ```
    
- **Reading Line by Line:**
    
    ```python
    with open('data/sample.txt', 'r', encoding='utf-8') as file:
        for line in file:
            print(line.strip())
    ```

**Writing Text Files:**

- **Writing to a File:**
    
    ```python
    with open('data/output.txt', 'w', encoding='utf-8') as file:
        file.write('This is an output file.\n')
        file.write('Writing multiple lines.')
    ```

**Reading CSV Files:**

- Using the `csv` module:
    
    ```python
    import csv

    with open('data/sample.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)
    ```
    
- Using `pandas`:
    
    ```python
    import pandas as pd

    df = pd.read_csv('data/sample.csv', encoding='utf-8')
    print(df.head())
    ```

**Reading JSON Files:**

- Using the `json` module:
    
    ```python
    import json

    with open('data/sample.json', 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        print(data)
    ```
    
---

#### **3.1.4 Handling Common Issues with Text Data**

**Dealing with Large Files:**

- Read files in chunks to avoid memory issues.
    
    ```python
    def read_in_chunks(file_object, chunk_size=1024):
        while True:
            data = file_object.read(chunk_size)
            if not data:
                break
            yield data

    with open('large_file.txt', 'r', encoding='utf-8') as file:
        for chunk in read_in_chunks(file):
            process(chunk)
    ```

**Handling Missing or Corrupt Data:**

- Use try-except blocks to handle exceptions.
    
    ```python
    try:
        with open('data/sample.txt', 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
    ```

**Normalizing Text Data:**

- Remove or replace special characters.
    
- Use Unicode normalization for consistent representation.
    
    ```python
    import unicodedata

    text = 'Café'
    normalized_text = unicodedata.normalize('NFKD', text)
    print(normalized_text)  # Output: 'Café'
    ```

---

#### **3.2 Text Preprocessing**

Text preprocessing involves transforming raw text into a clean and analyzable format. It's a critical step that significantly impacts the performance of NLP models.

#### **3.2.1 Basic Preprocessing Steps**

**1. Tokenization**

- **Definition:** Splitting text into smaller units called tokens (e.g., words, sentences).
- **Types of Tokenization:**
    - **Word Tokenization:** Splitting text into words.
    - **Sentence Tokenization:** Splitting text into sentences.
- **Tools and Libraries:**
    - **NLTK (Natural Language Toolkit):**
        
        ```python
        import nltk
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize, sent_tokenize

        text = "Hello world! NLP is exciting."
        word_tokens = word_tokenize(text)
        sentence_tokens = sent_tokenize(text)

        print("Word Tokens:", word_tokens)
        print("Sentence Tokens:", sentence_tokens)
        ```
        
    - **spaCy:**
        
        ```python
        import spacy

        nlp = spacy.load('en_core_web_sm')
        doc = nlp("Hello world! NLP is exciting.")

        word_tokens = [token.text for token in doc]
        sentence_tokens = [sent.text for sent in doc.sents]

        print("Word Tokens:", word_tokens)
        print("Sentence Tokens:", sentence_tokens)
        ```

**2. Normalization**

- **Lowercasing:**
    
    - Convert all text to lowercase to ensure uniformity.
    - Example: "Hello" and "hello" are treated the same.
    
    ```python
    text = text.lower()
    ```
    
- **Removing Punctuation:**
    
    - Strip punctuation marks which may not contribute to analysis.
    - Use regular expressions or string methods.
    
    ```python
    import string

    text = text.translate(str.maketrans('', '', string.punctuation))
    ```
    
- **Removing Numbers:**
    
    - Optionally remove numbers if they are not relevant.
    
    ```python
    import re

    text = re.sub(r'\d+', '', text)
    ```

**3. Stopword Removal**

- **Definition:** Stopwords are common words that may not add significant meaning (e.g., "the," "is," "in").
    
- **Why Remove Stopwords?**
    
    - Reduce noise in data.
    - Improve computational efficiency.
- **Implementation:**
    
    ```python
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    ```

**4. Stemming and Lemmatization**

- **Stemming:**
    
    - **Definition:** Reduces words to their root form by removing suffixes.
    - **Algorithm:** Porter Stemmer, Snowball Stemmer.
    
    ```python
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in word_tokens]
    ```
    
- **Lemmatization:**
    
    - **Definition:** Reduces words to their base or dictionary form (lemma).
    - **Algorithm:** WordNet Lemmatizer.
    
    ```python
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    ```
    
- **Differences:**
    
    - Stemming is faster but less accurate.
    - Lemmatization is slower but more accurate as it considers the context.

**5. Handling URLs, Emails, and Mentions**

- **Removal:**
    
    - Use regular expressions to identify and remove patterns.
    
    ```python
    import re

    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)          # Remove emails
    text = re.sub(r'@\w+', '', text)             # Remove mentions
    ```

**6. Handling Emojis and Special Characters**

- **Option 1: Remove Them**
    
    ```python
    import re

    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    ```
    
- **Option 2: Convert Them to Text**
    
    - Use `emoji` library to demojize.
    
    ```python
    import emoji

    text = emoji.demojize(text)
    ```

---


#### **3.2.2 Advanced Preprocessing Techniques**

**1. Text Standardization**

- Correct common misspellings, expand contractions (e.g., "don't" → "do not").
    
    ```python
    from contractions import fix
    text = fix("I don't think so.")  # Output: "I do not think so."
    ```

**2. Spelling Correction**

- Use libraries like `TextBlob` or `pyspellchecker`.
    
    ```python
    from textblob import TextBlob
    text = str(TextBlob(text).correct())
    ```

**3. Part-of-Speech (POS) Tagging**

- Assign grammatical categories (e.g., noun, verb) to words.
    
    ```python
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    print(pos_tags)
    ```

**4. Named Entity Recognition (NER)**

- Identify and classify named entities in text (e.g., persons, organizations).
    
    ```python
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)
    ```

**5. Removing Rare Words or Frequent Words**

- Remove words that appear too infrequently or too frequently.
    
    ```python
    from collections import Counter
    word_counts = Counter(word_tokens)
    rare_words = [word for word in word_counts if word_counts[word] == 1]
    ```

---

#### **3.2.3 Practical Examples with Code**

**Example 1: Full Preprocessing Pipeline**

```python
import re
import nltk
import string
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "The NLP course starts soon! Visit http://example.com for details. Contact us at info@example.com 😊."

# 1. Lowercasing
text = text.lower()

# 2. Removing URLs and Emails
text = re.sub(r'http\S+|www.\S+', '', text)
text = re.sub(r'\S+@\S+', '', text)

# 3. Removing Emojis
emoji_pattern = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+", flags=re.UNICODE
)
text = emoji_pattern.sub(r'', text)

# 4. Removing Punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# 5. Tokenization
word_tokens = nltk.word_tokenize(text)

# 6. Removing Stopwords
filtered_tokens = [word for word in word_tokens if word not in stop_words]

# 7. Lemmatization
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("Original Text:", text)
print("Filtered Tokens:", filtered_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
```
