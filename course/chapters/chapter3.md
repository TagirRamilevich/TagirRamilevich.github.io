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

