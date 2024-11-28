import os
import requests
import xml.etree.ElementTree as ET

# Конфигурации
ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=all:nlp&start=0&max_results=5"
MD_FILE_PATH = "extra_material/research_papers.md"

def fetch_arxiv_articles():
    """Получение статей из arXiv API."""
    response = requests.get(ARXIV_API_URL)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        articles = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip().replace("\n", " ")
            articles.append({"title": title, "link": link, "summary": summary})
        return articles
    else:
        raise Exception(f"Failed to fetch data from arXiv: {response.status_code}")

def get_next_article_number(file_path):
    """Определить следующий номер статьи из существующего markdown-файла."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in reversed(lines):
            if line.startswith("["):
                try:
                    return int(line.split("]")[0][1:]) + 1
                except ValueError:
                    continue
    return 1

def update_markdown_file(articles, file_path):
    """Добавление новых статей в markdown-файл."""
    next_number = get_next_article_number(file_path)
    with open(file_path, "a", encoding="utf-8") as file:
        for article in articles:
            file.write(f"\n[{next_number}] <a href={article['link']}>{article['title']}</a>  \n")
            file.write(f"{article['summary']}\n")
            next_number += 1

if __name__ == "__main__":
    try:
        print("Fetching articles from arXiv...")
        articles = fetch_arxiv_articles()
        print(f"Fetched {len(articles)} articles. Updating the markdown file...")
        update_markdown_file(articles, MD_FILE_PATH)
        print(f"Markdown file updated successfully at {MD_FILE_PATH}")
    except Exception as e:
        print(f"Error: {e}")
