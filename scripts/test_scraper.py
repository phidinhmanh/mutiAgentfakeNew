from trust_agents.rag.web_search import search_ddg_urls
from trust_agents.rag.content_extractor import extract_content_batch
import json

def test_scraper():
    query = "Việt Nam Hoa Kỳ nâng cấp quan hệ"
    print(f"--- Testing for query: {query} ---")

    # 1. Search for URLs
    print("\n[Step 1] Searching DuckDuckGo for URLs...")
    results = search_ddg_urls(query, num_results=3)
    if not results:
        print("FAIL: No URLs found")
        return

    urls = [r["url"] for r in results]
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")

    # 2. Extract content
    print("\n[Step 2] Extracting content from URLs (Markdown)...")
    content_results = extract_content_batch(urls, query=query, max_results=2)

    if not content_results:
        print("FAIL: No content extracted")
        return

    for i, res in enumerate(content_results, 1):
        print(f"\n--- Result {i} ({res['source']}) ---")
        print(f"Title: {res['title']}")
        print(f"Content Length: {len(res['content'])} chars")
        print(f"Content (snippet):\n{res['content'][:300]}...")

if __name__ == "__main__":
    test_scraper()
