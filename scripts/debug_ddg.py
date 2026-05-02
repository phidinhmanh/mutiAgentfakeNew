import requests
import bs4
import re

def debug_ddg():
    query = "Việt Nam nâng cấp quan hệ Hoa Kỳ"
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://duckduckgo.com/",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    try:
        session = requests.Session()
        # First hit the home page to get cookies
        session.get("https://duckduckgo.com/", headers=headers, timeout=10)

        resp = session.get(url, headers=headers, timeout=20)
        print(f"Status Code: {resp.status_code}")
        print(f"Content Length: {len(resp.text)}")

        soup = bs4.BeautifulSoup(resp.text, "html.parser")

        # Try different CSS selectors
        selectors_to_try = [
            (".result", "CSS: .result"),
            (".result__a", "CSS: .result__a"),
            (".result__snippet", "CSS: .result__snippet"),
            (".links_main", "CSS: .links_main"),
            ("article", "tag: article"),
            (".web-result", "CSS: .web-result"),
            (".nrn-react-div", "CSS: .nrn-react-div"),
        ]

        for selector, name in selectors_to_try:
            elements = soup.select(selector)
            if elements:
                print(f"\n{name}: found {len(elements)} elements")
                for i, el in enumerate(elements[:3]):
                    text = el.get_text(strip=True)[:150]
                    print(f"  [{i}] {text}")

        # Look for any link patterns
        links = soup.find_all("a", href=True)
        result_links = [a for a in links if "uddg" in a.get("href", "") or (a.get("href", "").startswith("http") and "duckduckgo" not in a.get("href", ""))]
        print(f"\nResult links (external or uddg): {len(result_links)}")
        for i, a in enumerate(result_links[:5]):
            print(f"  [{i}] href={a.get('href', '')[:100]} text={a.get_text(strip=True)[:80]}")

        # Check for bot detection signals
        if "captcha" in resp.text.lower() or "blocked" in resp.text.lower():
            print("\n[POTENTIAL BOT DETECTION]")
        if len(resp.text) < 10000:
            print("\n[SHORT RESPONSE - possibly bot detection]")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_ddg()