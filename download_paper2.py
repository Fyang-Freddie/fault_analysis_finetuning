import csv
import os
import re
import time

import requests


SAVE_DIR = "papers_failure_analysis2"
os.makedirs(SAVE_DIR, exist_ok=True)

EMAIL = "yf2028813437@outlook.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

KEYWORDS = [
    "nuclear power plant failure analysis",
    "nuclear power plant pipe failure analysis",
    "nuclear power plant stainless steel tube failure",
    "thermal power plant failure analysis",
    "boiler tube failure analysis",
    "power plant steam pipe failure analysis",
    "metallurgical failure analysis power plant",
    "failure analysis of boiler tube",
    "failure analysis of stainless steel pipe",
    "failure analysis SEM metallography hardness",
    "failure analysis chemical composition metallography",
    "failure analysis microstructure hardness",
    "核电 设备 失效分析",
    "核电 管道 失效分析",
    "火电 设备 失效分析",
    "火电 锅炉管 失效分析",
]


def search_article(keyword, per_page=10):
    """
    Search online papers and technical reports by keyword.
    """
    url = "https://api.openalex.org/works"
    params = {
        "search": keyword,
        "per-page": per_page,
        "filter": "is_oa:true",
        "mailto": EMAIL,
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print("Search failed:", keyword, e)
        return []

    papers = []
    for item in r.json().get("results", []):
        best = item.get("best_oa_location") or {}
        doi = item.get("doi") or ""
        if doi:
            doi = doi.replace("https://doi.org/", "")

        papers.append({
            "title": item.get("title") or "untitled",
            "year": item.get("publication_year") or "unknown",
            "doi": doi,
            "source": "OpenAlex",
            "paper_url": item.get("id") or "",
            "pdf_url": best.get("pdf_url") or "",
            "keyword": keyword,
        })

    return papers


def download_article(pdf_url):
    """
    Download PDF content by URL.
    """
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=40, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "").lower()
        if r.status_code == 200 and ("pdf" in ctype or r.content[:4] == b"%PDF"):
            return r.content
    except Exception as e:
        print("Download failed:", pdf_url, e)

    return None


def safe_name(text):
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180] or "untitled"


def main():
    all_papers = {}
    metadata_rows = []

    for keyword in KEYWORDS:
        print("\nSearching keyword:", keyword)
        results = search_article(keyword)
        print("Results:", len(results))

        for paper in results:
            key = paper.get("doi") or safe_name(paper.get("title", "").lower())
            if not key or key in all_papers:
                continue

            all_papers[key] = paper
            file_path = ""
            pdf_url = paper.get("pdf_url") or ""

            if pdf_url:
                filename = safe_name(f"{paper['year']}_{paper['title']}.pdf")
                file_path = os.path.join(SAVE_DIR, filename)

                if os.path.exists(file_path):
                    print("Skipped existing:", filename)
                else:
                    pdf = download_article(pdf_url)
                    if pdf:
                        with open(file_path, "wb") as f:
                            f.write(pdf)
                        print("Downloaded:", filename)

            paper["file_path"] = file_path
            metadata_rows.append(paper)

        time.sleep(8)

    csv_path = os.path.join(SAVE_DIR, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title", "year", "doi", "source",
                "paper_url", "pdf_url", "file_path", "keyword"
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nDone.")
    print("Total unique papers:", len(all_papers))
    print("Metadata saved to:", csv_path)


if __name__ == "__main__":
    main()
