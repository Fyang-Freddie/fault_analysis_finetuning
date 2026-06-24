import os
import re
import csv
import time
import requests
from urllib.parse import urlparse

SAVE_DIR = "papers_failure_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

EMAIL = "yf2028813437@outlook.com"  # Unpaywall 要求填写邮箱

QUERIES = [
    "failure analysis stainless steel pipe stress corrosion cracking SEM metallography",
    "failure analysis power plant pipe cracking SEM metallography",
    "failure analysis nuclear power plant stainless steel tube SCC",
    "failure analysis heat exchanger tube cracking SEM metallography",
    "failure analysis valve leakage stress corrosion cracking",
    "failure analysis boiler tube power plant SEM",
    "failure analysis pump shaft fracture SEM",
    "failure analysis turbine blade cracking metallography",
    "failure analysis steam pipe cracking creep",
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def safe_name(text):
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:180]

def is_pdf_response(r):
    ctype = r.headers.get("Content-Type", "").lower()
    return "pdf" in ctype or r.content[:4] == b"%PDF"

def download_pdf(pdf_url, filename):
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=40, allow_redirects=True)
        if r.status_code == 200 and is_pdf_response(r):
            path = os.path.join(SAVE_DIR, filename)
            if os.path.exists(path):
                return path
            with open(path, "wb") as f:
                f.write(r.content)
            print("Downloaded:", filename)
            return path
    except Exception as e:
        print("Download failed:", pdf_url, e)
    return None

def search_semantic_scholar(query, limit=50):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,authors,doi,url,openAccessPdf,abstract"
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        papers = []
        for p in r.json().get("data", []):
            papers.append({
                "source": "Semantic Scholar",
                "title": p.get("title"),
                "year": p.get("year"),
                "doi": p.get("doi"),
                "url": p.get("url"),
                "abstract": p.get("abstract"),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            })
        return papers
    except Exception as e:
        print("Semantic Scholar error:", e)
        return []

def search_openalex(query, per_page=50):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "filter": "is_oa:true",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        papers = []
        for w in r.json().get("results", []):
            doi = w.get("doi")
            if doi:
                doi = doi.replace("https://doi.org/", "")
            best = w.get("best_oa_location") or {}
            papers.append({
                "source": "OpenAlex",
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "doi": doi,
                "url": w.get("id"),
                "abstract": "",
                "pdf_url": best.get("pdf_url"),
            })
        return papers
    except Exception as e:
        print("OpenAlex error:", e)
        return []

def find_pdf_unpaywall(doi):
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": EMAIL}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf")
    except Exception:
        return None

def normalize_doi(doi):
    if not doi:
        return ""
    return doi.lower().replace("https://doi.org/", "").strip()

def paper_key(p):
    doi = normalize_doi(p.get("doi"))
    if doi:
        return doi
    return safe_name((p.get("title") or "").lower())

all_papers = {}
metadata_rows = []

for query in QUERIES:
    print("\nSearching:", query)

    results = []
    results.extend(search_semantic_scholar(query))
    time.sleep(1)
    results.extend(search_openalex(query))
    time.sleep(1)

    for p in results:
        key = paper_key(p)
        if not key or key in all_papers:
            continue
        all_papers[key] = p

        title = p.get("title") or "untitled"
        year = p.get("year") or "unknown"
        doi = p.get("doi") or ""

        pdf_url = p.get("pdf_url")

        if not pdf_url and doi:
            pdf_url = find_pdf_unpaywall(doi)
            time.sleep(1)

        file_path = ""
        if pdf_url:
            filename = safe_name(f"{year}_{title}.pdf")
            file_path = download_pdf(pdf_url, filename) or ""

        metadata_rows.append({
            "title": title,
            "year": year,
            "doi": doi,
            "source": p.get("source"),
            "paper_url": p.get("url"),
            "pdf_url": pdf_url or "",
            "file_path": file_path,
            "abstract": p.get("abstract") or "",
        })

    time.sleep(2)

csv_path = os.path.join(SAVE_DIR, "metadata.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "title", "year", "doi", "source",
            "paper_url", "pdf_url", "file_path", "abstract"
        ]
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print("\nDone.")
print("Total unique papers:", len(all_papers))
print("Metadata saved to:", csv_path)