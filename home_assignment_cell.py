# ============================================================
# HOME ASSIGNMENT: compare 2 blogs (10 articles each)
# Paste this whole cell into Colab AFTER the earlier cells have run
# (it needs: requests, BeautifulSoup, pandas, spacy `nlp`, TextBlob).
# ============================================================
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from textblob import TextBlob

# ---- 1. Choose your two blogs (you can change these) -------
# Each blog has a "feed" URL. A feed is a list of the latest posts,
# so we do not need to type article URLs by hand.
#
# NOTE: the old techncruncher blog now redirects to crunchhype.com, which
# Colab cannot connect to. So we use feeds that work. For each blog we give
# a list of feeds; the code uses the first one that works.
SOURCES = [
    {"name": "ai-news",
     "feeds": ["https://techcrunch.com/category/artificial-intelligence/feed/",
               "https://www.artificialintelligence-news.com/feed/"]},
    {"name": "machinelearningmastery",
     "feeds": ["https://machinelearningmastery.com/blog/feed/"]},
]
ARTICLES_PER_SOURCE = 10

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ---- 2. Get article URLs from a feed -----------------------
def get_urls_from_feed(feed_url, limit=15):
    """Return article links from an RSS or Atom feed."""
    res = requests.get(feed_url, headers=headers, timeout=15)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "xml")
    urls = []

    # Atom feed (Blogspot): <entry><link rel="alternate" href="...">
    for entry in soup.find_all("entry"):
        link = entry.find("link", rel="alternate")
        if link and link.get("href"):
            urls.append(link["href"])

    # RSS feed (WordPress): <item><link>...</link>
    for item in soup.find_all("item"):
        link = item.find("link")
        if link and link.get_text(strip=True):
            urls.append(link.get_text(strip=True))

    return urls[:limit]


# ---- 3. Scrape one article (works on many blog layouts) ----
def scrape_article(url):
    """Return (title, text) or None if the page has too little text."""
    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "lxml")

    # Title: try the page's <h1>, then <h3 class="post-title">, then <title>
    t = soup.find("h1") or soup.find("h3", class_="post-title") or soup.title
    title = t.get_text(strip=True) if t else url

    # Body: try known containers first, then fall back to <article> / <main>
    block = (soup.find("div", class_="post-body")
             or soup.find("article")
             or soup.find("main")
             or soup.body)
    if block is None:
        return None

    # Use <p> tags; if there are almost none (like the page that failed in
    # Exercise 1), use all the text in the block instead.
    text = " ".join(p.get_text(" ", strip=True) for p in block.find_all("p"))
    if len(text) < 300:
        text = block.get_text(" ", strip=True)

    return (title, text) if len(text) > 300 else None


# ---- 4. Scrape both sources politely ------------------------
rows = []
for src in SOURCES:
    kept = 0
    for feed in src["feeds"]:
        if kept >= ARTICLES_PER_SOURCE:
            break
        try:
            urls = get_urls_from_feed(feed, limit=20)
        except Exception as e:
            print(f"[ERROR] Could not read feed {feed}: {str(e)[:100]}")
            continue

        print(f"{src['name']}: found {len(urls)} links in {feed}")
        failures = 0
        for url in urls:
            if kept >= ARTICLES_PER_SOURCE:
                break
            if failures >= 3:   # site seems unreachable: try the next feed
                print("  Too many errors, moving on.")
                break
            try:
                result = scrape_article(url)
                if result is None:
                    print(f"  [SKIP] too little text: {url}")
                else:
                    title, text = result
                    rows.append({"source": src["name"], "url": url,
                                 "title": title, "text": text})
                    kept += 1
                    failures = 0
            except Exception as e:
                failures += 1
                print(f"  [ERROR] {url} -> {str(e)[:100]}")
            time.sleep(1)          # be polite to the server

    print(f"  kept {kept} articles from {src['name']}\n")

df_hw = pd.DataFrame(rows)
df_hw.to_csv("home_assignment_raw.csv", index=False)
print("Total scraped:", len(df_hw))
print(df_hw["source"].value_counts())


# ---- 5. Features: polarity and noun density ----------------
MAX_CHARS = 100_000  # keep spaCy fast and inside its length limit

def noun_density(text):
    doc = nlp(text[:MAX_CHARS])
    nouns = sum(1 for t in doc if t.pos_ == "NOUN")
    return nouns / len(doc) if len(doc) else 0.0

df_hw["polarity"] = df_hw["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df_hw["noun_density"] = df_hw["text"].apply(noun_density)
df_hw.to_csv("home_assignment_features.csv", index=False)


# ---- 6. Compare the two sources -----------------------------
summary = df_hw.groupby("source")[["polarity", "noun_density"]].agg(["mean", "std", "count"]).round(4)
print("\nSummary per source:")
display(summary)

means = df_hw.groupby("source")[["polarity", "noun_density"]].mean()
if len(means) == 2:
    a, b = means.index
    print(f"\nPolarity     : {a} = {means.loc[a,'polarity']:.4f}, "
          f"{b} = {means.loc[b,'polarity']:.4f}, "
          f"difference = {means.loc[a,'polarity'] - means.loc[b,'polarity']:+.4f}")
    print(f"Noun density : {a} = {means.loc[a,'noun_density']:.4f}, "
          f"{b} = {means.loc[b,'noun_density']:.4f}, "
          f"difference = {means.loc[a,'noun_density'] - means.loc[b,'noun_density']:+.4f}")

    # Optional: Welch t-test (tells us if the difference could be chance)
    from scipy import stats
    for col in ["polarity", "noun_density"]:
        g1 = df_hw.loc[df_hw.source == a, col]
        g2 = df_hw.loc[df_hw.source == b, col]
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        print(f"  t-test on {col}: p = {p_val:.4f}")


# ---- 7. Grouped bar chart ------------------------------------
plot_df = (df_hw.melt(id_vars="source", value_vars=["polarity", "noun_density"],
                      var_name="feature", value_name="value"))
plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x="feature", y="value", hue="source", errorbar="sd", capsize=0.1)
plt.title("Mean polarity and noun density by blog (error bars = std. dev.)")
plt.xlabel("Feature")
plt.ylabel("Mean value")
plt.tight_layout()
plt.savefig("home_assignment_chart.png", dpi=150)
plt.show()
