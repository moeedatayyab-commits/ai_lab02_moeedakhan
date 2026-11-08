# Lab 02 Report: Web Scraping, Feature Engineering and EDA

**Course:** Introduction to Artificial Intelligence (AI-101L)

---

## 1. What I did

I built a small pipeline:

**Blog URLs → requests + BeautifulSoup → pandas DataFrame → spaCy and TextBlob features → EDA plots**

I scraped blog articles, turned the text into numbers (features), and then looked at those numbers with plots. In the home assignment I compared two different blogs.

---

## 2. NumPy vs Python loop

I added up 10 million random numbers in two ways.

| Method | Sum | Time |
|---|---|---|
| NumPy (`np.sum`) | 4999262.42 | 0.00655 sec |
| Python `for` loop | 4999262.42 | 2.58943 sec |

**Result:** NumPy was about **395 times faster**. Both methods gave the same answer.

**Why:** A Python loop runs one step at a time, and Python has to check the type of every number. NumPy does the whole sum in one fast, compiled operation. So for big data, we should use NumPy instead of loops. (The exact speed-up changes from run to run and from machine to machine, but it is always hundreds of times.)

---

## 3. Web scraping

- I used **7 URLs**: the 5 original `techncruncher.blogspot.com` articles, plus 2 of my own (the two newest articles from the Machine Learning Mastery feed).
- Result: **Scraped 7 of 7 articles.**
- In my first run only 4 of 5 worked. The page `cj-affiliate-ultimate-guide-to.html` was skipped as `Too short`. After the fix in Exercise 1 (use all text of the content block when there are no `<p>` tags), it works too.
- The scraper uses a browser-like `User-Agent`, a timeout, a 1-second pause between pages, and `try/except`, so one bad page does not stop the whole run.
- I saved the raw data to `articles_raw.csv`, so the lab can run again without the internet.

---

## 4. Features I created

| # | Article | Tokens | Sentences | Entities | Nouns | Polarity | Subjectivity |
|---|---|---|---|---|---|---|---|
| 0 | Top 10 AI Tools That Will Transform Your Content Creation | 1188 | 47 | 24 | 321 | 0.229 | 0.584 |
| 1 | LimeWire AI Studio Review 2023 | 1723 | 65 | 103 | 420 | 0.219 | 0.574 |
| 2 | Top 10 AI Tools in 2023 That Will Make Your Life Easier | 1480 | 52 | 39 | 331 | 0.197 | 0.436 |
| 3 | Top 10 AI Content Generator & Writer Tools in 2022 | 1319 | 63 | 89 | 271 | 0.215 | 0.565 |
| 4 | Beginner Guide to CJ Affiliate | 1822 | 102 | 79 | 391 | 0.254 | 0.481 |
| 5 | How to Train Your Own Decision Model like Jev... (Machine Learning Mastery) | 1627 | 101 | 70 | 394 | 0.136 | 0.485 |
| 6 | Choosing an LLM Inference Optimization Technique... (Machine Learning Mastery) | 1098 | 57 | 22 | 259 | 0.125 | 0.426 |

I also made `title_length` (number of characters in the title).

---

## 5. EDA findings

**Important:** I only have **7 articles**. So everything below is a hint, not proof.

1. **Sentiment is mildly positive.** All polarity scores are between 0.12 and 0.25.
2. **The two technical articles are the least positive.** The two Machine Learning Mastery articles (technical tutorials) have the lowest polarity (0.136 and 0.125). The promotional "top 10 tools" articles are higher (0.20 to 0.25). This fits the style: promotional writing uses more positive words.
3. **Articles are fairly subjective.** Subjectivity is between 0.43 and 0.58.
4. **Longer articles have more entities and sentences.** Correlation of tokens with sentences is 0.72, and with entities 0.73. This is expected: more text means more of everything.
5. **Title length and entities** have a negative correlation of -0.64: in this sample, articles with longer titles have fewer entities. With 7 points this may be chance and I do not claim a real link.
6. **Polarity has weak links with the other features** (for example 0.39 with token count and about 0 with sentence count), so length does not explain sentiment.

---

## 6. TF-IDF interpretation

TF-IDF gives a high score to a word that is common in a document but not common everywhere. I removed English stop words and kept the top 20 terms. The matrix has shape (7 documents x 20 terms).

**Top terms:** `model`, `ai`, `content`, `cj`, `tool`, `creators`, `month`, `limewire`, `tools`, `features`, `plan`, `decision`, `creation`, `training`, `use`, `free`, `affiliate`, `ll`, `platform`, `make`.

**Meaning:**
- The words match the topics of the articles: AI tools for content creation (`content`, `creators`, `limewire`, `cj`, `affiliate`) and machine learning (`model`, `decision`, `training`).
- `month`, `plan` and `free` show that the promotional articles talk a lot about pricing.
- `ll` comes from contractions such as "we'll" and "you'll" that the tokenizer splits. It is noise and shows that the text needs cleaning.
- With only 7 documents, the IDF part is weak, so these scores are only a rough guide.

---

## 7. In-lab exercises

### Exercise 1: Why did one page fail?

I added two more URLs of my own: the two newest articles from the Machine Learning Mastery feed (`https://machinelearningmastery.com/blog/feed/`), picked automatically by the code. I also added a fallback to `<article>` / `<main>` for other blog layouts.

**Which page failed:** `https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html`. The scraper printed `[SKIP] Too short`.

**Why it failed (checked in the page HTML):**
I loaded the page again and checked it step by step:

| Check | Result |
|---|---|
| HTTP status | 200 (page loaded fine, no redirect) |
| Content `div` (`post-body entry-content`) found? | **Yes** |
| Number of `<p>` tags inside the div | **0** |
| Text length collected from `<p>` tags | **0** characters |
| Text length from the whole div | **9,350** characters |

So the selector **did match**. The problem is that the article has **no `<p>` tags at all**. The HTML inside the div starts with a `<table>` (an image with a caption) and the text is written directly inside the `<div>`, not in paragraphs. My code only collected text from `<p>` tags, so it got an empty string, and the "length > 100" quality check rejected the page as "too short".

**Fix:** if there are no (or very few) `<p>` tags, take all the text from the div instead:

```python
text = ' '.join(p.get_text(strip=True) for p in paragraphs)
if len(text) < 100:
    text = content_div.get_text(' ', strip=True)
```

After the fix, all 7 pages were scraped ("Scraped 7 of 7 articles").

**Lesson:** a scraper that depends on one HTML tag is fragile. Different posts on the same blog can have different HTML. A good scraper should have a fallback and should print why it skipped a page (as mine did).

### Exercise 2: New features
I added `num_verbs` and `avg_sentence_length` and included them in the pair plot.

| # | Verbs | Avg sentence length |
|---|---|---|
| 0 | 150 | 25.3 |
| 1 | 200 | 26.5 |
| 2 | 166 | 28.5 |
| 3 | 140 | 20.9 |
| 4 | 206 | 17.9 |
| 5 | 204 | 16.1 |
| 6 | 149 | 19.3 |

Article 2 has the longest sentences (28.5 tokens). Article 5 (a technical tutorial) has the shortest (16.1 tokens). The three newest articles (4, 5 and 6) have shorter sentences than the first four, so they use more, shorter sentences. The number of verbs grows with article length, as expected.

### Exercise 3: Why not report a KDE over a few points as a population distribution?
A KDE draws a smooth curve from the data. With only 5 to 7 points, the shape of the curve is decided mostly by the bandwidth setting and not by real data. It looks like a proper distribution, but it can be very wrong. One more article could change its shape a lot.

**What I would do instead:** collect many more articles (at least 30 to 50), show the raw points or a simple histogram, and report the mean with a range, saying clearly that the sample is small.

### Exercise 4: TF-IDF with and without stop words
With `stop_words='english'`, the top terms are real topic words: `model`, `ai`, `content`, `cj`, `tool`, `creators`.

With `stop_words=None`, the top terms are words like `the`, `and`, `to`, `for`, `of`, `it`. The 16 words that appear **only** when stop words are kept are: `and`, `as`, `can`, `for`, `in`, `is`, `it`, `of`, `on`, `that`, `the`, `this`, `to`, `with`, `you`, `your`.

**Explanation:** Stop words appear in every document, so they should get a low IDF score. But with only 7 documents, the IDF difference is small, and their very high counts (TF) still push them to the top. They tell us nothing about the topic, so removing them gives much more useful results.

---

## 8. Home assignment: comparing two blogs

**Question:** Do two blogs differ in average sentiment polarity and noun density?

**Data:** I scraped the 10 latest articles from each blog using their feeds (20 articles in total). I cleaned the text, then measured polarity (TextBlob) and noun density (nouns divided by total tokens, using spaCy).

- **Blog A, "ai-news":** TechCrunch, AI category (news articles)
- **Blog B, "machinelearningmastery":** Machine Learning Mastery (tutorials)

| Feature | Blog A (news) | Blog B (tutorials) | Difference (A - B) | Welch t-test p-value |
|---|---|---|---|---|
| Mean polarity | 0.1258 (std 0.0499) | 0.0958 (std 0.0396) | **+0.0300** | 0.1549 |
| Mean noun density | 0.1960 (std 0.0191) | 0.2460 (std 0.0235) | **-0.0500** | 0.0001 |

*(Chart: `home_assignment_chart.png`, a grouped bar chart with error bars showing standard deviation.)*

**Interpretation (one paragraph):**
Blog A was slightly more positive than Blog B (0.126 vs 0.096, a difference of +0.030), but this difference is small and the t-test gives p = 0.15, so with only 10 articles per blog it could easily be due to chance. The error bars also overlap a lot. The noun density difference is bigger: Blog B uses about 5 percentage points more nouns (24.6% vs 19.6%), and the p-value is 0.0001, so this difference is very unlikely to be random noise. This fits the type of writing: tutorials name many concepts, tools and terms, while news stories use more verbs and names. However, the result must be read with care. We have only 10 articles per blog, we took only the latest posts (not a random sample), and the two blogs differ in topic and style, not only in "source". So we can say the difference in noun density looks real for these samples, but we cannot say it holds for all articles on these blogs, and we cannot say the polarity difference exists at all.

---

## 9. Limits of this work

- Only 7 articles in the main lab and 20 in the home assignment, so results cannot be generalised.
- TextBlob is a simple, rule-based tool. It can miss sarcasm and context.
- The first 5 articles come from one blog, so they are similar in style.
- The home assignment used the latest posts of each blog (not a random sample), and the two blogs differ in topic and style, not only in source.
- Websites can change their HTML or move to a new domain (the original `techncruncher` blog now points to `crunchhype.com`), which can break a scraper. This is why I cached the raw data.
