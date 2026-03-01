"""
NewsBot — The Hindu AI Chatbot
Vercel-compatible Flask backend using Groq (free).
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from bs4 import BeautifulSoup
from datetime import datetime
import requests, time, os

# ── App ───────────────────────────────────────────────────────────────────────
import os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
app = Flask(__name__,
            template_folder=_os.path.join(_root, "templates"),
            static_folder=_os.path.join(_root, "static"))
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_9XnGgV0qxaTTNAwgT459WGdyb3FYks8Y8EMo97AJVDXDS8a3jn")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
CACHE_TTL    = 600  # 10 minutes

CATEGORIES = {
    "top":           "https://www.thehindu.com/",
    "national":      "https://www.thehindu.com/news/national/",
    "international": "https://www.thehindu.com/news/international/",
    "business":      "https://www.thehindu.com/business/",
    "sport":         "https://www.thehindu.com/sport/",
    "science":       "https://www.thehindu.com/sci-tech/science/",
    "technology":    "https://www.thehindu.com/sci-tech/technology/",
    "entertainment": "https://www.thehindu.com/entertainment/",
    "health":        "https://www.thehindu.com/sci-tech/health/",
    "education":     "https://www.thehindu.com/education/",
    "environment":   "https://www.thehindu.com/sci-tech/energy-and-environment/",
}

SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ── In-memory cache (resets per Vercel invocation — acceptable for news) ──────
_cache: dict[str, tuple[list, float]] = {}


# ── Scraper ───────────────────────────────────────────────────────────────────
def scrape(url: str, limit: int = 10) -> list[dict]:
    if url in _cache and time.time() < _cache[url][1]:
        return _cache[url][0]

    articles, seen = [], set()

    try:
        soup = BeautifulSoup(
            requests.get(url, headers=SCRAPE_HEADERS, timeout=12).text,
            "html.parser"
        )

        # Strategy 1: structured story cards
        for selector in ("div.story-card", "div.element", "article", "div.storylist-element"):
            for item in soup.select(selector):
                if len(articles) >= limit:
                    break
                hl = item.select_one("h1,h2,h3")
                if not hl:
                    continue
                headline = hl.get_text(strip=True)
                a_tag    = item.find("a", href=True)
                link     = a_tag["href"] if a_tag else ""
                if link and not link.startswith("http"):
                    link = "https://www.thehindu.com" + link
                if not link or link in seen or len(headline) < 10:
                    continue
                seen.add(link)
                summary_el = item.select_one("p.intro,p,.summary")
                pub_el     = item.select_one("time,.dateline,span.date")
                articles.append({
                    "headline":  headline,
                    "summary":   summary_el.get_text(strip=True)[:250] if summary_el else "",
                    "link":      link,
                    "published": pub_el.get_text(strip=True) if pub_el else "",
                })

        # Strategy 2: fallback — any /article links
        if len(articles) < 3:
            for a in soup.find_all("a", href=True):
                if len(articles) >= limit:
                    break
                href = a["href"]
                text = a.get_text(strip=True)
                if len(text) > 20 and "/article" in href and href not in seen:
                    full = href if href.startswith("http") else "https://www.thehindu.com" + href
                    seen.add(href)
                    articles.append({"headline": text, "summary": "", "link": full, "published": ""})

    except Exception as e:
        print(f"[Scraper] {url}: {e}")

    _cache[url] = (articles, time.time() + CACHE_TTL)
    return articles


def get_trending() -> list[dict]:
    try:
        soup  = BeautifulSoup(
            requests.get("https://www.thehindu.com/", headers=SCRAPE_HEADERS, timeout=12).text,
            "html.parser"
        )
        items = soup.select("div.most-popular li a, div.trending a, .popular-stories a")
        arts  = [
            {
                "headline":  a.get_text(strip=True),
                "link":      a["href"] if a["href"].startswith("http") else "https://www.thehindu.com" + a["href"],
                "summary":   "",
                "published": "",
            }
            for a in items[:6]
            if a.get_text(strip=True) and a.get("href")
        ]
        return arts or scrape("https://www.thehindu.com/", 6)
    except Exception:
        return scrape("https://www.thehindu.com/", 6)


# ── AI Layer ──────────────────────────────────────────────────────────────────
def build_context(category: str, query: str) -> str:
    articles = scrape(CATEGORIES.get(category, CATEGORIES["top"]))
    lines = [
        f"## The Hindu — {category.upper()}",
        f"## {datetime.now().strftime('%A, %d %B %Y %H:%M IST')}",
        "",
    ]
    for i, a in enumerate(articles, 1):
        lines.append(f"**{i}.** {a['headline']}")
        if a["summary"]:   lines.append(f"   {a['summary']}")
        if a["published"]: lines.append(f"   📅 {a['published']}")
        lines.append(f"   🔗 {a['link']}")
        lines.append("")

    if any(w in query.lower() for w in ("trend", "popular", "viral", "top", "most read", "hot")):
        trending = get_trending()
        if trending:
            lines += ["## Trending Now:"] + [f"• {t['headline']}" for t in trending] + [""]

    return "\n".join(lines)


def call_groq(system: str, messages: list[dict]) -> str:
    if not GROQ_API_KEY:
        return (
            "⚠️ **GROQ_API_KEY not set.**\n\n"
            "Add it as an environment variable:\n"
            "- **Vercel**: Project → Settings → Environment Variables → `GROQ_API_KEY`\n"
            "- **Local**: export `GROQ_API_KEY=your_key` or add to `.env`\n\n"
            "Get a free key (no credit card) at https://console.groq.com"
        )

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model":       GROQ_MODEL,
            "messages":    [{"role": "system", "content": system}] + messages,
            "max_tokens":  1024,
            "temperature": 0.6,
            "stream":      False,
        },
        timeout=30,
    )

    if resp.status_code == 401:
        return "❌ Invalid Groq API key. Check https://console.groq.com"
    if resp.status_code == 429:
        return "⚠️ Rate limit hit. Wait a moment and retry."
    if resp.status_code != 200:
        return f"❌ Groq error {resp.status_code}: {resp.text[:200]}"

    return resp.json()["choices"][0]["message"]["content"]


def ai_response(message: str, language: str, category: str, history: list) -> str:
    lang_rule = (
        "IMPORTANT: Respond entirely in Hindi (Devanagari script)."
        if language == "hi"
        else "Respond in clear, professional English."
    )
    system = f"""You are NewsBot — a smart AI news assistant with live articles from The Hindu.
{lang_rule}
TODAY: {datetime.now().strftime('%A, %d %B %Y')} | CATEGORY: {category.upper()}

{build_context(category, message)}

INSTRUCTIONS:
- Answer based on the articles above; cite URLs when helpful
- For unlisted topics, suggest thehindu.com directly
- Be conversational, neutral, and concise (3–5 paragraphs max)
- Use numbered lists for multi-story responses
{lang_rule}"""

    msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-6:]
        if m.get("role") in ("user", "assistant")
    ]
    msgs.append({"role": "user", "content": message})
    return call_groq(system, msgs)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    msg  = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        reply = ai_response(
            msg,
            data.get("language", "en"),
            data.get("category", "top"),
            data.get("history", []),
        )
        return jsonify({"reply": reply, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/<category>")
def get_news(category):
    url = CATEGORIES.get(category, CATEGORIES["top"])
    return jsonify({"articles": scrape(url, 8), "category": category})


@app.route("/api/trending")
def trending():
    return jsonify({"articles": get_trending()})


@app.route("/api/categories")
def categories():
    return jsonify({"categories": list(CATEGORIES.keys())})


@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "time":      datetime.now().isoformat(),
        "model":     GROQ_MODEL,
        "api_key":   "✅ Set" if GROQ_API_KEY else "❌ Missing",
    })


# ── Local dev entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  📰 NewsBot  |  {GROQ_MODEL}  |  http://localhost:5030\n")
    app.run(debug=True, host="0.0.0.0", port=5030)
