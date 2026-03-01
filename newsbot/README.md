# 📰 NewsBot — The Hindu AI Chatbot

AI news assistant powered by live scraping from The Hindu + Groq's free Llama 3.3 70B API.

## Project Structure

```
newsbot/
├── api/
│   └── index.py          # Flask app (Vercel entry point)
├── templates/
│   └── index.html        # Frontend UI
├── static/               # CSS/JS assets (if any)
├── vercel.json           # Vercel deployment config
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── .gitignore
```

---

## 🚀 Deploy to Vercel

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
gh repo create newsbot --public --push
```

### 2. Import to Vercel
1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repo
3. Framework: **Other**
4. Root directory: leave as `/`

### 3. Set Environment Variable
In Vercel: **Project → Settings → Environment Variables**
```
GROQ_API_KEY = your_key_here
```
Get a free key (no credit card) at https://console.groq.com

### 4. Deploy
Click **Deploy** — done! ✅

---

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run (load .env manually or use python-dotenv)
export GROQ_API_KEY=your_key_here
python api/index.py
```

Open http://localhost:5030

---

## 🔑 Get a Free Groq API Key

1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Click **API Keys → Create API Key**
4. Copy and set as `GROQ_API_KEY`

Free tier: **14,400 requests/day** with Llama 3.3 70B

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| POST | `/api/chat` | Chat with NewsBot |
| GET | `/api/news/<category>` | Fetch headlines by category |
| GET | `/api/trending` | Trending articles |
| GET | `/api/categories` | List all categories |
| GET | `/health` | Health check |
