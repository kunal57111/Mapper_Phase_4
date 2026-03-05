# Deploying Mapper to Vercel (Free Tier)

You can host the **frontend** on Vercel for free. The **backend** (FastAPI + MongoDB + embeddings) should run on a separate free host (e.g. Railway, Render) because of size and startup time.

## What runs where

| Part | Where | Notes |
|------|--------|------|
| **Frontend** (HTML/JS/CSS) | Vercel | Static + `/api/config` serverless |
| **Backend** (FastAPI, mapping, memory, LLM) | Railway / Render / etc. | Use a free tier and set its URL in Vercel |

## 1. Deploy backend elsewhere

1. Deploy the FastAPI app (this repo) to **Railway** or **Render** (both have free tiers).
2. Set env vars there: `MONGO_URI`, `MONGO_DB_NAME`, `OPENROUTER_API_KEY`, etc. (see `.env.example` or `backend/config.py`).
3. Note the public URL, e.g. `https://your-app.railway.app` or `https://your-app.onrender.com`.

## 2. Deploy frontend to Vercel

1. Push this repo to GitHub and import the project in [Vercel](https://vercel.com).
2. In Vercel project **Settings → Environment Variables**, add:
   - **`API_URL`** = your backend URL (e.g. `https://your-app.railway.app`) — no trailing slash.
3. Deploy. Vercel will:
   - Serve the `frontend/` folder as the site root.
   - Expose `/api/config`, which returns `{ "apiUrl": "<API_URL>" }` so the frontend knows where to call the backend.

## 3. CORS

The backend already allows all origins (`allow_origins=["*"]`). If you restrict CORS later, add your Vercel domain (e.g. `https://your-project.vercel.app`).

## 4. Local development

- **Frontend only**: Open `frontend/index.html` or run a static server; the app will use `http://127.0.0.1:8000` as the API base.
- **Full stack**: Run the backend with `uvicorn backend.main:app --reload` and use the frontend with the same API base (or set `API_URL` only in Vercel; locally the default is localhost).

## Summary

- **Yes, you can host on Vercel for free** by serving the frontend and a small config endpoint there.
- Set **`API_URL`** in Vercel to your backend URL so the frontend talks to your FastAPI app.
- Backend stays on Railway/Render (or another host) due to dependencies (MongoDB, SentenceTransformer, FAISS) and startup behavior.
