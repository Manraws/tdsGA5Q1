# RAG Colosseum API — Render deployment

This repository contains a small FastAPI retrieval API and the 50 passages from the supplied `knowledge_base.json`.

## Files

- `app.py` — FastAPI app with `POST /query`, `GET /health`, and `GET /`
- `passages.json` — passages with IDs 0–49 and their domain labels
- `knowledge_base.json` — original 50-string knowledge base, preserved as supplied
- `requirements.txt` — Python dependencies
- `render.yaml` — Render service configuration

## Deploy to Render

1. Create a new, empty GitHub repository.
2. Extract this ZIP file on your computer.
3. Upload **the contents of the extracted folder** to the root of the GitHub repository. Make sure `app.py`, `passages.json`, `requirements.txt`, and `render.yaml` are at the repository root (not hidden inside another folder).
4. In Render, choose **New → Web Service**.
5. Connect your GitHub repository. Render can read the `render.yaml` configuration.
6. If entering settings manually, use:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
7. Deploy. When the service is live, open `/health` and then `/docs` on your Render URL.

## API request

Send a POST request to `https://YOUR-RENDER-SERVICE.onrender.com/query`:

```json
{
  "question": "How does Raft handle a minority network partition?",
  "top_k": 3
}
```

The response has this shape:

```json
{
  "answer": "A passage-based answer.",
  "sources": [0, 1],
  "scores": [0.71, 0.42],
  "confidence": 0.71
}
```

The exact values depend on the question and retrieval results.

## Important notes

- This starter uses lightweight BM25 keyword retrieval. It does not download a neural embedding model, so deployment is smaller and startup is faster.
- Scores are normalized BM25 relevance values, not statistically calibrated probabilities.
- Answers are extractive: they are assembled from relevant sentences in the retrieved passages. This is a practical baseline, not a guarantee of a particular evaluation score. Test all evaluation probes before relying on it.
- Render Free web services spin down after 15 minutes without traffic; the first request after sleeping can take about a minute to wake up. If your evaluator enforces a strict 8-second limit on the very first request, the Free plan may not meet that requirement. See Render's free-instance documentation.
- Keep the knowledge base files in the repository root. The app reads `passages.json` at startup.
