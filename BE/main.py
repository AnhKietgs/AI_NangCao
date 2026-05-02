# ============================================================
# main.py — FastAPI Entry Point
# Next-Word Prediction Engine (Markov Chain + NLP)
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.predict import PredictRequest, PredictResponse
from services.predictor import PredictorService

app = FastAPI(
    title="Next-Word Prediction Engine",
    description="Markov Chain / N-gram based next-word predictor",
    version="1.0.0"
)

# ── CORS: allow React dev server to communicate with FastAPI ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Instantiate the predictor service (loads & trains on startup) ──
predictor = PredictorService()


@app.get("/")
def root():
    return {"message": "Next-Word Prediction API is running ✅"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Inference endpoint.
    - Input  : user text (the "evidence" in our probabilistic model)
    - Output : top-k next words with their transition probabilities
    """
    results = predictor.predict(
        text=request.text,
        top_k=request.top_k,
        n=request.n
    )
    return PredictResponse(
        input_text=request.text,
        predictions=results,
        model_order=request.n
    )


@app.get("/stats")
def stats():
    """Return knowledge-base statistics for the report / demo."""
    return predictor.get_stats()
