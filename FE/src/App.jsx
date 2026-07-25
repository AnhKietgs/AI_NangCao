import { useState, useCallback } from "react";
import PredictionBar from "./components/PredictionBar";
import StatsPanel from "./components/StatsPanel";
import "./App.css";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [text, setText] = useState("");
  const [nOrder, setNOrder] = useState(2);
  const [topK, setTopK] = useState(5);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [stats, setStats] = useState(null);

  const handlePredict = useCallback(async () => {
    if (!text.trim()) return;
    const wordCount = text.trim().split(/\s+/).length;
  if (nOrder === 3 && wordCount < 2) {
    setError("Trigram (N=3) cần ít nhất 2 từ để dự đoán. Vui lòng nhập thêm từ.");
    return;
  }
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, top_k: topK, n: nOrder }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setPredictions(data.predictions);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [text, nOrder, topK]);

  const appendWord = (word) => {
    setText((prev) => (prev.trim() ? prev.trim() + " " + word : word));
    setPredictions([]);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };

  return (
    <div className="app-container">
      <div className="bg-glow"></div>
      <h1>
         Next-Word Engine <br />
      </h1>
      <header className="app-header">
        <div className="logo-section">
          <p className="subtitle">Probabilistic NLP & Markov Chains</p>
        </div>
        <button
          className="glass-card"
          style={{
            padding: "0.8rem 1.5rem",
            borderRadius: "12px",
            cursor: "pointer",
            color: "var(--primary)",
            fontWeight: "bold",
          }}
          onClick={async () => {
            try {
              const res = await fetch(`${API_BASE}/stats`);
              const data = await res.json();
              setStats(data);
            } catch {
              setError("API Offline");
            }
          }}
        >
          System Stats
        </button>
      </header>

      <main className="grid-layout">
        {/* Cột trái: Điều khiển */}
        <div className="input-column">
          <section className="glass-card">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "1rem",
              }}
            >
              <span
                style={{
                  color: "var(--secondary)",
                  fontWeight: "600",
                  fontSize: "0.9rem",
                }}
              >
                CONTEXT INPUT
              </span>
            
<div className="model-selector">
  <button
    className={nOrder === 2 ? "active" : ""}
    onClick={() => setNOrder(2)}
  >
    N=2
  </button>
  <button
    className={nOrder === 3 ? "active" : ""}
    onClick={() => setNOrder(3)}
  >
    N=3
  </button>
</div>
            </div>

            <textarea
              className="modern-textarea"
              placeholder="Enter your seed text here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKey}
            />

            <div className="range-control">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "0.5rem",
                }}
              >
                <span className="text-dim">Diversity (Top-K)</span>
                <span style={{ color: "var(--primary)", fontWeight: "bold" }}>
                  {topK}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>

            <div style={{ display: "flex", gap: "1rem", marginTop: "2rem" }}>
              <button
                className={`primary-btn ${loading ? "loading" : ""}`}
                onClick={handlePredict}
                disabled={loading || !text.trim()}
              >
                {loading ? " Processing..." : "Generate Next Word"}
              </button>
              <button
                className="glass-card"
                style={{
                  padding: "0 1.5rem",
                  borderRadius: "14px",
                  cursor: "pointer",
                }}
                onClick={() => {
                  setText("");
                  setPredictions([]);
                }}
              >
                Clear
              </button>
            </div>
            {error && (
              <div
                style={{
                  color: "#f87171",
                  marginTop: "1rem",
                  fontSize: "0.9rem",
                }}
              >
                 {error}
              </div>
            )}
          </section>

        </div>

        {/* Cột phải: Kết quả */}
        <div className="results-column">
          {predictions.length > 0 ? (
            <div className="fade-in">
              <section className="glass-card" style={{ marginBottom: "2rem" }}>
                <h2
                  style={{
                    marginBottom: "1.5rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                  }}
                >
                   Predictions
                </h2>
                <div className="prediction-grid">
                  {predictions.map((p, i) => (
                    <button
                      key={p.word}
                      className="prediction-chip"
                      onClick={() => appendWord(p.word)}
                      style={{ animationDelay: `${i * 0.1}s` }}
                    >
                      <span className="word">{p.word}</span>
                      <span className="prob">
                        {(p.probability * 100).toFixed(1)}%
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              <section className="glass-card">
                <h3 style={{ marginBottom: "1.5rem" }}> Distribution</h3>
                <div className="bar-container">
                  {predictions.map((p, i) => (
                    <PredictionBar key={p.word} prediction={p} rank={i} />
                  ))}
                </div>
              </section>
            </div>
          ) : (
            <div
              className="glass-card"
              style={{ textAlign: "center", padding: "4rem 2rem" }}
            >
              <div style={{ fontSize: "3rem", marginBottom: "1rem" }}></div>
              <p className="text-dim">
                Ready for inference. Type something to start.
              </p>
            </div>
          )}

          {stats && (
            <div style={{ marginTop: "2rem" }}>
              <StatsPanel stats={stats} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
