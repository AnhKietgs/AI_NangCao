// components/PredictionBar.jsx
export default function PredictionBar({ prediction, rank }) {
  const { word, probability, count } = prediction;
  const pct = (probability * 100).toFixed(2);
  const colors = [
    "#4ade80", "#34d399", "#2dd4bf", "#38bdf8",
    "#818cf8", "#a78bfa", "#f472b6", "#fb923c",
    "#facc15", "#a3e635",
  ];
  const color = colors[rank % colors.length];

  return (
    <div className="bar-row">
      <span className="bar-word">{word}</span>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{ width: `${Math.min(pct * 10, 100)}%`, background: color }}
        />
      </div>
      <span className="bar-pct">{pct}%</span>
      <span className="bar-count">({count}×)</span>
    </div>
  );
}
