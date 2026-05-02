// components/StatsPanel.jsx
export default function StatsPanel({ stats }) {
  const items = [
    { label: "Total Tokens", value: stats.total_tokens, icon: "🔤" },
    { label: "Unique Words (Vocab)", value: stats.unique_words, icon: "📖" },
    { label: "Bigram Contexts", value: stats.bigram_contexts, icon: "2️⃣" },
    { label: "Trigram Contexts", value: stats.trigram_contexts, icon: "3️⃣" },
  ];

  return (
    <section className="card stats-card">
      <h2>📊 Knowledge Base Statistics</h2>
      <div className="stats-grid">
        {items.map((item) => (
          <div key={item.label} className="stat-item">
            <span className="stat-icon">{item.icon}</span>
            <span className="stat-value">{item.value.toLocaleString()}</span>
            <span className="stat-label">{item.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
