import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [reviewText, setReviewText] = useState('');
  const [rating, setRating] = useState(3);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [translation, setTranslation] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');
  const [modelReady, setModelReady] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const res = await axios.get('/api/status');
      setModelReady(res.data.loaded);
    } catch (err) {
      console.error('Backend not available:', err);
    }
  };

  const analyzeReview = async () => {
    if (!reviewText.trim()) return;
    setLoading(true);
    setActiveTab('analysis');
    try {
      const res = await axios.post('/api/predict', {
        text: reviewText,
        star_rating: rating,
        threshold: 0.5,
      });
      setAnalysis(res.data);
    } catch (err) {
      alert('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const translateReview = async () => {
    if (!reviewText.trim()) return;
    setLoading(true);
    setActiveTab('translation');
    try {
      const res = await axios.post('/api/translate', { text: reviewText });
      setTranslation(res.data);
    } catch (err) {
      alert('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      analyzeReview();
    }
  };

  const getSentiment = (sentiment) => {
    if (sentiment === 'positive') return { label: 'Positive', cls: 'pos' };
    if (sentiment === 'negative') return { label: 'Negative', cls: 'neg' };
    return { label: 'Neutral', cls: 'neu' };
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-mark">ع</div>
            <div className="logo-meta">
              <span className="logo-title">Franco-Arabic ABSA</span>
              <span className="logo-sub">Aspect-Based Sentiment</span>
            </div>
          </div>
          <div className={`status-pill ${modelReady ? 'ready' : 'pending'}`}>
            <span className="status-dot" />
            <span>{modelReady ? 'Model ready' : 'Connecting…'}</span>
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="layout">
        {/* Input panel */}
        <section className="panel panel-input">
          <p className="panel-label">Review input</p>

          <div className="field">
            <label className="field-label" htmlFor="review-text">
              Text
              <span className="field-hint">Arabic · English · Franco-Arabic</span>
            </label>
            <textarea
              id="review-text"
              ref={textareaRef}
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={"El akl kaan 7elw bas el khidma bt2a5er…\nالاكل كان حلو جدا لكن الخدمة بطيئة\nThe food was great but service was slow"}
              rows={6}
            />
          </div>

          <div className="field">
            <label className="field-label">Star rating</label>
            <div className="rating-row">
              {[1, 2, 3, 4, 5].map((r) => (
                <button
                  key={r}
                  className={`star-btn ${r <= rating ? 'active' : ''}`}
                  onClick={() => setRating(r)}
                  aria-label={`${r} star${r > 1 ? 's' : ''}`}
                >
                  <StarIcon filled={r <= rating} />
                </button>
              ))}
              <span className="rating-label">{rating} / 5</span>
            </div>
          </div>

          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={analyzeReview}
              disabled={loading || !reviewText.trim()}
            >
              {loading && activeTab === 'analysis' ? (
                <><Spinner /> Analyzing…</>
              ) : (
                'Analyze sentiment'
              )}
            </button>
            <button
              className="btn btn-ghost"
              onClick={translateReview}
              disabled={loading || !reviewText.trim()}
            >
              {loading && activeTab === 'translation' ? (
                <><Spinner /> Translating…</>
              ) : (
                'Translate to Arabic'
              )}
            </button>
          </div>

          <p className="shortcut-hint">
            <kbd>⌘</kbd><kbd>Enter</kbd> to analyze
          </p>
        </section>

        {/* Results panel */}
        <section className="panel panel-results">
          <div className="results-header">
            <p className="panel-label">Results</p>
            <div className="tab-bar">
              <button
                className={`tab ${activeTab === 'analysis' ? 'active' : ''}`}
                onClick={() => setActiveTab('analysis')}
              >
                Sentiment
              </button>
              <button
                className={`tab ${activeTab === 'translation' ? 'active' : ''}`}
                onClick={() => setActiveTab('translation')}
              >
                Translation
              </button>
            </div>
          </div>

          <div className="results-body">
            {activeTab === 'analysis' && (
              <>
                {!analysis && !loading && (
                  <EmptyState icon="chart" text="Enter a review and click Analyze" />
                )}
                {loading && activeTab === 'analysis' && (
                  <LoadingState text="Analyzing sentiment…" />
                )}
                {analysis && !loading && (
                  <AnalysisResults analysis={analysis} getSentiment={getSentiment} />
                )}
              </>
            )}

            {activeTab === 'translation' && (
              <>
                {!translation && !loading && (
                  <EmptyState icon="translate" text="Enter text and click Translate" />
                )}
                {loading && activeTab === 'translation' && (
                  <LoadingState text="Translating…" />
                )}
                {translation && !loading && (
                  <TranslationResults translation={translation} />
                )}
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────── */

function AnalysisResults({ analysis, getSentiment }) {
  const hasAspects =
    analysis.aspects &&
    analysis.aspects.length > 0 &&
    analysis.aspects[0] !== 'none';

  if (!hasAspects) {
    return <EmptyState icon="mood" text="No aspects detected — try a different review." />;
  }

  return (
    <div className="analysis-wrap">
      <div className="aspects-list">
        {analysis.aspects.map((asp) => {
          const sent = getSentiment(analysis.aspect_sentiments?.[asp]);
          const prob = analysis.aspect_probs?.[asp] ?? 0;
          const pct = (prob * 100).toFixed(1);
          return (
            <div key={asp} className="aspect-card">
              <div className="aspect-top">
                <span className="aspect-name">{asp}</span>
                <span className={`badge badge-${sent.cls}`}>{sent.label}</span>
              </div>
              <div className="conf-bar">
                <div
                  className={`conf-fill fill-${sent.cls}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="conf-footer">
                <span>Confidence</span>
                <span className="conf-pct">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="meta-grid">
        <MetaCell label="Language">
          {analysis.detected_lang || '—'}
          {analysis.was_franco && <span className="tag">Franco-Arabic</span>}
        </MetaCell>
        <MetaCell label="Rating">
          {analysis.star_rating ?? '—'} / 5
        </MetaCell>
        {analysis.clean && (
          <MetaCell label="Cleaned input" wide>
            <span className="mono">{analysis.clean}</span>
          </MetaCell>
        )}
      </div>
    </div>
  );
}

function TranslationResults({ translation }) {
  return (
    <div className="translation-wrap">
      <div className="t-block">
        <div className="t-label">Original</div>
        <div className="t-text">{translation.original}</div>
      </div>
      <div className="t-block">
        <div className="t-label">Arabic script</div>
        <div className="t-text t-arabic" dir="rtl">{translation.translated}</div>
      </div>
      {translation.detected_dialect && (
        <div className="meta-grid">
          <MetaCell label="Dialect">{translation.detected_dialect}</MetaCell>
        </div>
      )}
    </div>
  );
}

function MetaCell({ label, children, wide }) {
  return (
    <div className={`meta-cell ${wide ? 'meta-wide' : ''}`}>
      <p className="meta-key">{label}</p>
      <p className="meta-val">{children}</p>
    </div>
  );
}

function EmptyState({ icon, text }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        {icon === 'chart' && '◈'}
        {icon === 'translate' && '⇄'}
        {icon === 'mood' && '◻'}
      </div>
      <p>{text}</p>
    </div>
  );
}

function LoadingState({ text }) {
  return (
    <div className="loading-state">
      <div className="loading-ring" />
      <p>{text}</p>
    </div>
  );
}

function Spinner() {
  return <span className="btn-spinner" aria-hidden="true" />;
}

function StarIcon({ filled }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

export default App;