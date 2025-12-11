import React, { useEffect, useState } from 'react';
import { FiCheckCircle, FiAlertCircle, FiCopy, FiRefreshCw } from 'react-icons/fi';
import { toast } from 'react-toastify';
import { videoAPI } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import '../styles/components.css';

export default function ResultsDisplay({ analysisId, filename, directResult }) {
  const [result, setResult] = useState(directResult || null);
  const [isLoading, setIsLoading] = useState(!directResult);
  const [error, setError] = useState(null);

  useEffect(() => {
    const analyzeVideo = async () => {
      if (directResult) return; // Already have result

      try {
        setIsLoading(true);
        const response = await videoAPI.analyzeVideo(analysisId);
        setResult(response.data);
        toast.success('Analysis complete!');
      } catch (err) {
        setError(err.response?.data?.detail || 'Analysis failed');
        toast.error('Analysis failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };

    if (analysisId && !directResult) {
      analyzeVideo();
    }
  }, [analysisId, directResult]);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  if (isLoading) {
    return (
      <div className="results-container loading">
        <LoadingSpinner />
        <p>Analyzing your video...</p>
        <div className="processing-steps">
          <div className="processing-step active">
            <span className="step-dot"></span>
            <span>Extracting frames</span>
          </div>
          <div className="processing-step">
            <span className="step-dot"></span>
            <span>Detecting lips</span>
          </div>
          <div className="processing-step">
            <span className="step-dot"></span>
            <span>Transcribing</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-container error">
        <FiAlertCircle size={56} className="icon" />
        <h3>Analysis Failed</h3>
        <p>{error}</p>
        <button className="new-analysis-button" onClick={() => window.location.reload()}>
          <FiRefreshCw style={{ marginRight: 8 }} />
          Try Again
        </button>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  // Format error rates as percentages
  const werPercent = result.word_error_rate != null ? (result.word_error_rate * 100).toFixed(1) : null;
  const cerPercent = result.char_error_rate != null ? (result.char_error_rate * 100).toFixed(1) : null;
  const hasMetrics = werPercent != null && cerPercent != null;

  return (
    <div className="results-container success">
      <div className="results-header">
        <FiCheckCircle size={56} className="icon" />
        <h2>Analysis Complete!</h2>
      </div>

      {/* Main Predicted Text */}
      <div className="predicted-text-feature">
        <div className="predicted-text-label">Predicted Speech</div>
        <div className="predicted-text-value">
          "{result.predicted_text || 'No speech detected'}"
        </div>
        <button
          className="copy-button"
          onClick={() => copyToClipboard(result.predicted_text)}
          style={{ marginTop: 16 }}
        >
          <FiCopy size={16} />
          Copy Text
        </button>
      </div>

      {/* Ground Truth if available */}
      {result.ground_truth && (
        <div className="ground-truth-section">
          <div className="predicted-text-label">Ground Truth</div>
          <div className="ground-truth-value">
            "{result.ground_truth}"
          </div>
          {result.is_exact_match && (
            <div className="exact-match-badge">✅ Exact Match</div>
          )}
        </div>
      )}

      {/* Stats Grid */}
      <div className="results-grid">
        {/* WER Card */}
        {hasMetrics ? (
          <div className="result-card">
            <h4>Word Error Rate (WER)</h4>
            <div className="error-rate-display">
              <div className="error-rate-bar">
                <div
                  className="error-rate-fill"
                  style={{
                    width: `${Math.min(parseFloat(werPercent), 100)}%`,
                    backgroundColor: parseFloat(werPercent) <= 20 ? '#22c55e' : parseFloat(werPercent) <= 50 ? '#eab308' : '#ef4444'
                  }}
                />
              </div>
              <p className="stat-value">{werPercent}%</p>
            </div>
          </div>
        ) : (
          <div className="result-card">
            <h4>Word Error Rate</h4>
            <p className="stat-value muted">No ground truth</p>
          </div>
        )}

        {/* CER Card */}
        {hasMetrics ? (
          <div className="result-card">
            <h4>Character Error Rate (CER)</h4>
            <div className="error-rate-display">
              <div className="error-rate-bar">
                <div
                  className="error-rate-fill"
                  style={{
                    width: `${Math.min(parseFloat(cerPercent), 100)}%`,
                    backgroundColor: parseFloat(cerPercent) <= 20 ? '#22c55e' : parseFloat(cerPercent) <= 50 ? '#eab308' : '#ef4444'
                  }}
                />
              </div>
              <p className="stat-value">{cerPercent}%</p>
            </div>
          </div>
        ) : (
          <div className="result-card">
            <h4>Character Error Rate</h4>
            <p className="stat-value muted">No ground truth</p>
          </div>
        )}

        <div className="result-card">
          <h4>Video File</h4>
          <p className="filename-text">{filename}</p>
        </div>

        <div className="result-card">
          <h4>Frames Processed</h4>
          <p className="stat-value">{result.frames_detected}</p>
        </div>

        <div className="result-card">
          <h4>Processing Time</h4>
          <p className="stat-value">{result.processing_time.toFixed(2)}s</p>
        </div>
      </div>

      <button className="new-analysis-button" onClick={() => window.location.reload()}>
        <FiRefreshCw style={{ marginRight: 8 }} />
        Analyze Another Video
      </button>
    </div>
  );
}