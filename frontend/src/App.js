import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [NP3TOT, setNP3TOT] = useState(0);
  const [UPSIT, setUPSIT] = useState(0.0);
  const [COGCHG, setCOGCHG] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError(null);
    setProcessingTime(null);
    setLoading(true);

    try {
      // Create FormData object for prediction with file
      const formData = new FormData();
      formData.append("file", file);
      formData.append("NP3TOT", NP3TOT);
      formData.append("UPSIT_PRCNTGE", UPSIT);
      formData.append("COGCHG", COGCHG);
      
      // Set timeout for the request - 3 minutes
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout
      
      console.log("Submitting prediction request...");
      const startTime = Date.now();
      
      // Use the external API endpoint
      // In development, this will use the proxy from package.json
      // In production, use the REACT_APP_API_URL environment variable
      const API_URL = process.env.NODE_ENV === 'production' 
        ? process.env.REACT_APP_API_URL || 'https://parkinsons-prediction-api.onrender.com'
        : '';
        
      // The backend endpoint is at /api with the predict function at /
      const response = await axios.post(`${API_URL}/api`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      const clientProcessingTime = (Date.now() - startTime) / 1000;
      console.log(`Request completed in ${clientProcessingTime} seconds`);
      
      setResult(response.data);
      if (response.data.processing_time_seconds) {
        setProcessingTime(response.data.processing_time_seconds);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      if (err.name === 'AbortError' || err.code === 'ERR_CANCELED') {
        setError("Request timed out. The prediction is taking too long to process.");
      } else if (err.response) {
        setError(`Error: ${err.response.data.detail || err.response.data.error || 'Server error occurred'}`);
      } else if (err.request) {
        setError("No response received from server. Please check your connection.");
      } else {
        setError(`Error: ${err.message || 'Unknown error occurred'}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>🧠 Parkinson's Risk Prediction</h1>
      
      <form onSubmit={handleSubmit}>
        <div>
          <label>Upload DATSCAN File (.dcm or .nii.gz): </label>
          <input 
            type="file" 
            accept=".dcm,.nii,.nii.gz" 
            onChange={(e) => setFile(e.target.files[0])} 
            required 
          />
        </div>
        <div>
          <label>NP3TOT (Motor Symptoms Score): </label>
          <input 
            type="number" 
            min="0" 
            value={NP3TOT} 
            onChange={(e) => setNP3TOT(Number(e.target.value))} 
            required 
          />
        </div>
        <div>
          <label>UPSIT_PRCNTGE (Smell Test Score): </label>
          <input 
            type="number" 
            min="0" 
            step="0.01" 
            value={UPSIT} 
            onChange={(e) => setUPSIT(Number(e.target.value))} 
            required 
          />
        </div>
        <div>
          <label>COGCHG (Cognitive Change): </label>
          <select 
            value={COGCHG} 
            onChange={(e) => setCOGCHG(Number(e.target.value))}
          >
            <option value={0}>No</option>
            <option value={1}>Yes</option>
          </select>
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Processing... Please wait" : "Predict"}
        </button>
      </form>

      {loading && (
        <div className="loading">
          <p>Processing your brain scan... This may take a minute.</p>
          <div className="spinner"></div>
          <p className="loading-tip">The first prediction may take longer as the system initializes.</p>
        </div>
      )}

      {result && (
        <div className="result">
          <h2>Prediction Result</h2>
          <p><strong>Right Putamen:</strong> {result.right_putamen?.toFixed(4)}</p>
          <p><strong>Left Putamen:</strong> {result.left_putamen?.toFixed(4)}</p>
          <p><strong>Right Caudate:</strong> {result.right_caudate?.toFixed(4)}</p>
          <p><strong>Left Caudate:</strong> {result.left_caudate?.toFixed(4)}</p>
          <p><strong>Risk Percentage:</strong> {result.risk_percent?.toFixed(2) || "N/A"}%</p>
          <p className="risk-status"><strong>Status:</strong> {result.risk_status}</p>
          {processingTime && (
            <p className="processing-time">Processing time: {processingTime.toFixed(2)} seconds</p>
          )}
        </div>
      )}

      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;
