import React, { useState, useEffect } from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Header from './components/Header';
import VideoUpload from './components/VideoUpload';
import ResultsDisplay from './components/ResultsDisplay';
import { videoAPI } from './services/api';
import './styles/App.css';

function App() {
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await videoAPI.healthCheck();
      setApiHealth(response.data);
    } catch (error) {
      console.error('API health check failed:', error);
      setApiHealth({ status: 'unhealthy' });
    }
  };

  const handleUploadSuccess = (data) => {
    setUploadedVideo(data);
  };

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        {apiHealth?.status === 'unhealthy' && (
          <div className="warning-banner">
            ⚠️ Backend server is not responding. Please start the server first.
          </div>
        )}

        <div className="container">
          {!uploadedVideo ? (
            <VideoUpload onUploadSuccess={handleUploadSuccess} />
          ) : (
            <ResultsDisplay
              analysisId={uploadedVideo.analysisId}
              filename={uploadedVideo.filename}
              directResult={uploadedVideo.directResult}
            />
          )}
        </div>
      </main>

      <ToastContainer
        position="bottom-right"
        autoClose={4000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </div>
  );
}

export default App;