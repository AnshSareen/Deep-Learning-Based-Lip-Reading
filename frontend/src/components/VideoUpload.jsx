import React, { useState, useRef, useEffect } from 'react';
import { FiUploadCloud, FiPlay, FiFilm } from 'react-icons/fi';
import { toast } from 'react-toastify';
import { videoAPI } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import '../styles/components.css';

const DEMO_VIDEOS = [
  { name: 'bbad3n.mpg', label: 'Demo 1' },
  { name: 'bbas1s.mpg', label: 'Demo 2' },
  { name: 'bwba7a.mpg', label: 'Demo 3' },
  { name: 'prap6p.mpg', label: 'Demo 4' },
  { name: 'sgbp4n.mpg', label: 'Demo 5' },
  { name: 'sria7s.mpg', label: 'Demo 6' },
];

export default function VideoUpload({ onUploadSuccess }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showDemos, setShowDemos] = useState(false);
  const [selectedDemo, setSelectedDemo] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setSelectedDemo(null);
      }
    }
  };

  const validateFile = (file) => {
    const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/mpeg'];
    const maxSize = 100 * 1024 * 1024;

    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.mpg')) {
      toast.error('Please upload a video file (MP4, AVI, MOV, MKV, MPG)');
      return false;
    }

    if (file.size > maxSize) {
      toast.error('File size exceeds 100MB limit');
      return false;
    }

    return true;
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
        setSelectedDemo(null);
      }
    }
  };

  const handleDemoSelect = (demo) => {
    setSelectedDemo(demo);
    setSelectedFile(null);
  };

  const handleUpload = async () => {
    if (!selectedFile && !selectedDemo) {
      toast.error('Please select a video file or demo');
      return;
    }

    setIsLoading(true);
    try {
      if (selectedDemo) {
        // Use demo video endpoint
        toast.loading('Analyzing demo video...', { toastId: 'analyzing' });
        const response = await fetch(`http://localhost:8000/api/videos/demo/${selectedDemo.name}`, {
          method: 'POST',
        });
        const data = await response.json();

        toast.dismiss('analyzing');

        if (response.ok) {
          toast.success('Analysis complete!');
          onUploadSuccess({
            analysisId: data.analysis_id,
            filename: selectedDemo.name,
            directResult: data,
          });
        } else {
          toast.error(data.detail || 'Analysis failed');
        }
      } else {
        // Upload file
        toast.loading('Uploading video...', { toastId: 'uploading' });
        const response = await videoAPI.uploadVideo(selectedFile);

        toast.dismiss('uploading');
        toast.success('Video uploaded!');

        onUploadSuccess({
          analysisId: response.data.analysis_id,
          filename: selectedFile.name,
        });
      }

      setSelectedFile(null);
      setSelectedDemo(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      toast.dismiss();
      const errorMsg = error.response?.data?.detail || 'Operation failed. Please try again.';
      toast.error(errorMsg);
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-section-title">
        <h2>Analyze Lip Movements</h2>
        <p>Upload a video with visible lips to transcribe speech</p>
      </div>

      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {selectedFile ? (
          <div className="file-selected">
            <FiPlay size={48} className="icon" />
            <p className="filename">{selectedFile.name}</p>
            <p className="filesize">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
        ) : selectedDemo ? (
          <div className="file-selected">
            <FiFilm size={48} className="icon" />
            <p className="filename">{selectedDemo.name}</p>
            <p className="filesize">Demo Video</p>
          </div>
        ) : (
          <div className="file-prompt">
            <FiUploadCloud size={56} className="icon" />
            <p className="title">Drop your video here</p>
            <p className="subtitle">or click to browse files</p>
            <p className="formats">MP4, AVI, MOV, MKV, MPG • Max 100MB</p>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,.mpg"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
      </div>

      <div className="divider">or try a demo</div>

      <button
        className="demo-button"
        onClick={() => setShowDemos(!showDemos)}
        disabled={isLoading}
      >
        <FiFilm style={{ marginRight: 8 }} />
        {showDemos ? 'Hide Demo Videos' : 'Try Demo Videos'}
      </button>

      {showDemos && (
        <div className="demo-videos-section">
          <div className="demo-videos-grid">
            {DEMO_VIDEOS.map((demo) => (
              <div
                key={demo.name}
                className={`demo-video-card ${selectedDemo?.name === demo.name ? 'selected' : ''}`}
                onClick={() => handleDemoSelect(demo)}
              >
                <div className="video-icon">🎬</div>
                <div className="video-name">{demo.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(selectedFile || selectedDemo) && (
        <button
          className="analyze-button"
          onClick={handleUpload}
          disabled={isLoading}
        >
          {isLoading ? <LoadingSpinner /> : '✨ Analyze Video'}
        </button>
      )}

      {/* How it works */}
      <div className="how-it-works">
        <div className="how-it-works-step">
          <div className="step-number">1</div>
          <div className="step-text">Upload Video</div>
        </div>
        <div className="how-it-works-step">
          <div className="step-number">2</div>
          <div className="step-text">Detect Lips</div>
        </div>
        <div className="how-it-works-step">
          <div className="step-number">3</div>
          <div className="step-text">Transcribe</div>
        </div>
      </div>
    </div>
  );
}