import React, { useEffect, useRef } from 'react';
import '../styles/components.css';

export default function VideoPreview({ file }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (file && videoRef.current) {
      const url = URL.createObjectURL(file);
      videoRef.current.src = url;

      return () => URL.revokeObjectURL(url);
    }
  }, [file]);

  if (!file) {
    return null;
  }

  return (
    <div className="video-preview">
      <h4>Video Preview</h4>
      <video
        ref={videoRef}
        controls
        width="100%"
        maxHeight="300px"
        style={{ borderRadius: '8px' }}
      />
    </div>
  );
}