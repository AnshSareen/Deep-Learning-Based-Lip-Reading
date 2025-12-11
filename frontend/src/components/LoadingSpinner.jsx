import React from 'react';
import '../styles/components.css';

export default function LoadingSpinner() {
  return (
    <div className="spinner">
      <div className="spinner-inner"></div>
      <div className="spinner-inner"></div>
      <div className="spinner-inner"></div>
    </div>
  );
}