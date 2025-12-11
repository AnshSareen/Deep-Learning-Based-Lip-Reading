import React from 'react';
import { FiVideo } from 'react-icons/fi';
import '../styles/components.css';

export default function Header() {
  return (
    <header className="header">
      <div className="header-container">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <FiVideo size={28} color="white" />
            </div>
            <div>
              <h1>LipRead AI</h1>
              <p>Transform lip movements into text</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}