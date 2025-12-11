import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT = parseInt(import.meta.env.VITE_API_TIMEOUT || '30000');

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const videoAPI = {
  uploadVideo: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    return api.post('/api/videos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  analyzeVideo: async (analysisId) => {
    return api.post(`/api/videos/analyze/${analysisId}`);
  },

  getHistory: async (limit = 10, offset = 0) => {
    return api.get('/api/videos/history', {
      params: { limit, offset },
    });
  },

  getStatus: async (analysisId) => {
    return api.get(`/api/videos/status/${analysisId}`);
  },

  healthCheck: async () => {
    return api.get('/api/health/');
  },
};

export default api;