# Lip Reading Frontend - Windows Setup Guide

## Prerequisites

- **Node.js 16+** (Download from https://nodejs.org/)
- **npm** (comes with Node.js)
- Backend API running on port 8000

## Installation Steps

### Step 1: Navigate to Frontend Folder

Open **Command Prompt** or **PowerShell**:

```cmd
cd C:\Projects\lipreading_model_deployment\frontend
```

### Step 2: Install Dependencies

```cmd
npm install
```

**Installation time:** 2-3 minutes

### Step 3: Configure API Endpoint

The `.env` file is already configured. Verify the settings:

```env
VITE_API_BASE_URL=http://localhost:8000
```

If your backend is running on a different port or host, update this value.

### Step 4: Start Development Server

```cmd
npm run dev
```

**Expected output:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: http://192.168.x.x:5173/
```

The app will automatically open in your browser at http://localhost:5173

### Step 5: Test the Application

1. Make sure the backend is running at http://localhost:8000
2. Open http://localhost:5173 in your browser
3. Upload a video file
4. Click "Analyze" to process

## Building for Production

### Create Production Build

```cmd
npm run build
```

This creates an optimized build in the `dist/` folder.

### Preview Production Build

```cmd
npm run preview
```

### Deploy Production Build

The `dist/` folder can be deployed to:
- **IIS** (Internet Information Services)
- **Nginx**
- **Apache**
- **Any static file hosting**

#### Example: Deploy to IIS

1. Copy the `dist/` folder contents to `C:\inetpub\wwwroot\lipreading\`
2. Create a new website in IIS Manager
3. Point the physical path to the folder
4. Set binding to port 80 or 443 (HTTPS)

## Configuration

### Change API URL

Edit `.env` file:

```env
# For local development
VITE_API_BASE_URL=http://localhost:8000

# For production
VITE_API_BASE_URL=https://api.yourdomain.com
```

After changing `.env`, restart the dev server:
```cmd
npm run dev
```

### Change Port

Default port is 5173. To change:

**Windows Command Prompt:**
```cmd
set PORT=3000 && npm run dev
```

**Windows PowerShell:**
```powershell
$env:PORT=3000; npm run dev
```

## Troubleshooting

### Port Already in Use

**Error:** `Port 5173 is already in use`

**Solution 1:** Kill the process using the port
```cmd
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

**Solution 2:** Use a different port
```cmd
set PORT=3000 && npm run dev
```

### Cannot Connect to Backend

**Error:** `Network Error` or `CORS Error`

**Solution:**
1. Verify backend is running: http://localhost:8000/api/health/
2. Check `.env` has correct URL
3. Verify CORS settings in backend allow frontend origin

### Module Not Found

**Error:** `Cannot find module 'X'`

**Solution:**
```cmd
rm -rf node_modules package-lock.json
npm install
```

### Build Fails

**Error:** Build errors during `npm run build`

**Solution:**
```cmd
npm run lint
npm run build
```

Fix any linting errors and rebuild.

## Scripts Reference

| Command | Description |
|---------|-------------|
| `npm install` | Install dependencies |
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run ESLint |

## Features

- ✅ Video upload with drag & drop
- ✅ Real-time processing status
- ✅ Prediction results display
- ✅ Confidence score visualization
- ✅ Responsive design
- ✅ Error handling
- ✅ Loading indicators

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool & dev server
- **Axios** - HTTP client
- **React Icons** - Icon library
- **React Toastify** - Notifications
- **Framer Motion** - Animations

## Directory Structure

```
frontend/
├── src/
│   ├── components/      # React components
│   ├── services/        # API services
│   ├── utils/           # Utility functions
│   ├── App.jsx          # Main component
│   └── main.jsx         # Entry point
├── public/              # Static assets
├── .env                 # Environment variables
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
└── index.html           # HTML template
```

## Production Checklist

Before deploying:

- [ ] Backend API is accessible
- [ ] `.env` has production API URL
- [ ] Run `npm run build` successfully
- [ ] Test production build with `npm run preview`
- [ ] Configure web server (IIS/Nginx)
- [ ] Enable HTTPS
- [ ] Set up proper CORS
- [ ] Add error monitoring
- [ ] Configure caching headers

## Support

For issues:
- Check browser console (F12)
- Check network tab for API calls
- Verify backend is running
- Check `.env` configuration
