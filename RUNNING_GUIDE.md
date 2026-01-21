# Running the Application

This guide explains how to set up and run the full stack application (Flask Backend + Next.js Frontend).

## Prerequisites
- Python 3.9+
- Node.js 18+
- npm

## 1. Backend (Flask)

The backend is located in the `DATA-mining-project` directory.

### Setup
1. Navigate to the backend directory:
   ```bash
   cd DATA-mining-project
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   # Create venv if not exists
   python3 -m venv .venv

   # Activate (macOS/Linux)
   source .venv/bin/activate
   # Activate (Windows)
   # .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter errors, ensure you are in the virtual environment.*

### Running the Server
Start the Flask server:
```bash
python run.py
```
- The backend will start at **http://127.0.0.1:5001**
- It runs with CORS enabled to allow frontend connections.

## 2. Frontend (Next.js)

The frontend is located in the `data-processing-api` directory.

### Setup
1. Navigate to the frontend directory:
   ```bash
   cd data-processing-api
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

### Running the App
Start the development server:
```bash
npm run dev
```
- The app will likely start at **http://localhost:3000** (or 3001 if 3000 is busy).
- Open your browser to the URL shown in the terminal.

## Troubleshooting

### "Cannot find module" errors
If you see missing module errors in the frontend, try:
```bash
rm -rf .next
npm run dev
```

### "Failed to fetch" errors
This usually means the frontend cannot talk to the backend.
1. Ensure the Backend is running on port 5001.
2. Check `next.config.ts` has the correct rewrites (should be set up already).
3. Ensure no firewall is blocking `127.0.0.1:5001`.

### "AttributeError: 'Conv2D'..."
If you see deep learning errors in the backend console, ensure you are using the latest code which contains fixes for TensorFlow/Keras compatibility.
