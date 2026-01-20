# Data Processing & Machine Learning Platform

A full-stack application for data processing, machine learning, and clustering analysis with a modern web interface.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **Node.js 18+** installed
- **npm** or **yarn** package manager

### Installation & Running

#### 1️⃣ Backend Setup (Flask API)

Open a terminal and run:

```bash
# Navigate to backend directory
cd DATA-mining-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Run the backend server
python run.py
```

✅ Backend will run on **http://localhost:5000**

---

#### 2️⃣ Frontend Setup (Next.js)

Open a **new terminal** and run:

```bash
# Navigate to frontend directory
cd data-processing-api

# Install Node.js dependencies
npm install

# Run the development server
npm run dev
```

✅ Frontend will run on **http://localhost:3000**

---

#### 3️⃣ Access the Application

Open your browser and go to: **http://localhost:3000**

---

## 📁 Project Structure

```
fd-projet/
├── DATA-mining-project/          # Backend (Flask API)
│   ├── app/                      # Application code
│   │   ├── routes/               # API endpoints
│   │   └── utils/                # Utility functions
│   ├── uploads/                  # Uploaded CSV files
│   ├── static/                   # Generated charts & visualizations
│   ├── requirements.txt          # Python dependencies
│   └── run.py                    # Backend entry point
│
└── data-processing-api/          # Frontend (Next.js)
    ├── app/                      # Next.js pages
    │   ├── page.tsx              # Data Processing page
    │   ├── ml/                   # Machine Learning page
    │   └── clustering/           # Clustering page
    ├── components/               # React components
    ├── context/                  # Global state management
    ├── package.json              # Node dependencies
    └── next.config.ts            # Next.js configuration
```

---

## 🛠️ Dependencies

### Backend Dependencies (Python)

```
Flask==2.3.3                    # Web framework
pandas==2.0.3                   # Data manipulation
numpy==1.24.3                   # Numerical computing
scikit-learn==1.3.0             # Machine learning
matplotlib==3.7.1               # Plotting
seaborn==0.12.2                 # Statistical visualization
scipy==1.11.1                   # Scientific computing
flask-cors==4.0.0               # Cross-origin requests
```

### Frontend Dependencies (Node.js)

```json
{
  "next": "^15.1.5", // React framework
  "react": "^19.0.0", // UI library
  "recharts": "^2.15.0", // Charts
  "lucide-react": "^0.469.0", // Icons
  "tailwindcss": "^3.4.1" // Styling
}
```

---

## 📖 How to Use the Application

### 1. Data Processing (Home Page)

1. **Upload CSV** - Upload your dataset (one-time upload)
2. **Preview Data** - View and select columns to keep
3. **Statistics** - Analyze data with descriptive statistics
4. **Process Data** - Clean data (handle missing values, duplicates)
5. **Visualize** - Generate charts (histograms, box plots, scatter plots)

### 2. Machine Learning Page

1. Navigate to **Machine Learning** tab
2. **Select Algorithm** - Choose from:
   - Classification: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
   - Regression: Linear, Ridge, Lasso
3. **Configure Parameters** - Set test size, target column, features
4. **Train Model** - Run the algorithm
5. **View Results** - See metrics, confusion matrix, feature importance
6. **Compare** - Compare multiple algorithms

### 3. Clustering Page

1. Navigate to **Clustering** tab
2. **Select Algorithm** - K-Means, Hierarchical, or DBSCAN
3. **Elbow Method** - Find optimal number of clusters
4. **Dendrogram** - Visualize hierarchical relationships
5. **Run Clustering** - Execute clustering analysis
6. **Compare** - Compare different clustering methods

---

## 🔧 API Endpoints

### Backend API (Port 5000)

| Method | Endpoint                     | Description                         |
| ------ | ---------------------------- | ----------------------------------- |
| POST   | `/api/upload`                | Upload CSV file                     |
| POST   | `/api/create-sample`         | Create sample dataset               |
| GET    | `/api/preview`               | Preview uploaded data               |
| GET    | `/api/statistics`            | Get statistical summary             |
| POST   | `/api/process`               | Process data (clean, scale, encode) |
| POST   | `/api/visualize`             | Generate visualizations             |
| POST   | `/api/ml/train`              | Train ML model                      |
| GET    | `/api/ml/results`            | Get ML results                      |
| POST   | `/api/clustering/elbow`      | Generate elbow plot                 |
| POST   | `/api/clustering/dendrogram` | Generate dendrogram                 |
| POST   | `/api/clustering/run`        | Run clustering analysis             |

---

## 🧪 Testing

### Run Backend Tests

```bash
cd DATA-mining-project
source venv/bin/activate  # Activate virtual environment first
python run_all_tests.py
```

### Test Coverage

- Data processing logic
- Machine learning algorithms
- Clustering methods
- API endpoints

---

## ⚠️ Troubleshooting

### Backend Issues

**Port 5000 already in use:**

```bash
# macOS/Linux
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Module not found errors:**

```bash
pip install -r requirements.txt --upgrade
```

**Virtual environment not activating:**

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Issues

**Port 3000 already in use:**

```bash
# macOS/Linux
lsof -ti:3000 | xargs kill -9

# Or use different port
npm run dev -- -p 3001
```

**Dependencies not installing:**

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Build errors:**

```bash
# Clear Next.js cache
rm -rf .next
npm run dev
```

### CORS Errors

The backend is configured to accept requests from `http://localhost:3000`. If you change the frontend port, update the CORS configuration in `DATA-mining-project/run.py`.

---

## 📊 Features

### ✨ Data Processing

- ✅ CSV file upload and preview
- ✅ Column selection and filtering
- ✅ Statistical analysis (mean, median, std, etc.)
- ✅ Missing value handling (drop, fill mean/median/mode)
- ✅ Duplicate detection and removal
- ✅ Feature scaling (StandardScaler, MinMaxScaler)
- ✅ Encoding (Label Encoding, One-Hot Encoding)
- ✅ Data visualization (histograms, box plots, scatter plots)

### 🤖 Machine Learning

- ✅ Multiple algorithms (Classification & Regression)
- ✅ Hyperparameter configuration
- ✅ Train/test split
- ✅ Model evaluation metrics
- ✅ Confusion matrix
- ✅ Feature importance
- ✅ Algorithm comparison
- ✅ Cross-validation

### 🔍 Clustering

- ✅ K-Means clustering
- ✅ Hierarchical clustering (Agglomerative)
- ✅ DBSCAN
- ✅ Elbow method for optimal K
- ✅ Dendrogram visualization
- ✅ Silhouette score
- ✅ Cluster visualization
- ✅ Algorithm comparison

### 🎯 Key Highlights

- **Single Upload Workflow** - Upload once, use everywhere
- **Global State Management** - Data shared across all pages
- **Real-time Visualization** - Interactive charts and graphs
- **Responsive Design** - Works on desktop and mobile
- **Modern UI** - Built with Tailwind CSS and shadcn/ui

---

## 🚀 Production Deployment

### Backend (Flask)

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Frontend (Next.js)

```bash
# Build for production
npm run build

# Start production server
npm run start
```

---

## 📝 Environment Variables (Optional)

### Backend `.env`

```bash
FLASK_ENV=development
FLASK_APP=run.py
MAX_UPLOAD_SIZE=16777216
```

### Frontend `.env.local`

```bash
NEXT_PUBLIC_API_URL=http://localhost:5000
```

---

## 📚 Additional Documentation

- `SETUP_GUIDE.md` - Detailed setup instructions
- `FRONTEND_API_GUIDE.md` - API integration guide
- `ML_IMPLEMENTATION_SUMMARY.md` - ML implementation details
- `TESTING_GUIDE.md` - Testing documentation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## 📄 License

MIT License - feel free to use this project for learning and development.

---

## 💡 Tips

- **Always run both servers** (backend and frontend) for the application to work
- **Upload data first** in the Data Processing page before using ML or Clustering
- **Keep the terminal windows open** while using the application
- **Check browser console** for any frontend errors
- **Check backend terminal** for API errors

---

**Enjoy analyzing your data! 📊🚀**
