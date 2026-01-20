# Data Processing & Machine Learning Application

Complete guide for setting up and running the full-stack application.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 18.0 or higher
- **npm** or **yarn**: Latest version

## 🏗️ Project Structure

```
fd-projet/
├── DATA-mining-project/     # Backend (Flask API)
│   ├── app/                 # Application code
│   ├── uploads/             # Uploaded files
│   ├── static/              # Generated charts
│   └── requirements.txt     # Python dependencies
└── data-processing-api/     # Frontend (Next.js)
    ├── app/                 # Pages and routes
    ├── components/          # React components
    └── package.json         # Node dependencies
```

## �� Backend Setup (Flask API)

### 1. Navigate to Backend Directory
```bash
cd DATA-mining-project
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Backend Server
```bash
python run.py
```

Backend will run on: **http://localhost:5000**

### Backend API Endpoints

- `POST /api/upload` - Upload CSV file
- `POST /api/create-sample` - Create sample data
- `GET /api/preview` - Preview data
- `GET /api/statistics` - Get statistics
- `POST /api/process` - Process data
- `POST /api/ml/train` - Train ML model
- `POST /api/clustering/run` - Run clustering

## 🎨 Frontend Setup (Next.js)

### 1. Navigate to Frontend Directory
```bash
cd data-processing-api
```

### 2. Install Dependencies
```bash
npm install
# or
yarn install
```

### 3. Run Development Server
```bash
npm run dev
# or
yarn dev
```

Frontend will run on: **http://localhost:3000**

## 🔄 Complete Workflow

### 1. Start Both Servers

**Terminal 1 - Backend:**
```bash
cd DATA-mining-project
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run.py
```

**Terminal 2 - Frontend:**
```bash
cd data-processing-api
npm run dev
```

### 2. Access the Application

Open browser: **http://localhost:3000**

### 3. Using the Application

#### Data Processing Flow:
1. **Upload** - Upload CSV file (one time only)
2. **Preview** - View and select columns
3. **Statistics** - Analyze data
4. **Processing** - Clean and transform
5. **Visualization** - Create charts

#### Machine Learning Flow:
1. Navigate to **ML** page (uses uploaded data)
2. **Algorithm** - Select ML algorithm
3. **Configure** - Set parameters and run
4. **Results** - View metrics
5. **Compare** - Compare algorithms

#### Clustering Flow:
1. Navigate to **Clustering** page (uses uploaded data)
2. **Algorithm** - Select clustering method
3. **Elbow** - Find optimal clusters
4. **Dendrogram** - Hierarchical view
5. **Run** - Execute clustering
6. **Compare** - Compare methods

## 📦 Dependencies

### Backend (Python)
```txt
flask>=2.3.0
flask-cors>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Frontend (Node.js)
```json
{
  "next": "^15.1.5",
  "react": "^19.0.0",
  "recharts": "^2.15.0",
  "lucide-react": "^0.469.0",
  "sonner": "^1.7.3"
}
```

## 🔧 Troubleshooting

### Backend Issues

**Port 5000 already in use:**
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9
```

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill process
lsof -ti:3000 | xargs kill -9
# Or use different port
npm run dev -- -p 3001
```

**Module not found:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### CORS Issues

Backend already configured with CORS enabled for `http://localhost:3000`

### File Upload Issues

- Max file size: **16MB**
- Supported format: **CSV only**
- Ensure `uploads/` directory exists in backend

## 🧪 Testing

### Backend Tests
```bash
cd DATA-mining-project
python -m pytest
# or
python run_all_tests.py
```

### Frontend Tests
```bash
cd data-processing-api
npm run test
```

## 📝 Environment Variables

### Backend (.env - optional)
```bash
FLASK_ENV=development
FLASK_APP=run.py
MAX_UPLOAD_SIZE=16777216
```

### Frontend (.env.local - optional)
```bash
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## 🚀 Production Deployment

### Backend
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Frontend
```bash
npm run build
npm run start
```

## 📊 Features

### Data Processing
- ✅ CSV upload and preview
- ✅ Statistical analysis
- ✅ Data cleaning (missing values, duplicates)
- ✅ Feature scaling (StandardScaler, MinMaxScaler)
- ✅ Encoding (Label, One-Hot)
- ✅ Visualization (charts and graphs)

### Machine Learning
- ✅ Classification (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
- ✅ Regression (Linear, Ridge, Lasso)
- ✅ Model evaluation metrics
- ✅ Algorithm comparison
- ✅ Cross-validation

### Clustering
- ✅ K-Means clustering
- ✅ Hierarchical clustering
- ✅ DBSCAN
- ✅ Elbow method
- ✅ Dendrogram visualization
- ✅ Cluster metrics

## 🎯 Key Features

### Single Upload Workflow
Upload data **once** in Data Processing, then use it across:
- Data Processing page
- Machine Learning page
- Clustering page
- Comparison page

### Global State Management
Data is shared across all pages using React Context API.

## 📞 Support

For issues or questions, check:
- `FRONTEND_API_GUIDE.md` - API integration details
- `ML_IMPLEMENTATION_SUMMARY.md` - ML features
- `TESTING_GUIDE.md` - Testing information

## 📄 License

MIT License
