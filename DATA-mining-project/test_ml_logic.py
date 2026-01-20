import unittest
import pandas as pd
import numpy as np
import sys
import os
import requests
import tempfile

# Ensure we can import from app
sys.path.append(os.getcwd())

from app.routes.ml import prepare_data

# Base URL for the API
API_URL = "http://localhost:5001"

class TestMLLogic(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Check if server is running before running tests."""
        try:
            response = requests.get(f"{API_URL}/")
            print(f"✅ Server is running at {API_URL}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"❌ Server is not running at {API_URL}. Please start the server first with 'python run.py'")
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_classification_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [5, 6, 7, 8, 9, 10, 11, 12],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        self.sample_regression_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            'target': [10.2, 20.5, 30.1, 40.8, 50.3, 60.7]
        })
        
    def _upload_data(self, df, filename='test_ml_data.csv'):
        """Helper method to upload data via API."""
        # Save to temporary file
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        df.to_csv(temp_path, index=False)
        
        # Upload via API
        with open(temp_path, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = requests.post(f"{API_URL}/upload-data", files=files)
        
        # Clean up temp file
        os.remove(temp_path)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        return data.get('path')
    
    # ===== PREPARE_DATA TESTS =====
    
    def test_prepare_data_float_labels_for_classification(self):
        """Test that float labels (like 0.0, 1.0) are converted to integers for classification."""
        data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0.0, 1.0, 0.0, 1.0]
        }
        df = pd.DataFrame(data)
        
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        
        # Check types
        self.assertTrue(np.issubdtype(y_train.dtype, np.integer), 
                       f"Expected integer dtype, got {y_train.dtype}")
        
        # Check values
        unique_vals = np.unique(np.concatenate([y_train, y_test]))
        self.assertTrue(np.array_equal(unique_vals, [0, 1]), 
                       f"Expected [0, 1], got {unique_vals}")

    def test_prepare_data_mixed_labels_for_classification(self):
        """Test that mixed type labels or messy floats work."""
        data = {
            'feature1': [1, 2, 3, 4],
            'target': [0.0, 1.0, 2.0, 0.0] 
        }
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        self.assertTrue(np.issubdtype(y_train.dtype, np.integer))

    def test_prepare_data_string_labels_classification(self):
        """Test that string labels are properly encoded for classification."""
        data = {
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [2, 3, 4, 5, 6, 7],
            'target': ['cat', 'dog', 'cat', 'dog', 'cat', 'dog']
        }
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        
        # Should be encoded as integers
        self.assertTrue(np.issubdtype(y_train.dtype, np.integer))
        # Should have 2 unique classes
        all_labels = np.concatenate([y_train, y_test])
        self.assertEqual(len(np.unique(all_labels)), 2)

    def test_prepare_data_categorical_features(self):
        """Test that categorical features are properly encoded."""
        data = {
            'feature1': ['A', 'B', 'A', 'B', 'A', 'B'],
            'feature2': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        
        # All features should be numeric after encoding
        self.assertTrue(np.issubdtype(X_train.iloc[:, 0].dtype, np.integer))

    def test_prepare_data_handles_nan_values(self):
        """Test that NaN values are properly dropped."""
        data = {
            'feature1': [1, 2, np.nan, 4, 5, 6],
            'feature2': [2, 3, 4, np.nan, 6, 7],
            'target': [0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        
        # Check that total samples is less than original (NaN rows dropped)
        total_samples = len(X_train) + len(X_test)
        self.assertLess(total_samples, len(df))
        
        # Check no NaN values remain
        self.assertFalse(X_train.isnull().any().any())
        self.assertFalse(X_test.isnull().any().any())

    def test_prepare_data_test_size_ratio(self):
        """Test that test_size parameter correctly splits data."""
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data, 
            'target', 
            test_size=0.25,
            is_classification=True
        )
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        # Should be approximately 0.25 (allowing small variance due to rounding)
        self.assertAlmostEqual(test_ratio, 0.25, delta=0.05)

    def test_prepare_data_regression_numeric_target(self):
        """Test that regression preserves numeric target values."""
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_regression_data,
            'target',
            is_classification=False
        )
        
        # Target should remain float for regression
        self.assertTrue(np.issubdtype(y_train.dtype, np.floating) or 
                       np.issubdtype(y_train.dtype, np.integer))

    def test_prepare_data_shape_consistency(self):
        """Test that shapes are consistent across train/test splits."""
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        # Feature dimensions should match
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        
        # Sample counts should match between X and y
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    # ===== API ENDPOINT TESTS =====
    
    def test_knn_api_endpoint(self):
        """Test KNN API endpoint."""
        print("\n🔍 Testing KNN API endpoint")
        path = self._upload_data(self.sample_classification_data)
        
        response = requests.post(f"{API_URL}/ml/knn", json={
            'path': path,
            'target': 'target',
            'max_k': 5
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('best_k', data['metrics'])
        self.assertIn('accuracy', data['metrics'])
        print(f"✅ KNN Test Passed - Accuracy: {data['metrics']['accuracy']:.4f}")
        
    def test_naive_bayes_api_endpoint(self):
        """Test Naive Bayes API endpoint."""
        print("\n🔍 Testing Naive Bayes API endpoint")
        path = self._upload_data(self.sample_classification_data)
        
        response = requests.post(f"{API_URL}/ml/naive-bayes", json={
            'path': path,
            'target': 'target'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('accuracy', data['metrics'])
        print(f"✅ Naive Bayes Test Passed - Accuracy: {data['metrics']['accuracy']:.4f}")
        
    def test_decision_tree_api_endpoint(self):
        """Test Decision Tree API endpoint."""
        print("\n🔍 Testing Decision Tree API endpoint")
        path = self._upload_data(self.sample_classification_data)
        
        response = requests.post(f"{API_URL}/ml/decision-tree", json={
            'path': path,
            'target': 'target',
            'algorithm_type': 'cart'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        print(f"✅ Decision Tree Test Passed - Accuracy: {data['metrics']['accuracy']:.4f}")
        
    def test_linear_regression_api_endpoint(self):
        """Test Linear Regression API endpoint."""
        print("\n🔍 Testing Linear Regression API endpoint")
        path = self._upload_data(self.sample_regression_data)
        
        response = requests.post(f"{API_URL}/ml/linear-regression", json={
            'path': path,
            'target': 'target'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('mse', data['metrics'])
        self.assertIn('r2_score', data['metrics'])
        print(f"✅ Linear Regression Test Passed - R²: {data['metrics']['r2_score']:.4f}")
        
    def test_neural_network_api_endpoint(self):
        """Test Neural Network API endpoint."""
        print("\n🔍 Testing Neural Network API endpoint")
        path = self._upload_data(self.sample_classification_data)
        
        response = requests.post(f"{API_URL}/ml/neural-network", json={
            'path': path,
            'target': 'target',
            'hidden_layers': [10, 5]
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('accuracy', data['metrics'])
        print(f"✅ Neural Network Test Passed - Accuracy: {data['metrics']['accuracy']:.4f}")
    
    def test_ml_comparison_endpoint(self):
        """Test ML comparison endpoint."""
        print("\n🔍 Testing ML Comparison endpoint")
        
        # Run multiple algorithms first
        path = self._upload_data(self.sample_classification_data)
        
        # Run KNN
        requests.post(f"{API_URL}/ml/knn", json={
            'path': path,
            'target': 'target',
            'max_k': 3
        })
        
        # Run Naive Bayes
        requests.post(f"{API_URL}/ml/naive-bayes", json={
            'path': path,
            'target': 'target'
        })
        
        # Get comparison
        response = requests.get(f"{API_URL}/ml/comparison")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('knn', data)
        self.assertIn('naive_bayes', data)
        print(f"✅ ML Comparison Test Passed - {len(data)} algorithms compared")

    # ===== KNN TESTS =====
    
    def test_knn_k_range_validation(self):
        """Test that KNN runs with valid k range."""
        from sklearn.neighbors import KNeighborsClassifier
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data, 
            'target', 
            is_classification=True
        )
        
        # Test with k=3
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        
        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_knn_predictions_shape(self):
        """Test that KNN predictions have correct shape."""
        from sklearn.neighbors import KNeighborsClassifier
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Predictions shape should match test labels
        self.assertEqual(len(y_pred), len(y_test))

    # ===== NAIVE BAYES TESTS =====
    
    def test_naive_bayes_probabilities(self):
        """Test that Naive Bayes produces valid probability predictions."""
        from sklearn.naive_bayes import GaussianNB
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        probabilities = gnb.predict_proba(X_test)
        
        # Each row should sum to approximately 1
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(X_test)))

    def test_naive_bayes_class_predictions(self):
        """Test that Naive Bayes predictions are valid classes."""
        from sklearn.naive_bayes import GaussianNB
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        
        # All predictions should be in the set of training classes
        unique_train = np.unique(y_train)
        self.assertTrue(np.all(np.isin(y_pred, unique_train)))

    # ===== DECISION TREE TESTS =====
    
    def test_decision_tree_gini_criterion(self):
        """Test Decision Tree with Gini criterion (CART)."""
        from sklearn.tree import DecisionTreeClassifier
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        dt = DecisionTreeClassifier(criterion='gini', random_state=42)
        dt.fit(X_train, y_train)
        
        # Should have feature_importances_ attribute
        self.assertTrue(hasattr(dt, 'feature_importances_'))
        self.assertEqual(len(dt.feature_importances_), X_train.shape[1])

    def test_decision_tree_entropy_criterion(self):
        """Test Decision Tree with Entropy criterion (ID3/C4.5)."""
        from sklearn.tree import DecisionTreeClassifier
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
        dt.fit(X_train, y_train)
        accuracy = dt.score(X_test, y_test)
        
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_decision_tree_confusion_matrix(self):
        """Test that confusion matrix has correct dimensions."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import confusion_matrix
        
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        n_classes = len(np.unique(y_train))
        
        # Confusion matrix should be square with dimensions = number of classes
        self.assertEqual(cm.shape, (n_classes, n_classes))

    # ===== LINEAR REGRESSION TESTS =====
    
    def test_linear_regression_coefficients(self):
        """Test that Linear Regression produces coefficients."""
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_regression_data,
            'target',
            is_classification=False
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Should have coefficients for each feature
        self.assertEqual(len(lr.coef_), X_train.shape[1])
        # Should have an intercept
        self.assertIsNotNone(lr.intercept_)

    def test_linear_regression_predictions_numeric(self):
        """Test that Linear Regression predictions are numeric."""
        from sklearn.linear_model import LinearRegression
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_regression_data,
            'target',
            is_classification=False
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        # Predictions should be numeric (float)
        self.assertTrue(np.issubdtype(y_pred.dtype, np.floating))
        self.assertEqual(len(y_pred), len(y_test))

    def test_linear_regression_metrics(self):
        """Test that regression metrics are calculated correctly."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_regression_data,
            'target',
            is_classification=False
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MSE should be non-negative
        self.assertGreaterEqual(mse, 0.0)
        # R2 can be negative but typically between -inf and 1
        self.assertLessEqual(r2, 1.0)

    # ===== NEURAL NETWORK TESTS =====
    
    def test_neural_network_hidden_layers_parsing(self):
        """Test parsing of hidden layer configurations."""
        test_cases = [
            (100, (100,)),
            ("(100,)", (100,)),
            ("100, 50", (100, 50)),
            ([100, 50], (100, 50)),
        ]
        
        for input_val, expected in test_cases:
            # Test the parsing logic from your neural network route
            if isinstance(input_val, int):
                hidden_layers = (input_val,)
            elif isinstance(input_val, str):
                sanitized = input_val.strip('()[] ')
                if ',' in sanitized:
                    hidden_layers = tuple(int(x.strip()) for x in sanitized.split(',') if x.strip())
                else:
                    hidden_layers = (int(sanitized),)
            else:
                hidden_layers = tuple(input_val)
            
            self.assertEqual(hidden_layers, expected)

    def test_neural_network_training(self):
        """Test that Neural Network trains successfully."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        
        # Should have loss curve after training
        self.assertTrue(hasattr(mlp, 'loss_curve_'))
        self.assertGreater(len(mlp.loss_curve_), 0)

    def test_neural_network_convergence(self):
        """Test that Neural Network converges or reaches max iterations."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=50, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        
        # Number of iterations should not exceed max_iter
        self.assertLessEqual(mlp.n_iter_, 50)

    # ===== METRICS TESTS =====
    
    def test_classification_metrics_range(self):
        """Test that all classification metrics are in valid range [0, 1]."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        X_train, X_test, y_train, y_test = prepare_data(
            self.sample_classification_data,
            'target',
            is_classification=True
        )
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, f"{metric_name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{metric_name} should be <= 1")

    def test_zero_division_handling(self):
        """Test that metrics handle edge cases with zero_division parameter."""
        from sklearn.metrics import precision_score
        
        # Edge case: all predictions same class
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        
        # Should not raise error with zero_division=0
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.assertIsNotNone(precision)

    # ===== EDGE CASES =====
    
    def test_minimum_sample_size(self):
        """Test handling of minimum sample size."""
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        
        X_train, X_test, y_train, y_test = prepare_data(
            small_data,
            'target',
            test_size=0.25,
            is_classification=True
        )
        
        # Should still split successfully
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

    def test_single_class_edge_case(self):
        """Test behavior with single class in data."""
        single_class_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 0, 0, 0]
        })
        
        X_train, X_test, y_train, y_test = prepare_data(
            single_class_data,
            'target',
            is_classification=True
        )
        
        # Should have only one unique class
        all_labels = np.concatenate([y_train, y_test])
        self.assertEqual(len(np.unique(all_labels)), 1)


if __name__ == '__main__':
    # Print banner
    print("="*80)
    print("🚀 Starting ML Logic Tests")
    print("="*80)
    print(f"📡 Server URL: {API_URL}")
    print("="*80)
    
    unittest.main(verbosity=2)
