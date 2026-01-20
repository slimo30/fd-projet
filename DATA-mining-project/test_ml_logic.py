import unittest
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from app
sys.path.append(os.getcwd())

from app.routes.ml import prepare_data

class TestMLLogic(unittest.TestCase):
    def test_prepare_data_float_labels_for_classification(self):
        """Test that float labels (like 0.0, 1.0) are converted to integers for classification."""
        data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0.0, 1.0, 0.0, 1.0]
        }
        df = pd.DataFrame(data)
        
        X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
        
        # Check types - use .values.dtype if it's a series
        dtype = y_train.dtype
        self.assertTrue(np.issubdtype(dtype, np.integer), f"Expected integer dtype, got {dtype}")
        
        # Check values
        unique_vals = np.unique(np.concatenate([y_train, y_test]))
        self.assertTrue(np.array_equal(unique_vals, [0, 1]), f"Expected [0, 1], got {unique_vals}")

    def test_prepare_data_mixed_labels_for_classification(self):
         """Test that mixed type labels or messy floats work."""
         data = {
            'feature1': [1, 2, 3, 4],
            'target': [0.0, 1.0, 2.0, 0.0] 
        }
         df = pd.DataFrame(data)
         X_train, X_test, y_train, y_test = prepare_data(df, 'target', is_classification=True)
         self.assertTrue(np.issubdtype(y_train.dtype, np.integer))

if __name__ == '__main__':
    unittest.main()
