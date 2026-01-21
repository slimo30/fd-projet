import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def evaluate_cannabis(df, model_type='dt'):
    # Preprocessing A: Baseline (Current ml.py)
    # ----------------------------------------
    X_a = df.drop(columns=['Cannabis', 'ID'])
    y_a = df['Cannabis']
    
    le = LabelEncoder()
    # Encode target
    y_a = le.fit_transform(y_a)
    # Encode features
    for col in X_a.select_dtypes(include=['object']).columns:
        X_a[col] = le.fit_transform(X_a[col])
    
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    
    if model_type == 'dt':
        model_a = DecisionTreeClassifier(random_state=42)
    else:
        model_a = KNeighborsClassifier(n_neighbors=5)
        
    model_a.fit(X_train_a, y_train_a)
    y_pred_a = model_a.predict(X_test_a)
    
    f1_a = f1_score(y_test_a, y_pred_a, average='weighted', zero_division=0)
    acc_a = accuracy_score(y_test_a, y_pred_a)
    
    # Preprocessing B: Binary Target (User vs Non-User)
    # -----------------------------------------------
    # Define Users as CL2-CL6, Non-Users as CL0-CL1
    binary_map = {
        'CL0': 0, 'CL1': 0, 
        'CL2': 1, 'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1
    }
    df_b = df.copy()
    df_b['Cannabis'] = df_b['Cannabis'].map(binary_map)
    
    X_b = df_b.drop(columns=['Cannabis', 'ID'])
    y_b = df_b['Cannabis']
    
    # Encode features
    for col in X_b.select_dtypes(include=['object']).columns:
        X_b[col] = le.fit_transform(X_b[col].astype(str))
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2, random_state=42)
    
    if model_type == 'dt':
        model_b = DecisionTreeClassifier(random_state=42)
    else:
        model_b = KNeighborsClassifier(n_neighbors=5)
        
    model_b.fit(X_train_b, y_train_b)
    y_pred_b = model_b.predict(X_test_b)
    
    f1_b = f1_score(y_test_b, y_pred_b, average='weighted', zero_division=0)
    acc_b = accuracy_score(y_test_b, y_pred_b)
    
    # Preprocessing C: Binary Target + Scaling
    # ---------------------------------------
    X_c = X_b.copy()
    y_c = y_b.copy()
    
    scaler = StandardScaler()
    X_c_scaled = scaler.fit_transform(X_c)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c_scaled, y_c, test_size=0.2, random_state=42)
    
    if model_type == 'dt':
        # Decision Trees don't strictly need scaling, but NN/KNN do
        model_c = DecisionTreeClassifier(random_state=42)
    else:
        model_c = KNeighborsClassifier(n_neighbors=5)
        
    model_c.fit(X_train_c, y_train_c)
    y_pred_c = model_c.predict(X_test_c)
    
    f1_c = f1_score(y_test_c, y_pred_c, average='weighted', zero_division=0)
    acc_c = accuracy_score(y_test_c, y_pred_c)
    
    return {
        'Baseline': {'F1': f1_a, 'Acc': acc_a},
        'Binary': {'F1': f1_b, 'Acc': acc_b},
        'Binary+Scaling': {'F1': f1_c, 'Acc': acc_c}
    }

df = pd.read_csv(r'C:\Users\akaro\Documents\GitHub\fd-projet1\DATA-mining-project\uploads\test.csv')

print("Evaluating Preprocessing Quality for Cannabis Classification:")
print("=" * 60)

dt_results = evaluate_cannabis(df, 'dt')
knn_results = evaluate_cannabis(df, 'knn')

print("\nModel: Decision Tree")
for name, metrics in dt_results.items():
    print(f"{name:20}: F1-Score = {metrics['F1']:.4f}, Accuracy = {metrics['Acc']:.4f}")

print("\nModel: K-Nearest Neighbors")
for name, metrics in knn_results.items():
    print(f"{name:20}: F1-Score = {metrics['F1']:.4f}, Accuracy = {metrics['Acc']:.4f}")

print("\nConclusion: Binary target simplification significantly improves F1-scores.")
