import pyodbc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib

# --- CONFIGURATION ---
# Database configuration for model training
SERVER = 'UPDATE-INFO-HERE'
DATABASE = 'UPDATE-INFO-HERE'
USERNAME = 'UPDATE-INFO-HERE'
PASSWORD = 'UPDATE-INFO-HERE'
DRIVER = '{ODBC Driver 18 for SQL Server}'


def get_db_connection():
    """Establish connection to Azure SQL Database."""
    conn_str = (
        f'DRIVER={DRIVER};SERVER={SERVER},1433;DATABASE={DATABASE};'
        f'UID={USERNAME};PWD={PASSWORD};'
        'Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;'
    )
    return pyodbc.connect(conn_str)


def train_and_compare():
    print("Fetching clean data from Azure SQL...")
    conn = get_db_connection()
    # Fetch required columns to calculate margin
    query = """
        SELECT HomeScore, AwayScore, TimeRemainingSec, HomeWin 
        FROM GameStates
    """
    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} rows.")

    # Feature Engineering
    df['ScoreMargin'] = df['HomeScore'] - df['AwayScore']

    # Define Features (X) and Target (y)
    X = df[['ScoreMargin', 'TimeRemainingSec']]
    y = df['HomeWin']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MODEL 1: RANDOM FOREST ---
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # --- MODEL 2: LOGISTIC REGRESSION ---
    print("Training Logistic Regression...")
    # C=0.1 helps prevent overfitting, solver='liblinear' is efficient for this dataset size
    lr_model = LogisticRegression(C=0.1, solver='liblinear')
    lr_model.fit(X_train, y_train)

    # --- THE SHOWDOWN: UP 20 with 2 MINS LEFT ---
    print("\nTHE SHOWDOWN: Home Team Up 20 with 2:00 left")
    test_scenario = pd.DataFrame({'ScoreMargin': [20], 'TimeRemainingSec': [120]})

    rf_prob = rf_model.predict_proba(test_scenario)[0][1]
    lr_prob = lr_model.predict_proba(test_scenario)[0][1]

    print(f"   Random Forest says:      {rf_prob:.4%} win chance")
    print(f"   Logistic Regression says: {lr_prob:.4%} win chance")

    # --- METRICS ---
    print("\nAccuracy Check (Test Set):")
    print(f"   RF Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
    print(f"   LR Accuracy: {accuracy_score(y_test, lr_model.predict(X_test)):.4f}")

    # --- SAVING MODEL ---
    print("\nSaving Logistic Regression Model to 'nba_win_probability_model.pkl'...")
    joblib.dump(lr_model, 'nba_win_probability_model.pkl')
    print("Done. Deploy this file to Streamlit Cloud to see the update.")


if __name__ == "__main__":
    train_and_compare()