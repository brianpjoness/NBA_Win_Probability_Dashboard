# NBA Win Probability Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![Azure](https://img.shields.io/badge/Cloud-Azure%20SQL-0078D4)

**Live Demo:** [https://nbawinprobabilitydashboard-dilmbos42xkoivkaeajfzk.streamlit.app/]

## Features
- **Real-Time Inference:** Fetches live NBA scores, game clock status, and team data via API to generate instant win probability predictions for ongoing games.
- **Historical Simulation:** Replays 2018-19 NBA games play-by-play to visualize win probability evolution over time.
- **Context-Aware ML Engine:** Predicts win likelihood based on score margin and time remaining using Random Forest and Logistic Regression models.
- **Dynamic Visualization:** Renders interactive momentum charts with Altair that update in real-time, featuring dynamic team branding.
- **Automated ETL:** Ingests raw play-by-play data, reconstructs scoreboards, and identifies teams automatically.
- **Hybrid Cloud Architecture:** Leverages Microsoft Azure SQL for historical data storage and Streamlit Cloud for the frontend interface.

## Architecture
1. **ETL Pipeline:** Extracts raw CSV data, transforms margin/time features, and loads clean game states into Azure SQL.
2. **ML Model:** Trains and compares classifiers (Random Forest vs. Logistic Regression) to optimize accuracy and serializes the best model.
3. **Frontend Dashboard:** A tabbed Streamlit interface that connects to Azure SQL for historical replay and the NBA Live API for real-time scoring data.
4. **Deployment:** Hosted on Streamlit Cloud with secure secret management for database credentials.

## Tech Stack
**Python, Streamlit, Altair, scikit-learn, pandas, Microsoft Azure SQL, pymssql, joblib, nba_api**

## Repository Structure
- `app.py`: Main application entry point containing the tabbed interface (Live/Historical) and inference logic.
- `ingest_v6_teams.py`: Data transformation script for cleaning, feature engineering, and uploading raw data to Azure.
- `train_model_rf.py`: Model training pipeline that compares algorithms and generates the serialized model.
- `requirements.txt`: Python dependencies for cloud deployment.

## How to Run

1. **Clone the repository**
   ```bash
   git clone [https://github.com/brianpjoness/nba-win-probability.git](https://github.com/brianpjoness/nba-win-probability.git)
   cd nba-win-probability

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Configure Secrets**
   ```bash
   [secrets]
   DB_SERVER = "your-server.database.windows.net"
   DB_DATABASE = "nba_db"
   DB_USERNAME = "your_username"
   DB_PASSWORD = "your_password" 
   
4. **Run the Dashboard**
   ```bash
   streamlit run app.py

Created by Brian Jones | Data Scientist & Engineer