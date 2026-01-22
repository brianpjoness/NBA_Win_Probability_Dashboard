# NBA Win Probability Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![Azure](https://img.shields.io/badge/Cloud-Azure%20SQL-0078D4)


**Live Demo:** [https://nbawinprobabilitydashboard-dilmbos42xkoivkaeajfzk.streamlit.app/]

## Features
- **Historical Simulation:** Replays 2018-19 NBA games play-by-play to visualize win probability evolution.
- **Context-Aware ML Engine:** Predicts win likelihood based on score margin and time remaining using Random Forest and Logistic Regression.
- **Dynamic Visualization:** Renders interactive momentum charts with Altair that update in real-time.
- **Automated ETL:** Ingests raw play-by-play data, reconstructs scoreboards, and identifies teams automatically.
- **Hybrid Cloud Architecture:** leveraged Microsoft Azure SQL for storage and Streamlit Cloud for the frontend interface.

## Architecture
1. **ETL Pipeline:** Extracts raw CSV data, transforms margin/time features, and loads clean game states into Azure SQL.
2. **ML Model:** Trains and compares classifiers (Random Forest vs. Logistic Regression) to optimize accuracy; serializes the best model.
3. **Frontend Dashboard:** Connects to Azure SQL via pymssql to fetch game data and perform real-time inference.
4. **Deployment:** Hosted on Streamlit Cloud with secure secret management for database credentials.

## Tech Stack
**Python, Streamlit, Altair, scikit-learn, pandas, Microsoft Azure SQL, pymssql, joblib**

## Repository Structure
- `app.py`: Main application entry point and dashboard logic.
- `ingest_v6_teams.py`: Data transformation script for cleaning and uploading raw data.
- `train_model_rf.py`: Model training pipeline and comparison logic.
- `requirements.txt`: Python dependencies for cloud deployment.

## How to Run
1. **Clone the repository**
   ```bash
   git clone [https://github.com/brianpjoness/nba-win-probability.git](https://github.com/brianpjoness/nba-win-probability.git)
   cd nba-win-probability
   
2. **Install dependencies**
pip install -r requirements.txt

3. *Configure Secrets Create a .streamlit/secrets.toml file in the root directory:*
[secrets]
DB_SERVER = "your-server.database.windows.net"
DB_DATABASE = "nba_db"
DB_USERNAME = "your_username"
DB_PASSWORD = "your_password"

4. *Run the Dashboard*
streamlit run app.py

Created by Brian Jones | Data Scientist & Engineer 