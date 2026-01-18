


# NBA Live Win Probability Pipeline

## Project Overview
Predicts the probability of the home team winning NBA games using an end-to-end ETL + ML + Azure pipeline.

## Features
- Extracts historical and live NBA data
- Transforms and cleans data for ML
- Trains an XGBoost model to predict win probability
- Serves predictions via interactive dashboard (Flask / Dash)
- Hosted and automated in Azure

## Architecture
1. **ETL Pipeline:** Extract → Transform → Load into Azure SQL
2. **ML Model:** Predict home team win probability, saved to Azure Blob Storage
3. **Dashboard:** Live predictions with stats, hosted on Azure App Service
4. **Automation:** Azure Functions refresh data & retrain model

## Tech Stack
Python, Pandas, NumPy, XGBoost, scikit-learn, Flask/Dash, Docker, Azure SQL, Azure App Service, Azure Functions

## How to Run
1. Clone repo
2. Install dependencies
3. Configure Azure credentials
4. Run ETL pipeline
5. Launch dashboard