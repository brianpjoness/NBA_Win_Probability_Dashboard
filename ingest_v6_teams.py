import pyodbc
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = r"C:\Users\brian\Downloads\2018-19_pbp.csv"   # Check path before execution
PASSWORD = 'UPDATE-INFO-HERE'
SERVER = 'UPDATE-INFO-HERE'
DATABASE = 'UPDATE-INFO-HERE'
USERNAME = 'UPDATE-INFO-HERE'
DRIVER = '{ODBC Driver 18 for SQL Server}'


def get_db_connection():
    """Establish connection to Azure SQL Database."""
    conn_str = (
        f'DRIVER={DRIVER};'
        f'SERVER={SERVER},1433;' 
        f'DATABASE={DATABASE};'
        f'UID={USERNAME};'
        f'PWD={PASSWORD};'
        'Encrypt=yes;'
        'TrustServerCertificate=yes;'
        'Connection Timeout=30;'
    )
    return pyodbc.connect(conn_str)


def time_to_seconds(time_str):
    """Convert time string (MM:SS) to total seconds."""
    try:
        if pd.isna(time_str): return 0
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except:
        return 0


def ingest_teams_fix():
    print("Reading CSV file...")

    # Load specific columns required for team identification and score reconstruction
    cols = [
        'GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SCOREMARGIN', 'EVENTNUM',
        'PLAYER1_TEAM_ID', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION'
    ]
    # Use a lambda to ensure only existing columns are loaded
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in cols)

    print(f"   Loaded {len(df)} rows. Starting team identification...")

    # --- 1. DERIVE HOME/AWAY TEAMS ---
    # Create temp dataframes for Home and Away events
    home_events = df.dropna(subset=['HOMEDESCRIPTION', 'PLAYER1_TEAM_ID'])
    away_events = df.dropna(subset=['VISITORDESCRIPTION', 'PLAYER1_TEAM_ID'])

    # Find the most common TeamID for Home/Away per Game
    home_map = home_events.groupby('GAME_ID')['PLAYER1_TEAM_ID'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
    away_map = away_events.groupby('GAME_ID')['PLAYER1_TEAM_ID'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0)

    # Map these back to the main dataframe
    df['HomeTeamID'] = df['GAME_ID'].map(home_map).fillna(0).astype(int)
    df['AwayTeamID'] = df['GAME_ID'].map(away_map).fillna(0).astype(int)

    print("   Teams identified for each game.")

    # --- 2. CLEAN MARGIN & TIME ---
    df['SCOREMARGIN'] = df['SCOREMARGIN'].astype(str).replace('TIE', '0').replace('nan', np.nan)
    df['SCOREMARGIN'] = df['SCOREMARGIN'].ffill().fillna('0')
    df['Margin'] = pd.to_numeric(df['SCOREMARGIN'], errors='coerce').fillna(0)

    df['ClockSeconds'] = df['PCTIMESTRING'].apply(time_to_seconds)
    df['Quarter'] = df['PERIOD']
    df['TrueTimeSec'] = ((4 - df['Quarter']) * 720) + df['ClockSeconds']
    df.loc[df['Quarter'] > 4, 'TrueTimeSec'] = df['ClockSeconds']

    # --- 3. RECONSTRUCT SCORES ---
    print("   Reconstructing Scoreboard...")
    df = df.sort_values(['GAME_ID', 'EVENTNUM'], ascending=[True, True])
    df['PrevMargin'] = df.groupby('GAME_ID')['Margin'].shift(1).fillna(0)
    df['Delta'] = df['Margin'] - df['PrevMargin']

    df['HomePoints'] = np.where(df['Delta'] > 0, df['Delta'], 0)
    df['AwayPoints'] = np.where(df['Delta'] < 0, abs(df['Delta']), 0)
    df['HomeScore'] = df.groupby('GAME_ID')['HomePoints'].cumsum()
    df['AwayScore'] = df.groupby('GAME_ID')['AwayPoints'].cumsum()

    # --- 4. DETERMINE WINNER ---
    final_scores = df.sort_values(['GAME_ID', 'EVENTNUM'], ascending=[True, False]) \
        .groupby('GAME_ID').head(1)[['GAME_ID', 'HomeScore', 'AwayScore']]
    final_scores['HomeWin'] = np.where(final_scores['HomeScore'] > final_scores['AwayScore'], 1, 0)
    df = df.merge(final_scores[['GAME_ID', 'HomeWin']], on='GAME_ID', how='left')

    # --- 5. PREPARE UPLOAD ---
    upload_df = pd.DataFrame()
    upload_df['GameID'] = df['GAME_ID'].astype(str)
    upload_df['HomeTeamID'] = df['HomeTeamID']
    upload_df['AwayTeamID'] = df['AwayTeamID']
    upload_df['Quarter'] = df['Quarter']
    upload_df['TimeRemainingSec'] = df['TrueTimeSec']
    upload_df['HomeScore'] = df['HomeScore']
    upload_df['AwayScore'] = df['AwayScore']
    upload_df['HomeWin'] = df['HomeWin']

    upload_df = upload_df.dropna(subset=['Quarter'])

    print(f"Uploading {len(upload_df)} rows with detected teams to Azure...")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.fast_executemany = True

    cursor.execute("TRUNCATE TABLE GameStates")
    conn.commit()

    records = upload_df.values.tolist()
    query = """
        INSERT INTO GameStates 
        (GameID, HomeTeamID, AwayTeamID, Quarter, TimeRemainingSec, 
         HomeScore, AwayScore, HomeWin, GameDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
    """

    chunk_size = 5000
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            cursor.executemany(query, chunk)
            conn.commit()
            if i % 50000 == 0: print(f"   ...Uploaded chunk {i}")
        except Exception as e:
            print(f"   Error: {e}")

    conn.close()
    print("SUCCESS! Team IDs have been successfully reverse-engineered and saved.")


if __name__ == "__main__":
    ingest_teams_fix()