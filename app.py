import streamlit as st
import pandas as pd
import time
import joblib
import os
import pyodbc
import altair as alt

# --- CONFIGURATION ---
server = st.secrets["DB_SERVER"]
database = st.secrets["DB_DATABASE"]
username = st.secrets["DB_USERNAME"]
password = st.secrets["DB_PASSWORD"]
DRIVER = '{ODBC Driver 18 for SQL Server}'

# --- NBA TEAMS DICTIONARY ---
NBA_TEAMS = {
    1610612737: "Hawks", 1610612738: "Celtics", 1610612739: "Cavaliers",
    1610612740: "Pelicans", 1610612741: "Bulls", 1610612742: "Mavericks",
    1610612743: "Nuggets", 1610612744: "Warriors", 1610612745: "Rockets",
    1610612746: "Clippers", 1610612747: "Lakers", 1610612748: "Heat",
    1610612749: "Bucks", 1610612750: "Timberwolves", 1610612751: "Nets",
    1610612752: "Knicks", 1610612753: "Magic", 1610612754: "Pacers",
    1610612755: "76ers", 1610612756: "Suns", 1610612757: "Blazers",
    1610612758: "Kings", 1610612759: "Spurs", 1610612760: "Thunder",
    1610612761: "Raptors", 1610612762: "Jazz", 1610612763: "Grizzlies",
    1610612764: "Wizards", 1610612765: "Pistons", 1610612766: "Hornets"
}

st.set_page_config(page_title="NBA AI Predictor", page_icon="🏀", layout="wide")


# --- DATABASE & MODEL ---
def get_db_connection():
    # Driver 18 requires the port to be part of the SERVER string (server,1433)
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


@st.cache_resource
def load_model():
    if os.path.exists('nba_win_probability_model.pkl'):
        return joblib.load('nba_win_probability_model.pkl')
    return None


model = load_model()


@st.cache_data(ttl=600)
def get_available_games():
    conn = get_db_connection()
    # Get a list of games with team IDs
    query = """
        SELECT DISTINCT TOP 50 GameID, HomeTeamID, AwayTeamID 
        FROM GameStates ORDER BY GameID
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Format labels for the dropdown (e.g., "0021800001: Celtics vs 76ers")
    game_options = {}
    for _, row in df.iterrows():
        h_name = NBA_TEAMS.get(row['HomeTeamID'], "Home")
        a_name = NBA_TEAMS.get(row['AwayTeamID'], "Away")
        label = f"{row['GameID']}: {h_name} vs {a_name}"
        game_options[label] = row['GameID']

    return game_options


@st.cache_data(ttl=600)
def get_game_data(game_id):
    conn = get_db_connection()
    query = f"""
        SELECT TimeRemainingSec, HomeScore, AwayScore, Quarter, HomeTeamID, AwayTeamID 
        FROM GameStates 
        WHERE GameID = '{game_id}' 
        ORDER BY TimeRemainingSec DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# --- HELPER: FORMAT TIME FOR GRAPH ---
def format_time_label(seconds_remaining):
    """Converts 720 -> 'Q4 12:00'"""
    # 2880 total seconds
    # Q1: 2880-2160, Q2: 2160-1440, Q3: 1440-720, Q4: 720-0
    if seconds_remaining > 2160:
        q = "Q1"
    elif seconds_remaining > 1440:
        q = "Q2"
    elif seconds_remaining > 720:
        q = "Q3"
    else:
        q = "Q4"

    # Calculate clock within quarter
    rem_in_q = seconds_remaining % 720
    if rem_in_q == 0 and seconds_remaining > 0: rem_in_q = 720

    m = int(rem_in_q // 60)
    s = int(rem_in_q % 60)
    return f"{q} {m}:{s:02d}"


# --- UI START ---
st.title("🏀 NBA Win Probability")

# Sidebar
game_dict = get_available_games()
selected_label = st.sidebar.selectbox("Select Game", list(game_dict.keys()))
selected_game_id = game_dict[selected_label]
speed = st.sidebar.slider("Replay Speed", 0.01, 1.0, 0.05)
start_btn = st.sidebar.button("▶️ Start Replay")

# Layout
col1, col2, col3 = st.columns(3)
with col1: home_metric = st.empty()
with col2: prob_metric = st.empty()
with col3: away_metric = st.empty()

st.divider()
chart_placeholder = st.empty()

if start_btn:
    game_data = get_game_data(selected_game_id)

    # Get Team Names
    h_id = game_data.iloc[0]['HomeTeamID']
    a_id = game_data.iloc[0]['AwayTeamID']
    home_name = NBA_TEAMS.get(h_id, "Home")
    away_name = NBA_TEAMS.get(a_id, "Away")

    # Prepare Graph Data
    graph_history = pd.DataFrame(columns=["Elapsed", "Probability", "TimeLabel"])

    # Process only every 5th frame for speed
    stream_data = game_data.iloc[::5].copy()

    for _, row in stream_data.iterrows():
        t = row['TimeRemainingSec']
        h_score = row['HomeScore']
        a_score = row['AwayScore']
        margin = h_score - a_score

        # Predict
        input_data = pd.DataFrame({'ScoreMargin': [margin], 'TimeRemainingSec': [t]})
        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = 0.5

        # Update Metrics
        home_metric.metric(home_name, h_score)
        away_metric.metric(away_name, a_score)
        prob_metric.metric(f"{home_name} Win %", f"{prob:.1%}", delta=f"{margin}")

        # Update Graph Data
        # X-axis: 0 to 48 minutes (Elapsed Time)
        elapsed_min = (2880 - t) / 60
        time_lbl = format_time_label(t)

        new_row = pd.DataFrame({
            "Elapsed": [elapsed_min],
            "Probability": [prob],
            "TimeLabel": [time_lbl]  # For tooltip
        })
        graph_history = pd.concat([graph_history, new_row], ignore_index=True)

        # --- ALTAIR CHART ---
        # Defines the pretty chart
        chart = alt.Chart(graph_history).mark_line(color='#ff4b4b').encode(
            x=alt.X('Elapsed', title='Game Time (Minutes)', scale=alt.Scale(domain=[0, 48])),
            y=alt.Y('Probability', title='Win Probability', scale=alt.Scale(domain=[0, 1])),
            tooltip=['TimeLabel', alt.Tooltip('Probability', format='.1%')]
        ).properties(height=300)

        # Add Quarter Lines (Vertical rules at 12, 24, 36)
        rules = alt.Chart(pd.DataFrame({'x': [12, 24, 36]})).mark_rule(color='gray', strokeDash=[5, 5]).encode(x='x')

        chart_placeholder.altair_chart(chart + rules, use_container_width=True)

        time.sleep(speed)

    # --- 🆕 ADD THIS BLOCK AFTER THE LOOP FINISHES ---
    # Force the graph to touch 0% or 100% at the very end
    final_margin = stream_data.iloc[-1]['HomeScore'] - stream_data.iloc[-1]['AwayScore']
    final_prob = 1.0 if final_margin > 0 else 0.0

    # Create the 'Game Over' data point (Minute 48.0)
    final_row = pd.DataFrame({
        "Elapsed": [48.0],
        "Probability": [final_prob],
        "TimeLabel": ["FINAL"]
    })

    # Append and redraw one last time
    graph_history = pd.concat([graph_history, final_row], ignore_index=True)

    chart = alt.Chart(graph_history).mark_line(color='#ff4b4b').encode(
        x=alt.X('Elapsed', title='Game Time (Minutes)', scale=alt.Scale(domain=[0, 48])),
        y=alt.Y('Probability', title='Win Probability', scale=alt.Scale(domain=[0, 1])),
        tooltip=['TimeLabel', alt.Tooltip('Probability', format='.1%')]
    ).properties(height=300)

    # Add rules again
    rules = alt.Chart(pd.DataFrame({'x': [12, 24, 36]})).mark_rule(color='gray', strokeDash=[5, 5]).encode(x='x')

    chart_placeholder.altair_chart(chart + rules, use_container_width=True)

    # Update text to show finality
    if final_margin > 0:
        st.success(f"🏁 FINAL: {home_name} Wins!")
        prob_metric.metric(f"{home_name} Win %", "100.0%", delta="Winner")
    else:
        st.error(f"🏁 FINAL: {away_name} Wins!")
        prob_metric.metric(f"{home_name} Win %", "0.0%", delta="Loser")