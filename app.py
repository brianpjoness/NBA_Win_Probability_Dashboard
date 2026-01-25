import streamlit as st
import pandas as pd
import time
import joblib
import os
import pymssql
import altair as alt
from nba_api.live.nba.endpoints import scoreboard  # New import for live data

# --- CONFIGURATION ---
# Load secrets securely
server = st.secrets["DB_SERVER"]
database = st.secrets["DB_DATABASE"]
username = st.secrets["DB_USERNAME"]
password = st.secrets["DB_PASSWORD"]

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
    """Establish connection to Azure SQL Database with retry logic for 'Sleeping' DBs."""
    max_retries = 3
    retry_delay = 5  # Seconds

    for attempt in range(max_retries):
        try:
            return pymssql.connect(
                server=f"{server}:1433",
                user=username,
                password=password,
                database=database,
                login_timeout=30
            )
        except pymssql.OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise e


@st.cache_resource
def load_model():
    """Load the trained machine learning model from disk."""
    if os.path.exists('nba_win_probability_model.pkl'):
        return joblib.load('nba_win_probability_model.pkl')
    return None


model = load_model()


@st.cache_data(ttl=600)
def get_available_games():
    """Fetch list of available games from the database."""
    conn = get_db_connection()
    # Get a list of games with team IDs
    query = """
        SELECT DISTINCT TOP 50 GameID, HomeTeamID, AwayTeamID 
        FROM GameStates ORDER BY GameID
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Format labels for the dropdown
    game_options = {}
    for _, row in df.iterrows():
        h_name = NBA_TEAMS.get(row['HomeTeamID'], "Home")
        a_name = NBA_TEAMS.get(row['AwayTeamID'], "Away")
        label = f"{row['GameID']}: {h_name} vs {a_name}"
        game_options[label] = row['GameID']

    return game_options


@st.cache_data(ttl=600)
def get_game_data(game_id):
    """Fetch play-by-play data for a specific game."""
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


# --- HELPERS ---
def format_time_label(seconds_remaining):
    """Converts total seconds remaining into Quarter and Clock format (e.g., Q4 12:00)."""
    if seconds_remaining > 2160:
        q = "Q1"
    elif seconds_remaining > 1440:
        q = "Q2"
    elif seconds_remaining > 720:
        q = "Q3"
    else:
        q = "Q4"

    rem_in_q = seconds_remaining % 720
    if rem_in_q == 0 and seconds_remaining > 0: rem_in_q = 720

    m = int(rem_in_q // 60)
    s = int(rem_in_q % 60)
    return f"{q} {m}:{s:02d}"


def parse_iso8601_time(duration_str):
    """Parses ISO 8601 duration strings (e.g., PT10M00S) from the live API."""
    if not duration_str: return 720
    try:
        import re
        # Extract Minutes and Seconds using Regex
        match = re.search(r'PT(\d+)M(\d+\.?\d*)S', duration_str)
        if match:
            mins = int(match.group(1))
            secs = float(match.group(2))
            return mins * 60 + int(secs)
        return 0
    except:
        return 0


# --- UI START ---
st.title("🏀 NBA Win Probability")

# Expander explaining the logic
with st.expander("ℹ️ How does this model work?"):
    st.write("""
    This application uses a **Random Forest Classifier** trained on over **500,000 historical NBA plays**.

    * **The Input:** It looks at the current Score Margin (e.g., +10 points) and Time Remaining (e.g., 5:00 in Q4).
    * **The Brain:** The AI compares this situation to thousands of similar historical scenarios.
    * **The Output:** It calculates the precise probability of the Home Team holding onto the lead.
    """)

# Sidebar Footer (Bio)
st.sidebar.markdown("###  Created by Brian Jones")
st.sidebar.info(
    """
    **Data Scientist & Engineer** [LinkedIn](https://www.linkedin.com/in/brianpjoness) | [GitHub](https://github.com/brianpjoness)
    """
)
st.sidebar.divider()

# --- TABS FOR NAVIGATION ---
tab_live, tab_history = st.tabs(["🔴 Live Games", "📜 Historical Replay"])

# ==========================================
# TAB 1: LIVE GAMES
# ==========================================
with tab_live:
    st.header("Today's Live Predictions")

    if st.button("🔄 Refresh Live Scores"):
        try:
            board = scoreboard.ScoreBoard()
            games = board.games.get_dict()

            if not games:
                st.warning("No games found for today yet.")

            for game in games:
                # 1. Parse Data & Logos
                home_team = game['homeTeam']['teamName']
                home_id = game['homeTeam']['teamId']
                home_logo = f"https://cdn.nba.com/logos/nba/{home_id}/global/L/logo.svg"

                away_team = game['awayTeam']['teamName']
                away_id = game['awayTeam']['teamId']
                away_logo = f"https://cdn.nba.com/logos/nba/{away_id}/global/L/logo.svg"

                h_score = game['homeTeam']['score']
                a_score = game['awayTeam']['score']
                period = game['period']
                status = game['gameStatusText']

                # 2. Render Game Card
                with st.container():
                    # Layout: Away Logo | Away Name/Score | Status | Home Name/Score | Home Logo
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

                    with col1:
                        st.image(away_logo, width=60)
                    with col2:
                        st.metric(away_team, a_score)

                    with col3:
                        st.markdown(f"<h3 style='text-align: center;'>{status}</h3>", unsafe_allow_html=True)

                    with col4:
                        st.metric(home_team, h_score)
                    with col5:
                        st.image(home_logo, width=60)

                    # 3. Decision Logic
                    # CASE A: Game is Final
                    if "Final" in status:
                        if h_score > a_score:
                            prob = 1.0
                            st.success(f"✅ FINAL: {home_team} Won")
                        else:
                            prob = 0.0
                            st.error(f"❌ FINAL: {away_team} Won")
                        st.progress(prob)

                    # CASE B: Game is Active (Live)
                    elif "Live" in status or period >= 1:
                        # Calculate Inputs
                        margin = h_score - a_score  # Home Perspective

                        # Parse Time (Approximate for now)
                        clock_str = game['gameClock']  # PT10M00S
                        seconds_left_in_q = parse_iso8601_time(clock_str)
                        total_seconds_left = ((4 - period) * 720) + seconds_left_in_q
                        if total_seconds_left < 0: total_seconds_left = 0

                        # Predict
                        input_df = pd.DataFrame({'ScoreMargin': [margin], 'TimeRemainingSec': [total_seconds_left]})
                        prob = model.predict_proba(input_df)[0][1]

                        st.progress(prob)
                        st.caption(f"Home Win Probability: **{prob:.1%}**")

                    # CASE C: Game hasn't started
                    else:
                        st.info(f"Tip-off scheduled for {status}")

                    st.divider()

        except Exception as e:
            st.error(f"Error fetching live data: {e}")

# ==========================================
# TAB 2: HISTORICAL REPLAY (Existing Logic)
# ==========================================
with tab_history:
    st.header("Game Settings")

    game_dict = get_available_games()
    if game_dict:
        selected_label = st.selectbox("Select Game", list(game_dict.keys()))
        selected_game_id = game_dict[selected_label]
        speed = st.slider("Replay Speed", 0.01, 1.0, 0.05)
        start_btn = st.button("▶️ Start Replay")

        # Initialize the chart placeholder inside the tab
        chart_placeholder = st.empty()

        if start_btn:
            game_data = get_game_data(selected_game_id)

            # Get Team Names & IDs
            h_id = game_data.iloc[0]['HomeTeamID']
            a_id = game_data.iloc[0]['AwayTeamID']
            home_name = NBA_TEAMS.get(h_id, "Home")
            away_name = NBA_TEAMS.get(a_id, "Away")

            # Generate Dynamic Logo URLs
            home_logo_url = f"https://cdn.nba.com/logos/nba/{h_id}/global/L/logo.svg"
            away_logo_url = f"https://cdn.nba.com/logos/nba/{a_id}/global/L/logo.svg"

            # Layout setup inside the button to draw logos first
            col1, col2, col3 = st.columns([1, 2, 1])  # Ratios: Middle column wider for the metric

            with col1:
                st.image(home_logo_url, width=80)
                home_metric = st.empty()  # Placeholder for score

            with col2:
                st.write("")  # Spacer
                st.write("")
                prob_metric = st.empty()  # Placeholder for probability

            with col3:
                st.image(away_logo_url, width=80)
                away_metric = st.empty()  # Placeholder for score

            # Prepare Graph Data
            graph_history = pd.DataFrame(columns=["Elapsed", "Probability", "TimeLabel"])
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
                # Update the empty placeholders created above
                home_metric.metric(home_name, h_score)
                away_metric.metric(away_name, a_score)

                # Detailed probability metric
                prob_metric.metric(
                    f"{home_name} Win Probability",
                    f"{prob:.1%}",
                    delta=f"{margin} pts",
                    delta_color="normal"
                )

                # Update Graph Data
                elapsed_min = (2880 - t) / 60
                time_lbl = format_time_label(t)

                new_row = pd.DataFrame({
                    "Elapsed": [elapsed_min],
                    "Probability": [prob],
                    "TimeLabel": [time_lbl]
                })
                graph_history = pd.concat([graph_history, new_row], ignore_index=True)

                chart = alt.Chart(graph_history).mark_line(color='#ff4b4b').encode(
                    x=alt.X('Elapsed', title='Game Time (Minutes)', scale=alt.Scale(domain=[0, 48])),
                    y=alt.Y('Probability', title='Win Probability', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['TimeLabel', alt.Tooltip('Probability', format='.1%')]
                ).properties(height=300)

                rules = alt.Chart(pd.DataFrame({'x': [12, 24, 36]})).mark_rule(color='gray', strokeDash=[5, 5]).encode(
                    x='x')
                chart_placeholder.altair_chart(chart + rules, use_container_width=True)
                time.sleep(speed)

            # --- FINAL WHISTLE LOGIC ---
            final_margin = stream_data.iloc[-1]['HomeScore'] - stream_data.iloc[-1]['AwayScore']
            final_prob = 1.0 if final_margin > 0 else 0.0
            final_row = pd.DataFrame({"Elapsed": [48.0], "Probability": [final_prob], "TimeLabel": ["FINAL"]})
            graph_history = pd.concat([graph_history, final_row], ignore_index=True)

            chart = alt.Chart(graph_history).mark_line(color='#ff4b4b').encode(
                x=alt.X('Elapsed', title='Game Time (Minutes)', scale=alt.Scale(domain=[0, 48])),
                y=alt.Y('Probability', title='Win Probability', scale=alt.Scale(domain=[0, 1])),
                tooltip=['TimeLabel', alt.Tooltip('Probability', format='.1%')]
            ).properties(height=300)

            chart_placeholder.altair_chart(chart + rules, use_container_width=True)

            if final_margin > 0:
                st.success(f"  FINAL: {home_name} Wins!")
                prob_metric.metric(f"{home_name} Win %", "100.0%", delta="Winner")
            else:
                st.error(f"  FINAL: {away_name} Wins!")
                prob_metric.metric(f"{home_name} Win %", "0.0%", delta="Loser")
    else:
        st.warning("No historical games found or database connection failed.")   # if nothing can be found