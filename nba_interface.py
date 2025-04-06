import os
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import json
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster, playergamelog, leaguestandings

st.set_page_config(layout="wide", page_title="NBA Cheef", initial_sidebar_state="expanded")

# -----------------------------------
# Header-Bild (mittig, 2 Varianten)
# -----------------------------------
banner_path = r"C:\Users\zabun\Desktop\moneyline9x\Master chief.jpg"

if os.path.exists(banner_path):
    # Variante A: per Spalten-Layout
    col_left, col_mid, col_right = st.columns([1,2,1])
    with col_mid:
        st.image(banner_path, width=250)
        st.markdown("<h2 style='text-align: center;'>NBA Chief – by Master Halil</h2>", unsafe_allow_html=True)

    # Variante B (auskommentiert): per HTML
    # banner_path_forward = banner_path.replace("\\", "/")
    # st.markdown(
    #     f"""
    #     <p style='text-align: center;'>
    #         <img src='file:///{banner_path_forward}' width='250'/>
    #     </p>
    #     <h2 style='text-align: center;'>NBA Chief – by Master Halil</h2>
    #     """,
    #     unsafe_allow_html=True
    # )

else:
    st.warning("Bild konnte nicht geladen werden. Bitte Pfad prüfen.")

# -----------------------------------
# Teams und IDs
# -----------------------------------
team_list = teams.get_teams()
sorted_team_names = sorted([team["full_name"] for team in team_list])
sorted_team_names = ["Bitte wählen"] + sorted_team_names

team_id_map = {}
for team in team_list:
    team_id_map[team["full_name"]] = team["id"]

# -----------------------------------
# Historie-Pfade
# -----------------------------------
history_dir = "C:/Users/zabun/Desktop/moneyline9x/Historie"
history_csv = os.path.join(history_dir, "historie.csv")
os.makedirs(history_dir, exist_ok=True)

# -----------------------------------
# Hilfsfunktion, um DataFrames datums- und json-kompatibel zu machen
# -----------------------------------
def df_to_json_compatible(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df.to_dict(orient="records")

# Tabellen-Helfer
def show_table(title: str, df: pd.DataFrame):
    if len(df) > 5:
        with st.expander(title):
            st.dataframe(df)
    else:
        st.markdown(f"**{title}**")
        st.dataframe(df)

# -----------------------------------
# API-Funktionen
# -----------------------------------
def get_last_n_games(team_id, num_games=10):
    from nba_api.stats.endpoints import leaguegamefinder
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable="2024-25")
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games.sort_values(by="GAME_DATE", ascending=False).head(num_games)

def get_all_season_games(team_id):
    from nba_api.stats.endpoints import leaguegamefinder
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable="2024-25")
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games.sort_values(by="GAME_DATE", ascending=True)

def compute_rolling_features(games_df, stat, n):
    if len(games_df) == 0:
        return np.nan
    return games_df.head(n)[stat].mean()

def calculate_team_elo(team_id, K=10):
    games = get_all_season_games(team_id)
    elo = 1500
    for idx, game in games.iterrows():
        result = 1 if game['WL'] == 'W' else 0
        expected = 1 / (1 + 10 ** ((1500 - elo) / 400))
        elo = elo + K * (result - expected)
    return elo

def build_model_input(home_team_id, away_team_id, elo_df, manual_values=None):
    home_games = get_last_n_games(home_team_id)
    away_games = get_last_n_games(away_team_id)

    def get_value(col, team_id, default):
        if manual_values and col in manual_values and manual_values[col] != "":
            return float(manual_values[col])
        if default == "ELO":
            return calculate_team_elo(team_id)
        return elo_df.loc[elo_df["TEAM_ID"] == team_id, default].values[0]

    elo_home_val = get_value("ELO_HOME", home_team_id, "ELO")
    elo_away_val = get_value("ELO_AWAY", away_team_id, "ELO")

    if manual_values and "MOMENTUM_HOME_CLEAN" in manual_values and manual_values["MOMENTUM_HOME_CLEAN"] != "":
        momentum_home_val = float(manual_values["MOMENTUM_HOME_CLEAN"])
    else:
        momentum_home_val = elo_df.loc[elo_df["TEAM_ID"] == home_team_id, "MOMENTUM"].values[0]

    if manual_values and "MOMENTUM_AWAY_CLEAN" in manual_values and manual_values["MOMENTUM_AWAY_CLEAN"] != "":
        momentum_away_val = float(manual_values["MOMENTUM_AWAY_CLEAN"])
    else:
        momentum_away_val = elo_df.loc[elo_df["TEAM_ID"] == away_team_id, "MOMENTUM"].values[0]

    features = {
        "ELO_HOME": elo_home_val,
        "ELO_AWAY": elo_away_val,
        "MOMENTUM_HOME_CLEAN": momentum_home_val,
        "MOMENTUM_AWAY_CLEAN": momentum_away_val,
    }

    stats = ["PTS", "FG_PCT", "REB", "AST", "STL", "BLK"]
    for stat in stats:
        features[f"AVG_HOME_{stat}_3"] = compute_rolling_features(home_games, stat, 3)
        features[f"AVG_HOME_{stat}_7"] = compute_rolling_features(home_games, stat, 7)
        features[f"AVG_AWAY_{stat}_3"] = compute_rolling_features(away_games, stat, 3)
        features[f"AVG_AWAY_{stat}_7"] = compute_rolling_features(away_games, stat, 7)

    return pd.DataFrame([features])

def get_last_10_games_table(team_id):
    games = get_last_n_games(team_id, 10)
    games['Heim/Auswärts'] = games['MATCHUP'].apply(lambda x: "Heim" if "vs." in x else "Auswärts")
    games['Gegner'] = games['MATCHUP'].apply(lambda x: x.split(" ")[-1])
    games['FG_PCT'] = games['FG_PCT'] * 100
    return games[['GAME_DATE', 'Heim/Auswärts', 'Gegner', 'WL', 'PTS', 'PLUS_MINUS', 'FG_PCT', 'REB', 'AST', 'STL', 'BLK']]

def get_team_standings_info(team_id):
    from nba_api.stats.endpoints import leaguestandings
    standings_df = leaguestandings.LeagueStandings(season="2024-25").get_data_frames()[0]
    col_team_id = "TeamID" if "TeamID" in standings_df.columns else "TEAM_ID"
    row = standings_df[standings_df[col_team_id] == team_id]
    if row.empty:
        return None

    def safe_get(colname):
        for c in standings_df.columns:
            if colname.lower() in c.lower():
                return row[c].values[0]
        return "Nicht verfügbar"

    data = {
        "TeamName": safe_get("TeamName"),
        "ConferenceRank": safe_get("ConferenceRank"),
        "PlayoffRank": safe_get("PlayoffRank"),
        "WINS": safe_get("WINS"),
        "LOSSES": safe_get("LOSSES"),
        "WinPCT": round(float(safe_get("WinPCT")), 3) if safe_get("WinPCT") != "Nicht verfügbar" else "Nicht verfügbar",
    }
    return data

def save_full_history(big_dict, timestamp):
    file_name = f"eintrag_{timestamp.replace(':', '-')}.json"
    file_path = os.path.join(history_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(big_dict, f, indent=4, ensure_ascii=False)

def convert_min_to_float(min_str):
    try:
        if isinstance(min_str, str) and ':' in min_str:
            parts = min_str.split(':')
            return int(parts[0]) + int(parts[1]) / 60
        return float(min_str)
    except:
        return 0.0

def get_combined_boxscores(loaded_data):
    all_records = []
    home_team_id = loaded_data.get("home_team_id")
    away_team_id = loaded_data.get("away_team_id")
    for game_id, bs in loaded_data["boxscores_home"].items():
        for rec in bs:
            rec["GAME_ID"] = game_id
            if "TEAM_ID" in rec:
                if rec["TEAM_ID"] == home_team_id:
                    rec["TEAM"] = loaded_data["home_team"]
                elif rec["TEAM_ID"] == away_team_id:
                    rec["TEAM"] = loaded_data["away_team"]
                else:
                    rec["TEAM"] = "Unbekannt"
            else:
                rec["TEAM"] = loaded_data["home_team"]
            all_records.append(rec)
    for game_id, bs in loaded_data["boxscores_away"].items():
        for rec in bs:
            rec["GAME_ID"] = game_id
            if "TEAM_ID" in rec:
                if rec["TEAM_ID"] == home_team_id:
                    rec["TEAM"] = loaded_data["home_team"]
                elif rec["TEAM_ID"] == away_team_id:
                    rec["TEAM"] = loaded_data["away_team"]
                else:
                    rec["TEAM"] = "Unbekannt"
            else:
                rec["TEAM"] = loaded_data["away_team"]
            all_records.append(rec)
    return pd.DataFrame(all_records)

def get_boxscore_for_game(game_id):
    from nba_api.stats.endpoints import boxscoretraditionalv2
    print(f"Lade Boxscore für Spiel-ID: {game_id}")
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    df = boxscore.get_data_frames()[0]
    if "PLAYER_NAME" in df.columns:
        df.rename(columns={"PLAYER_NAME": "PLAYER"}, inplace=True)
    df = df[df["PLAYER"] != "TEAM"]  # Zeile mit 240 Min. filtern
    def played(min_val):
        if isinstance(min_val, str):
            return min_val not in ["0:00", "0", ""]
        return False
    filtered_df = df[df['MIN'].apply(played)].reset_index(drop=True)
    print(f"Geladene Spieler-Stats: {len(filtered_df)} Spieler, die gespielt haben")
    return filtered_df

def get_last_games_for_stats(team_id, num_games=5):
    games = get_last_n_games(team_id, num_games)
    return games[['GAME_ID', 'GAME_DATE', 'MATCHUP']]

def plot_aggregated_player_stats(loaded_data):
    df_all = get_combined_boxscores(loaded_data)
    if df_all.empty:
        st.warning("Keine Boxscore-Daten für Aggregation gefunden.")
        return
    stats_cols = ["PTS", "AST", "REB", "STL", "BLK"]
    df_avg = df_all.groupby(["TEAM", "PLAYER"])[stats_cols].mean().reset_index()
    teams_in_data = df_avg["TEAM"].unique()
    for team in teams_in_data:
        df_team = df_avg[df_avg["TEAM"] == team]
        fig = px.bar(df_team, x="PLAYER", y=stats_cols, barmode="group",
                     title=f"Durchschnittliche Spieler-Statistiken (letzten 5 Spiele) – {team}")
        st.plotly_chart(fig)

def plot_player_trend(loaded_data, player_name, stat="PTS"):
    records = []
    for game_id, boxscore in loaded_data["boxscores_home"].items():
        for rec in boxscore:
            if rec.get("PLAYER") == player_name:
                game_date = next((game.get("GAME_DATE") for game in loaded_data["last_5_home"] if game.get("GAME_ID") == game_id), None)
                if game_date:
                    rec["GAME_DATE"] = game_date
                    records.append(rec)
    for game_id, boxscore in loaded_data["boxscores_away"].items():
        for rec in boxscore:
            if rec.get("PLAYER") == player_name:
                game_date = next((game.get("GAME_DATE") for game in loaded_data["last_5_away"] if game.get("GAME_ID") == game_id), None)
                if game_date:
                    rec["GAME_DATE"] = game_date
                    records.append(rec)
    if records:
        df_player = pd.DataFrame(records).sort_values(by="GAME_DATE")
        st.markdown(f"### Entwicklung von {player_name} – {stat}")
        st.line_chart(df_player.set_index("GAME_DATE")[[stat]])
    else:
        st.warning(f"Keine Daten für {player_name} gefunden.")

def plot_stats_correlation(loaded_data):
    df_all = get_combined_boxscores(loaded_data)
    if df_all.empty:
        st.warning("Keine Boxscore-Daten für Korrelation gefunden.")
        return
    stats_cols = ["PTS", "AST", "REB", "STL", "BLK"]
    corr_matrix = df_all[stats_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Korrelation zwischen Spieler-Statistiken")
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig)

def plot_team_radar_enhanced(loaded_data):
    df_all = get_combined_boxscores(loaded_data)
    if df_all.empty:
        st.warning("Keine Boxscore-Daten für Team-Radar gefunden.")
        return
    stats_cols = ["PTS", "AST", "REB", "STL", "BLK"]
    df_team = df_all.groupby("TEAM")[stats_cols].mean().reset_index()
    colors = px.colors.qualitative.Plotly
    for idx, row in df_team.iterrows():
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[row[stat] for stat in stats_cols],
            theta=stats_cols,
            fill='toself',
            name=row["TEAM"],
            marker=dict(color=colors[idx % len(colors)])
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, df_team[stats_cols].max().max() + 10])
            ),
            showlegend=True,
            title=f"Team-Radar – {row['TEAM']}",
            width=800, height=600
        )
        st.plotly_chart(fig)

def compute_team_outcome(perf_df):
    if perf_df.empty:
        return {}
    outcome = {
        "Predicted PTS": perf_df["PTS"].sum(),
        "Predicted REB": perf_df["REB"].sum(),
        "Predicted AST": perf_df["AST"].sum(),
        "Predicted STL": perf_df["STL"].sum(),
        "Predicted BLK": perf_df["BLK"].sum()
    }
    return outcome

def plot_aggregated_player_stats_for_team(perf_df, team_label):
    if perf_df.empty:
        st.warning(f"Keine Daten für {team_label}")
        return
    stats_cols = ["PTS", "REB", "AST", "STL", "BLK"]
    fig = px.bar(perf_df, x="PLAYER", y=stats_cols, barmode="group",
                 title=f"Durchschnittliche Spieler-Statistiken (letzten 5 Spiele) – {team_label}")
    st.plotly_chart(fig)

# -----------------------------------
# CSV-Historie laden
# -----------------------------------
history_df = pd.DataFrame()
if os.path.exists(history_csv):
    history_df = pd.read_csv(history_csv)
    history_df["label"] = history_df.apply(lambda row: f"{row['timestamp']} | {row['home_team']} vs {row['away_team']} | {float(row['prob']):.0%}", axis=1)
    selected_history = st.selectbox("▶ Vergangene Begegnung laden", ["(neu)"] + history_df["label"].tolist(), key="history_select")
else:
    selected_history = "(neu)"

# -----------------------------------
# Falls eine vergangene Begegnung geladen wurde
# -----------------------------------
if selected_history != "(neu)":
    selected_row = history_df[history_df["label"] == selected_history].iloc[0]
    json_path = os.path.join(history_dir, f"eintrag_{selected_row['timestamp'].replace(':', '-')}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        st.subheader(f"Geladene Begegnung: {loaded_data['home_team']} vs {loaded_data['away_team']}")
        st.success(f"Gespeicherte Siegwahrscheinlichkeit: {loaded_data['prob']:.2%}")

        st.markdown("### Features")
        st.dataframe(pd.DataFrame([loaded_data['features']]))

        st.markdown("### Letzte 10 Spiele beider Teams")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{loaded_data['home_team']} (Home)**")
            st.dataframe(pd.DataFrame(loaded_data['last_5_home']))
        with col2:
            st.markdown(f"**{loaded_data['away_team']} (Away)**")
            st.dataframe(pd.DataFrame(loaded_data['last_5_away']))

        if "player_performance_home" in loaded_data:
            st.markdown("#### Vorhersage Spielerleistung – Home Team (Gespeichert)")
            st.dataframe(pd.DataFrame(loaded_data["player_performance_home"]))
        if "player_performance_away" in loaded_data:
            st.markdown("#### Vorhersage Spielerleistung – Away Team (Gespeichert)")
            st.dataframe(pd.DataFrame(loaded_data["player_performance_away"]))

        if "player_performance_home" in loaded_data and "player_performance_away" in loaded_data:
            home_perf_df = pd.DataFrame(loaded_data["player_performance_home"])
            away_perf_df = pd.DataFrame(loaded_data["player_performance_away"])
            home_outcome = compute_team_outcome(home_perf_df)
            away_outcome = compute_team_outcome(away_perf_df)
            outcome_df = pd.DataFrame([
                {"Team": loaded_data["home_team"], **home_outcome},
                {"Team": loaded_data["away_team"], **away_outcome}
            ])
            st.markdown("### Predicted Game Outcome")
            st.dataframe(outcome_df)

            st.markdown("### Durchschnittliche Spieler-Statistiken (letzten 5 Spiele)")
            plot_aggregated_player_stats_for_team(home_perf_df, loaded_data["home_team"])
            plot_aggregated_player_stats_for_team(away_perf_df, loaded_data["away_team"])

            st.markdown("### Korrelation zwischen Spieler-Statistiken (Groß)")
            df_all = get_combined_boxscores(loaded_data)
            stats_cols = ["PTS", "AST", "REB", "STL", "BLK"]
            corr_matrix = df_all[stats_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, title="Korrelation zwischen Spieler-Statistiken")
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr)

            st.markdown("### Team-Radar-Diagramme (Groß & Bunt)")
            plot_team_radar_enhanced(loaded_data)

        st.markdown("### Gespeicherte Spieler-Stats (Boxscores)")
        st.markdown("#### Home-Team")
        for game_id, boxscore in loaded_data["boxscores_home"].items():
            st.markdown(f"**Spiel-ID: {game_id}**")
            st.dataframe(pd.DataFrame(boxscore))
        st.markdown("#### Away-Team")
        for game_id, boxscore in loaded_data["boxscores_away"].items():
            st.markdown(f"**Spiel-ID: {game_id}**")
            st.dataframe(pd.DataFrame(boxscore))
        st.stop()

# ------------------------------------------------------------
# Falls (neu) oder kein JSON – frische Vorhersage
# ------------------------------------------------------------
if not os.path.exists(history_csv) or selected_history == "(neu)":
    home_team = "Bitte wählen"
    away_team = "Bitte wählen"

# Team-Auswahl (alphabetisch, mit "Bitte wählen")
home_team = st.selectbox("Home Team", sorted_team_names, index=sorted_team_names.index(home_team), key="home_team_select")
away_team = st.selectbox("Away Team", sorted_team_names, index=sorted_team_names.index(away_team), key="away_team_select")

# Falls "Bitte wählen" => Abbruch
if home_team == "Bitte wählen" or away_team == "Bitte wählen":
    st.warning("Bitte wähle zuerst beide Teams aus, um fortzufahren.")
    st.stop()

home_info = get_team_standings_info(team_id_map[home_team])
away_info = get_team_standings_info(team_id_map[away_team])

st.markdown("### 📊 Aktuelle Team-Infos (Saison-Übersicht)")
col3, col4 = st.columns(2)
with col3:
    if home_info:
        st.subheader(f"{home_team}")
        st.write(f"**Conference Rank**: {home_info['ConferenceRank']}")
        st.write(f"**Playoff Rank**: {home_info['PlayoffRank']}")
        st.write(f"**WINS**: {home_info['WINS']}")
        st.write(f"**LOSSES**: {home_info['LOSSES']}")
        st.write(f"**Win %**: {home_info['WinPCT']}")
with col4:
    if away_info:
        st.subheader(f"{away_team}")
        st.write(f"**Conference Rank**: {away_info['ConferenceRank']}")
        st.write(f"**Playoff Rank**: {away_info['PlayoffRank']}")
        st.write(f"**WINS**: {away_info['WINS']}")
        st.write(f"**LOSSES**: {away_info['LOSSES']}")
        st.write(f"**Win %**: {away_info['WinPCT']}")

st.markdown("### ✏️ Manuelle Eingabe von ELO- und Momentum-Werten (optional)")
col_input1, col_input2 = st.columns(2)
with col_input1:
    elo_home = st.text_input("ELO_HOME (optional)", value="", key="elo_home_input")
    momentum_home = st.text_input("MOMENTUM_HOME_CLEAN (optional)", value="", key="mom_home_input")
with col_input2:
    elo_away = st.text_input("ELO_AWAY (optional)", value="", key="elo_away_input")
    momentum_away = st.text_input("MOMENTUM_AWAY_CLEAN (optional)", value="", key="mom_away_input")

manual_inputs = {
    "ELO_HOME": elo_home,
    "ELO_AWAY": elo_away,
    "MOMENTUM_HOME_CLEAN": momentum_home,
    "MOMENTUM_AWAY_CLEAN": momentum_away
}

with st.expander(f"Alle Spiele der Saison – {home_team}"):
    all_games_home = get_all_season_games(team_id_map[home_team])
    st.dataframe(all_games_home)

with st.expander(f"Alle Spiele der Saison – {away_team}"):
    all_games_away = get_all_season_games(team_id_map[away_team])
    st.dataframe(all_games_away)

if st.button("Vorhersage starten", key="btn_predict"):
    elo_df = pd.read_csv("Aktualisierte_Team-Tabelle.csv")
    df_input = build_model_input(team_id_map[home_team], team_id_map[away_team], elo_df, manual_values=manual_inputs)

    model_bundle = joblib.load("master_nba_model.pkl")
    expected_columns = model_bundle["scaler"].feature_names_in_
    df_input = df_input[expected_columns]
    X_scaled = model_bundle["scaler"].transform(df_input)
    prob = model_bundle["model"].predict_proba(X_scaled)[0][1]

    st.success(f"Siegwahrscheinlichkeit für {home_team}: **{prob:.2%}**")
    st.markdown("#### Feature-Input für Modell")
    st.dataframe(df_input)

    last_10_home = get_last_10_games_table(team_id_map[home_team])
    last_10_away = get_last_10_games_table(team_id_map[away_team])

    st.markdown("### Letzte 10 Spiele beider Teams")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{home_team} (Home)**")
        st.dataframe(last_10_home)
    with col2:
        st.markdown(f"**{away_team} (Away)**")
        st.dataframe(last_10_away)

    last_10_home["GAME_DATE"] = last_10_home["GAME_DATE"].astype(str)
    last_10_away["GAME_DATE"] = last_10_away["GAME_DATE"].astype(str)

    st.markdown("### Vorhersage der Spielerleistung für die nächste Begegnung")
    home_games_stats = get_last_games_for_stats(team_id_map[home_team], num_games=5)
    boxscores_home = {}
    for idx, row in home_games_stats.iterrows():
        game_id = row['GAME_ID']
        df_box = get_boxscore_for_game(game_id)
        boxscores_home[game_id] = df_box.to_dict(orient="records")
        time.sleep(1)
    away_games_stats = get_last_games_for_stats(team_id_map[away_team], num_games=5)
    boxscores_away = {}
    for idx, row in away_games_stats.iterrows():
        game_id = row['GAME_ID']
        df_box = get_boxscore_for_game(game_id)
        boxscores_away[game_id] = df_box.to_dict(orient="records")
        time.sleep(1)

    def compute_player_performance(boxscores, selected_team_id=None):
        all_records = []
        for game_id, records in boxscores.items():
            for rec in records:
                rec["GAME_ID"] = game_id
                all_records.append(rec)
        if not all_records:
            return pd.DataFrame()
        df = pd.DataFrame(all_records)
        if selected_team_id is not None and "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"] == selected_team_id]
        df_avg = df.groupby("PLAYER").agg({
            "PTS": "mean",
            "AST": "mean",
            "REB": "mean",
            "STL": "mean",
            "BLK": "mean",
            "MIN": lambda x: np.mean([convert_min_to_float(val) for val in x])
        }).reset_index()
        return df_avg

    def add_predicted_score(df):
        if df.empty:
            return df
        team_avg_pts = df["PTS"].mean()
        team_avg_ast = df["AST"].mean()
        team_avg_reb = df["REB"].mean()
        team_avg_pts = team_avg_pts if team_avg_pts > 0 else 1
        team_avg_ast = team_avg_ast if team_avg_ast > 0 else 1
        team_avg_reb = team_avg_reb if team_avg_reb > 0 else 1
        df["Predicted Score"] = (0.5 * df["PTS"] / team_avg_pts +
                                 0.3 * df["AST"] / team_avg_ast +
                                 0.2 * df["REB"] / team_avg_reb)
        return df

    home_perf = compute_player_performance(boxscores_home, selected_team_id=team_id_map[home_team])
    away_perf = compute_player_performance(boxscores_away, selected_team_id=team_id_map[away_team])
    home_perf = add_predicted_score(home_perf)
    away_perf = add_predicted_score(away_perf)

    st.markdown("#### Vorhersage Spielerleistung – Home Team")
    st.dataframe(home_perf.sort_values(by="Predicted Score", ascending=False))

    st.markdown("#### Vorhersage Spielerleistung – Away Team")
    st.dataframe(away_perf.sort_values(by="Predicted Score", ascending=False))

    new_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": team_id_map[home_team],
        "away_team_id": team_id_map[away_team],
        "elo_home": elo_home,
        "elo_away": elo_away,
        "momentum_home": momentum_home,
        "momentum_away": momentum_away,
        "prob": prob
    }])

    if os.path.exists(history_csv):
        existing_df = pd.read_csv(history_csv)
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
    else:
        updated_df = new_entry
    updated_df.to_csv(history_csv, index=False)

    feature_dict = df_input.iloc[0].to_dict()

    big_dict = {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": team_id_map[home_team],
        "away_team_id": team_id_map[away_team],
        "prob": float(prob),
        "features": feature_dict,
        "last_5_home": last_10_home.to_dict(orient="records"),
        "last_5_away": last_10_away.to_dict(orient="records"),
        "boxscores_home": boxscores_home,
        "boxscores_away": boxscores_away,
        "player_performance_home": home_perf.to_dict(orient="records"),
        "player_performance_away": away_perf.to_dict(orient="records")
    }

    current_ts = new_entry.iloc[0]["timestamp"]
    save_full_history(big_dict, current_ts)

    st.session_state["home_team"] = home_team
    st.session_state["away_team"] = away_team
    st.session_state["prob"] = prob
    st.session_state["current_ts"] = current_ts

if st.button("Spieler Stats anzeigen", key="btn_stats"):
    if "home_team" not in st.session_state or "away_team" not in st.session_state:
        st.warning("Bitte zuerst auf 'Vorhersage starten' klicken.")
    else:
        home_team_session = st.session_state["home_team"]
        away_team_session = st.session_state["away_team"]
        st.markdown("## Spieler-Stats: Letzte 5 Spiele pro Team")
        col_home, col_away = st.columns(2)

        with col_home:
            st.subheader(f"{home_team_session}: Letzte 5 Spiele (alle Spieler)")
            home_games_stats = get_last_games_for_stats(team_id_map[home_team_session], num_games=5)
            for idx, row in home_games_stats.iterrows():
                game_id = row['GAME_ID']
                date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
                matchup_str = row['MATCHUP']
                st.markdown(f"### {date_str} – {matchup_str}")
                boxscore = get_boxscore_for_game(game_id)
                show_table(f"Boxscore (Spiel-ID: {game_id})", boxscore)
                time.sleep(1)
        with col_away:
            st.subheader(f"{away_team_session}: Letzte 5 Spiele (alle Spieler)")
            away_games_stats = get_last_games_for_stats(team_id_map[away_team_session], num_games=5)
            for idx, row in away_games_stats.iterrows():
                game_id = row['GAME_ID']
                date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
                matchup_str = row['MATCHUP']
                st.markdown(f"### {date_str} – {matchup_str}")
                boxscore = get_boxscore_for_game(game_id)
                show_table(f"Boxscore (Spiel-ID: {game_id})", boxscore)
                time.sleep(1)
        st.success("Alle 5 Spiele wurden geladen und angezeigt.")
