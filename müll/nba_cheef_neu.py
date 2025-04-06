
import os
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import json
from datetime import datetime

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster, playergamelog, leaguestandings

st.set_page_config(layout="wide", page_title="NBA Cheef", initial_sidebar_state="expanded")

st.markdown("""
<style>
.title { font-size: 3rem; font-weight: bold; color: #333333; text-align: center; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'>🏀 NBA Cheef – by Master Halil</h1>", unsafe_allow_html=True)

team_list = teams.get_teams()
team_names = [team["full_name"] for team in team_list]
team_id_map = {team["full_name"]: team["id"] for team in team_list}

history_dir = "C:/Users/zabun/Desktop/moneyline9x/Historie"
history_csv = os.path.join(history_dir, "historie.csv")
os.makedirs(history_dir, exist_ok=True)

def df_to_json_compatible(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df.to_dict(orient="records")

def get_last_n_games(team_id, num_games=10):
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games.sort_values(by="GAME_DATE", ascending=False).head(num_games)

def compute_rolling_features(games_df, stat, n):
    if len(games_df) == 0:
        return np.nan
    return games_df.head(n)[stat].mean()

def build_model_input(home_team_id, away_team_id, elo_df, manual_values=None):
    home_games = get_last_n_games(home_team_id)
    away_games = get_last_n_games(away_team_id)

    def get_value(col, team_id, default):
        if manual_values and col in manual_values and manual_values[col] != "":
            return float(manual_values[col])
        return elo_df.loc[elo_df["TEAM_ID"] == team_id, default].values[0]

    features = {
        "ELO_HOME": get_value("ELO_HOME", home_team_id, "ELO"),
        "ELO_AWAY": get_value("ELO_AWAY", away_team_id, "ELO"),
        "MOMENTUM_HOME_CLEAN": get_value("MOMENTUM_HOME_CLEAN", home_team_id, "MOMENTUM"),
        "MOMENTUM_AWAY_CLEAN": get_value("MOMENTUM_AWAY_CLEAN", away_team_id, "MOMENTUM"),
    }

    stats = ["PTS", "FG_PCT", "REB", "AST", "STL", "BLK"]
    for stat in stats:
        features[f"AVG_HOME_{stat}_3"] = compute_rolling_features(home_games, stat, 3)
        features[f"AVG_HOME_{stat}_7"] = compute_rolling_features(home_games, stat, 7)
        features[f"AVG_AWAY_{stat}_3"] = compute_rolling_features(away_games, stat, 3)
        features[f"AVG_AWAY_{stat}_7"] = compute_rolling_features(away_games, stat, 7)

    return pd.DataFrame([features])

def get_team_standings_info(team_id):
    standings_df = leaguestandings.LeagueStandings(season='2024-25').get_data_frames()[0]
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

def get_last_game_players(team_id):
    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    player_stats = []
    for _, row in roster.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER']
        try:
            logs = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
            last_game = logs.iloc[0]
            print(f"Lade {player_name}: {last_game['PTS']} Punkte")
            player_stats.append({
                "Name": player_name,
                "Matchup": last_game["MATCHUP"],
                "MIN": last_game["MIN"],
                "PTS": last_game["PTS"],
                "REB": last_game["REB"],
                "AST": last_game["AST"],
                "STL": last_game["STL"],
                "BLK": last_game["BLK"],
                "FG_PCT": round(last_game["FG_PCT"] * 100, 1)
            })
        except:
            continue
        time.sleep(0.7)
    return pd.DataFrame(player_stats)

# Der Rest deines Codes bleibt wie er ist. Du kannst nun nach Vorhersageauswahl `get_last_game_players(team_id_map[home_team])` oder `away_team` verwenden.
