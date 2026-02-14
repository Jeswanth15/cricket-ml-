import os, sys
import mysql.connector
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="cricket_ml"
)
cursor = conn.cursor(dictionary=True)



cursor.execute("SELECT * FROM matches WHERE actual_winner_team_id IS NOT NULL")
matches = cursor.fetchall()

cursor.execute("SELECT * FROM match_playing_xi")
playing = cursor.fetchall()

cursor.execute("SELECT * FROM player_phase_performance")
ppp = cursor.fetchall()

cursor.execute("SELECT * FROM team_phase_performance")
tpp = cursor.fetchall()

cursor.execute("SELECT * FROM player_vs_bowling_type")
pvt = cursor.fetchall()

cursor.execute("SELECT * FROM bowler_phase_performance")
bpp = cursor.fetchall()

cursor.execute("SELECT * FROM player")
players = cursor.fetchall()

cursor.execute("SELECT * FROM ground")
grounds = cursor.fetchall()


playing_dict = {}
for r in playing:
    playing_dict.setdefault((r["match_id"], r["team_id"]), []).append(r["player_id"])

ppp_dict = {(r["player_id"], r["phase"]): r for r in ppp}
tpp_dict = {(r["team_id"], r["phase"]): r for r in tpp}
pvt_dict = {(r["player_id"], r["bowling_type"], r["phase"]): r for r in pvt}
bpp_dict = {(r["bowler_id"], r["phase"]): r for r in bpp}
ground_dict = {r["ground_id"]: r for r in grounds}
bowling_type = {r["player_id"]: r["bowling_type"] for r in players}

phases = ["powerplay","middle","death"]


def lineup_player_phase_score(players):
    total, count = 0.0, 0
    for pid in players:
        for ph in phases:
            key = (pid, ph)
            if key in ppp_dict:
                total += float(ppp_dict[key]["strike_rate"])
                count += 1
    return (total/count)/100 if count else 0.0

def team_structural_phase(team_id):
    total = 0.0
    for ph in phases:
        key = (team_id, ph)
        if key in tpp_dict:
            total += float(tpp_dict[key]["run_rate"])
    return total/10.0

def matchup_score(batting_players, bowling_players):
    total, count = 0.0, 0
    for bat in batting_players:
        for bowl in bowling_players:
            btype = bowling_type.get(bowl)
            if not btype:
                continue
            for ph in phases:
                bat_key = (bat, btype, ph)
                bowl_key = (bowl, ph)
                if bat_key in pvt_dict and bowl_key in bpp_dict:
                    sr = float(pvt_dict[bat_key]["strike_rate"])
                    eco = float(bpp_dict[bowl_key]["economy"])
                    total += (sr/100.0 - eco/6.0)
                    count += 1
    return total/count if count else 0.0

def ground_features(ground_id):
    if ground_id in ground_dict:
        g = ground_dict[ground_id]
        scoring = float(g["avg_first_innings_score"] or 150)/200.0
        chase = float(g["chase_win_rate"] or 50)/100.0
        home = g["home_team_id"]
        return scoring, chase, home
    return 0.75, 0.5, None



X, y = [], []

for m in matches:
    t1 = m["team1_id"]
    t2 = m["team2_id"]
    g  = m["ground_id"]

    xi1 = playing_dict.get((m["match_id"], t1), [])
    xi2 = playing_dict.get((m["match_id"], t2), [])

    if not xi1 or not xi2:
        continue

    player_phase_diff = (lineup_player_phase_score(xi1) -
                         lineup_player_phase_score(xi2)) * 3.0

    team_phase_diff = (team_structural_phase(t1) -
                       team_structural_phase(t2)) * 3.0

    matchup_diff = (matchup_score(xi1, xi2) -
                    matchup_score(xi2, xi1)) * 2.5

    scoring, chase_rate, home_team = ground_features(g)

    toss_adv = 0.01 if m["toss_winner_team_id"] == t1 else \
              -0.01 if m["toss_winner_team_id"] == t2 else 0.0

    home_adv = 0.01 if home_team == t1 else \
              -0.01 if home_team == t2 else 0.0

    day_factor = 0.01 if m["day_or_night"] == "night" else 0.0

    X.append([
        player_phase_diff,
        team_phase_diff,
        matchup_diff,
        scoring * 0.3,
        chase_rate * 0.3,
        toss_adv,
        home_adv,
        day_factor
    ])

    y.append(1 if m["actual_winner_team_id"] == t1 else 0)

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

print("\nTotal Samples:", len(X))
print("Class Distribution:", np.bincount(y))


model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

print("\n===== XGBOOST CROSS VALIDATION =====")
print("Accuracy :", round(np.mean(scores["test_accuracy"]),3))
print("Precision:", round(np.mean(scores["test_precision"]),3))
print("Recall   :", round(np.mean(scores["test_recall"]),3))
print("F1 Score :", round(np.mean(scores["test_f1"]),3))

model.fit(X, y)


TEAM_A_ID = int(sys.argv[1])
TEAM_B_ID = int(sys.argv[2])
GROUND_ID = int(sys.argv[3])
TOSS_WINNER_ID = int(sys.argv[4])
TOSS_DECISION = sys.argv[5]
DAY_OR_NIGHT = sys.argv[6]
TEAM_A_PLAYERS = list(map(int, sys.argv[7].split(",")))
TEAM_B_PLAYERS = list(map(int, sys.argv[8].split(",")))

player_phase_diff = (lineup_player_phase_score(TEAM_A_PLAYERS) -
                     lineup_player_phase_score(TEAM_B_PLAYERS)) * 3.0

team_phase_diff = (team_structural_phase(TEAM_A_ID) -
                   team_structural_phase(TEAM_B_ID)) * 3.0

matchup_diff = (matchup_score(TEAM_A_PLAYERS, TEAM_B_PLAYERS) -
                matchup_score(TEAM_B_PLAYERS, TEAM_A_PLAYERS)) * 2.5

scoring_g, chase_rate, home_team = ground_features(GROUND_ID)

toss_adv = 0.01 if TOSS_WINNER_ID == TEAM_A_ID else \
          -0.01 if TOSS_WINNER_ID == TEAM_B_ID else 0.0

home_adv = 0.01 if home_team == TEAM_A_ID else \
          -0.01 if home_team == TEAM_B_ID else 0.0

day_factor = 0.01 if DAY_OR_NIGHT == "night" else 0.0

features = np.array([[
    player_phase_diff,
    team_phase_diff,
    matchup_diff,
    scoring_g * 0.3,
    chase_rate * 0.3,
    toss_adv,
    home_adv,
    day_factor
]], dtype=float)

prob = model.predict_proba(features)[0]

print("\n===== FINAL MATCH PREDICTION (XGBOOST) =====")
print("Predicted Winner:", "TEAM A" if prob[1] > prob[0] else "TEAM B")
print("Team A Win Probability:", round(prob[1]*100,2), "%")
print("Team B Win Probability: ", round(prob[0]*100,2), "%")

print("\nFeature Importance:")
feature_names = ["player_phase","team_phase","matchup","ground","chase","toss","home","day"]
importances = model.feature_importances_

for name, val in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(name, ":", round(val,3))

cursor.close()
conn.close()
