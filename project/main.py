import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
from numpy import random
import xml.etree.ElementTree as ET

with sqlite3.connect('./data/database.sqlite') as con:
    league_id = 1729
    season = '2013/2014'
    query = "SELECT stage, date(date) as match_day, home_team_api_id, away_team_api_id, t1.team_short_name home, home_team_goal, away_team_goal, t2.team_short_name away, m.shoton, m.shotoff from Match m" \
            "  INNER JOIN Team t1 ON home_team_api_id = t1.team_api_id" \
            "  INNER JOIN Team t2 ON away_team_api_id = t2.team_api_id" \
            "  WHERE season = '%s' AND league_id = %d" \
            "  ORDER BY stage, t1.team_short_name" \
            % (season, league_id)
    matches = pd.read_sql_query(query, con)
    print(matches.head())

def count_shot(xml, team):
    return len(xml.findall("*/[team='%s']" % team))


def clean_data(match):
    home_team_id = str(match['home_team_api_id'][0])
    away_team_id = str(match['away_team_api_id'][0])
    shoton = ET.fromstring(match['shoton'][0])
    shotoff = ET.fromstring(match['shotoff'][0])

    home_shot = count_shot(shoton, home_team_id) + count_shot(shotoff, home_team_id)
    away_shot = count_shot(shoton, away_team_id) + count_shot(shotoff, away_team_id)

    print("home team: ", home_shot)
    print("away team: ", away_shot)

    match

for match in matches:
    clean_data(match)