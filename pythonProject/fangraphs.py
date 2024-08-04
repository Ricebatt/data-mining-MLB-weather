import requests
import pandas as pd
from datetime import datetime


def get_mlb_game_data(year):
    base_url = 'https://statsapi.mlb.com/api/v1/schedule'
    start_date = f'{year}-03-30'
    end_date = f'{year}-10-01'

    params = {
        'sportId': 1,
        'startDate': start_date,
        'endDate': end_date,
        'hydrate': 'team,linescore,boxscore',
        'language': 'en'
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    games = []
    for date in data['dates']:
        for game in date['games']:
            game_id = game['gamePk']
            game_url = f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore'
            game_response = requests.get(game_url)
            game_data = game_response.json()

            if 'teams' in game_data and 'home' in game_data['teams'] and 'away' in game_data['teams']:
                game_datetime = datetime.strptime(game['gameDate'], '%Y-%m-%dT%H:%M:%SZ')
                game_time = game_datetime.strftime('%Y-%m-%d %H:%M:%S')

                home_stats = game_data['teams']['home']['teamStats']['batting']
                away_stats = game_data['teams']['away']['teamStats']['batting']

                for team, stats in [('Home', home_stats), ('Away', away_stats)]:
                    singles = stats['hits'] - (stats['doubles'] + stats['triples'] + stats['homeRuns'])
                    total_bases = (singles + 2 * stats['doubles'] + 3 * stats['triples'] + 4 * stats['homeRuns'])
                    slugging_pct = total_bases / stats['atBats'] if stats['atBats'] > 0 else 0

                    game_info = {
                        'Date': game['officialDate'],
                        'Time': game_time,
                        'Location': game['venue']['name'],
                        'Team': game['teams'][team.lower()]['team']['name'],
                        'Opponent': game['teams']['away' if team == 'Home' else 'home']['team']['name'],
                        'At Bats': stats['atBats'],
                        'Singles': singles,
                        'Doubles': stats['doubles'],
                        'Triples': stats['triples'],
                        'Home Runs': stats['homeRuns'],
                        'Total Bases': total_bases,
                        'Slugging Pct': slugging_pct
                    }
                    games.append(game_info)

    return games


# Collect data for each year from 2013 to 2023
all_games = []
for year in range(2013, 2024):
    print(f"Fetching data for {year}...")
    year_games = get_mlb_game_data(year)
    all_games.extend(year_games)

# Convert to DataFrame
df = pd.DataFrame(all_games)

# Save to a CSV file
df.to_csv('mlb_games_2013_to_2023.csv', index=False)

print("Data for 2013 to 2023 has been saved to mlb_games_2013_to_2023.csv")
