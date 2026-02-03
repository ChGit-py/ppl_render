"""
Fantasy Premier League Analytics Dashboard
Track Defensive Contributions, Expected Metrics, and Value Picks
"""

import requests
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# DATA FETCHING
# =============================================================================

FPL_BASE_URL = "https://fantasy.premierleague.com/api"

def fetch_bootstrap_data():
    response = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    response.raise_for_status()
    return response.json()


def fetch_fixtures():
    """Fetch all fixtures for the season."""
    response = requests.get(f"{FPL_BASE_URL}/fixtures/")
    response.raise_for_status()
    return response.json()


def calculate_fixture_difficulty(fixtures, teams_df, current_gw, num_gameweeks=5):
    """
    Calculate average fixture difficulty for each team over the next N gameweeks.
    Returns dict of team_id -> {fixtures: [...], avg_fdr: float}
    """
    # Get upcoming gameweeks
    upcoming_gws = list(range(current_gw + 1, current_gw + num_gameweeks + 1))

    # Filter to upcoming fixtures
    upcoming_fixtures = [f for f in fixtures if f.get('event') in upcoming_gws]

    # Build team fixture data
    team_fixtures = {}

    for team_id in teams_df['id'].unique():
        team_name = teams_df[teams_df['id'] == team_id]['name'].values[0]
        team_fixtures[team_id] = {
            'team_name': team_name,
            'fixtures': [],
            'fdr_values': [],
            'opponents': []
        }

    for fixture in upcoming_fixtures:
        home_team = fixture['team_h']
        away_team = fixture['team_a']
        gw = fixture['event']

        home_fdr = fixture.get('team_h_difficulty', 3)
        away_fdr = fixture.get('team_a_difficulty', 3)

        home_name = teams_df[teams_df['id'] == home_team]['name'].values[0] if home_team in teams_df['id'].values else 'Unknown'
        away_name = teams_df[teams_df['id'] == away_team]['name'].values[0] if away_team in teams_df['id'].values else 'Unknown'

        # Home team's fixture
        if home_team in team_fixtures:
            team_fixtures[home_team]['fixtures'].append({
                'gw': gw,
                'opponent': away_name,
                'venue': 'H',
                'fdr': home_fdr
            })
            team_fixtures[home_team]['fdr_values'].append(home_fdr)
            team_fixtures[home_team]['opponents'].append(f"{away_name} (H)")

        # Away team's fixture
        if away_team in team_fixtures:
            team_fixtures[away_team]['fixtures'].append({
                'gw': gw,
                'opponent': home_name,
                'venue': 'A',
                'fdr': away_fdr
            })
            team_fixtures[away_team]['fdr_values'].append(away_fdr)
            team_fixtures[away_team]['opponents'].append(f"{home_name} (A)")

    # Calculate averages
    for team_id in team_fixtures:
        fdr_values = team_fixtures[team_id]['fdr_values']
        team_fixtures[team_id]['avg_fdr'] = sum(fdr_values) / len(fdr_values) if fdr_values else 3.0
        team_fixtures[team_id]['fixture_count'] = len(fdr_values)
        # Create fixture string for display
        team_fixtures[team_id]['fixture_string'] = ', '.join(team_fixtures[team_id]['opponents'][:num_gameweeks])

    return team_fixtures


def fetch_player_history(player_id):
    """Fetch individual player's match-by-match history."""
    try:
        response = requests.get(f"{FPL_BASE_URL}/element-summary/{player_id}/", timeout=10)
        response.raise_for_status()
        return response.json().get('history', [])
    except Exception as e:
        print(f"Error fetching player {player_id}: {e}")
        return []


def calculate_bonus_consistency(player_ids, min_minutes=60, bonus_threshold=10):
    """
    Calculate defcon bonus hit rate for multiple players.
    Returns dict of player_id -> stats
    """
    results = {}

    def process_player(player_id):
        history = fetch_player_history(player_id)
        if not history:
            return player_id, None

        # Filter to games with 60+ minutes
        qualifying_games = [g for g in history if g.get('minutes', 0) >= min_minutes]

        if not qualifying_games:
            return player_id, None

        # Count games hitting bonus threshold
        bonus_games = [g for g in qualifying_games if g.get('defensive_contribution', 0) >= bonus_threshold]

        # Calculate stats
        defcon_values = [g.get('defensive_contribution', 0) for g in qualifying_games]

        stats = {
            'qualifying_games': len(qualifying_games),
            'bonus_games': len(bonus_games),
            'hit_rate': (len(bonus_games) / len(qualifying_games)) * 100 if qualifying_games else 0,
            'avg_defcon': sum(defcon_values) / len(defcon_values) if defcon_values else 0,
            'max_defcon': max(defcon_values) if defcon_values else 0,
            'min_defcon': min(defcon_values) if defcon_values else 0,
        }
        return player_id, stats

    # Use threading for faster fetching
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_player, pid): pid for pid in player_ids}
        for future in as_completed(futures):
            player_id, stats = future.result()
            if stats:
                results[player_id] = stats

    return results

# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_player_data(data):
    elements = data['elements']
    teams = {t['id']: t['name'] for t in data['teams']}
    positions = {p['id']: p['singular_name_short'] for p in data['element_types']}

    df = pd.DataFrame(elements)

    df['team_name'] = df['team'].map(teams)
    df['position'] = df['element_type'].map(positions)
    df['price'] = df['now_cost'] / 10

    numeric_cols = ['influence', 'creativity', 'threat', 'ict_index',
                    'expected_goals', 'expected_assists', 'expected_goal_involvements',
                    'expected_goals_conceded']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Defensive Contribution calculations
    df['defcon'] = df['defensive_contribution'].fillna(0).astype(int)
    df['minutes_safe'] = df['minutes'].replace(0, np.nan)
    df['defcon_per_90'] = ((df['defcon'] / df['minutes_safe']) * 90).round(2)

    df['games_played'] = df['minutes_safe'] / 90
    df['defcon_vs_bonus'] = df['defcon_per_90'] - 10
    df['bonus_rate'] = (df['defcon_per_90'] / 10) * 100

    position_defcon_rates = df[df['minutes'] > 450].groupby('position')['defcon_per_90'].mean()
    df['position_avg_defcon_rate'] = df['position'].map(position_defcon_rates)
    df['expected_defcon'] = (df['minutes_safe'] / 90) * df['position_avg_defcon_rate']
    df['defcon_diff'] = df['defcon'] - df['expected_defcon']

    df['points_per_million'] = df['total_points'] / df['price']
    df['xg_diff'] = df['goals_scored'] - df['expected_goals']
    df['xa_diff'] = df['assists'] - df['expected_assists']

    df['form'] = pd.to_numeric(df['form'], errors='coerce')
    df['ppg'] = pd.to_numeric(df['points_per_game'], errors='coerce')
    df['form_vs_season'] = df['form'] - df['ppg']
    df['ownership'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')

    df['cs_per_90'] = (df['clean_sheets'] / df['minutes_safe']) * 90
    df['gc_per_90'] = (df['goals_conceded'] / df['minutes_safe']) * 90

    return df

def get_current_gameweek(data):
    for event in data['events']:
        if event['is_current']:
            return event
    return None

def get_next_gameweek(data):
    for event in data['events']:
        if event['is_next']:
            return event
    return None

# =============================================================================
# STYLING
# =============================================================================

COLORS = {
    'primary': '#37003c',
    'secondary': '#00ff87',
    'accent': '#e90052',
    'background': '#f5f5f5',
    'card_bg': '#ffffff',
    'text_dark': '#333333',
    'text_light': '#666666',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'borderRadius': '12px',
    'padding': '24px',
    'marginBottom': '20px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'border': '1px solid #e0e0e0'
}

STAT_CARD_STYLE = {
    'backgroundColor': COLORS['card_bg'],
    'borderRadius': '12px',
    'padding': '20px',
    'textAlign': 'center',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
    'border': '1px solid #e0e0e0',
    'minHeight': '140px'
}

TABLE_STYLE_CELL = {
    'textAlign': 'left',
    'padding': '12px 16px',
    'fontFamily': 'Arial, sans-serif',
    'fontSize': '14px',
}

TABLE_STYLE_HEADER = {
    'backgroundColor': COLORS['primary'],
    'color': 'white',
    'fontWeight': '600',
    'textTransform': 'uppercase',
    'fontSize': '12px',
    'letterSpacing': '0.5px',
    'padding': '14px 16px'
}

TABLE_STYLE_DATA = {
    'backgroundColor': 'white',
    'borderBottom': '1px solid #e0e0e0'
}

# =============================================================================
# FETCH AND PROCESS DATA
# =============================================================================

print("Fetching FPL data...")
bootstrap_data = fetch_bootstrap_data()
df = process_player_data(bootstrap_data)
current_gw = get_current_gameweek(bootstrap_data)
next_gw = get_next_gameweek(bootstrap_data)

df_active = df[df['minutes'] > 0].copy()

# Fetch bonus consistency data for players with 200+ minutes (outfield only)
print("Fetching player match history for bonus consistency analysis...")
consistency_players = df_active[
    (df_active['minutes'] >= 200) &
    (df_active['position'].isin(['DEF', 'MID', 'FWD']))
]['id'].tolist()

print(f"  Fetching data for {len(consistency_players)} players...")
consistency_data = calculate_bonus_consistency(consistency_players)
print(f"  Retrieved data for {len(consistency_data)} players")

# Add consistency columns to df_active
df_active['qualifying_games'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('qualifying_games'))
df_active['bonus_games'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('bonus_games'))
df_active['hit_rate'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('hit_rate'))
df_active['avg_defcon_qualifying'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('avg_defcon'))
df_active['max_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('max_defcon'))
df_active['min_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('min_defcon'))

# Fetch fixture difficulty data
print("Fetching fixture data...")
fixtures_data = fetch_fixtures()
teams_df = pd.DataFrame(bootstrap_data['teams'])
current_gw_num = current_gw['id'] if current_gw else 1
fixture_difficulty = calculate_fixture_difficulty(fixtures_data, teams_df, current_gw_num, num_gameweeks=5)
print(f"  Calculated fixture difficulty for {len(fixture_difficulty)} teams")

# Add fixture difficulty columns to df_active
df_active['avg_fdr_5'] = df_active['team'].map(lambda x: fixture_difficulty.get(x, {}).get('avg_fdr'))
df_active['fixture_string'] = df_active['team'].map(lambda x: fixture_difficulty.get(x, {}).get('fixture_string'))
df_active['fixture_count'] = df_active['team'].map(lambda x: fixture_difficulty.get(x, {}).get('fixture_count'))

total_managers = bootstrap_data['total_players']
avg_gw_score = current_gw['average_entry_score'] if current_gw else 0
highest_gw_score = current_gw['highest_score'] if current_gw else 0

top_scorer = df_active.nlargest(1, 'total_points').iloc[0] if len(df_active) > 0 else None
most_selected = df_active.nlargest(1, 'ownership').iloc[0] if len(df_active) > 0 else None
best_value = df_active[df_active['minutes'] > 450].nlargest(1, 'points_per_million').iloc[0] if len(df_active[df_active['minutes'] > 450]) > 0 else None
top_form = df_active.nlargest(1, 'form').iloc[0] if len(df_active) > 0 else None

# =============================================================================
# DASH APPLICATION
# =============================================================================

app = Dash(__name__)
server = app.server

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_table_data(dataframe, columns):
    """
    Prepare dataframe for DataTable by selecting columns and converting to proper format.
    Uses pandas JSON handling to ensure proper serialization.
    """
    if dataframe.empty:
        return []

    try:
        # Select only needed columns
        df_subset = dataframe[columns].copy()

        # Replace inf with NaN
        df_subset = df_subset.replace([np.inf, -np.inf], np.nan)

        # Use pandas to_json and back - this guarantees valid JSON serialization
        import json
        json_str = df_subset.to_json(orient='records', date_format='iso')
        records = json.loads(json_str)

        return records
    except Exception as e:
        print(f"Error preparing table data: {e}")
        return []


def build_stat_card(title, value, subtitle=None, color=COLORS['primary']):
    return html.Div([
        html.P(title, style={
            'color': COLORS['text_light'],
            'fontSize': '14px',
            'marginBottom': '8px',
            'fontWeight': '500',
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px'
        }),
        html.H2(value, style={
            'color': color,
            'margin': '0 0 8px 0',
            'fontSize': '32px',
            'fontWeight': '700'
        }),
        html.P(subtitle, style={
            'color': COLORS['text_light'],
            'fontSize': '13px',
            'margin': '0'
        }) if subtitle else None
    ], style=STAT_CARD_STYLE)


def build_player_spotlight(player, title, metric_label, metric_value):
    if player is None:
        return html.Div()

    return html.Div([
        html.Div([
            html.Span(title, style={
                'backgroundColor': COLORS['secondary'],
                'color': COLORS['primary'],
                'padding': '4px 12px',
                'borderRadius': '20px',
                'fontSize': '12px',
                'fontWeight': '600',
                'textTransform': 'uppercase'
            })
        ], style={'marginBottom': '16px'}),
        html.H3(player['web_name'], style={
            'color': COLORS['primary'],
            'margin': '0 0 4px 0',
            'fontSize': '22px',
            'fontWeight': '700'
        }),
        html.P(f"{player['team_name']} • {player['position']} • £{player['price']:.1f}m", style={
            'color': COLORS['text_light'],
            'margin': '0 0 16px 0',
            'fontSize': '14px'
        }),
        html.Div([
            html.Span(metric_label, style={'color': COLORS['text_light'], 'fontSize': '13px'}),
            html.Span(metric_value, style={
                'color': COLORS['primary'],
                'fontWeight': '700',
                'fontSize': '18px',
                'marginLeft': '8px'
            })
        ])
    ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '220px'})


def build_position_breakdown_chart():
    position_stats = df_active[df_active['minutes'] > 450].groupby('position').agg({
        'total_points': 'mean',
        'points_per_million': 'mean',
        'price': 'mean'
    }).round(2).reset_index()

    position_order = ['GKP', 'DEF', 'MID', 'FWD']
    position_stats['position'] = pd.Categorical(position_stats['position'], categories=position_order, ordered=True)
    position_stats = position_stats.sort_values('position')

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Avg Points', x=position_stats['position'], y=position_stats['total_points'],
                         marker_color=COLORS['primary'], text=position_stats['total_points'].round(1), textposition='outside'))
    fig.add_trace(go.Bar(name='Avg Pts/£m (x10)', x=position_stats['position'], y=position_stats['points_per_million'] * 10,
                         marker_color=COLORS['secondary'], text=position_stats['points_per_million'].round(2), textposition='outside'))

    fig.update_layout(barmode='group', template='plotly_white', height=350,
                      margin=dict(t=40, b=40, l=40, r=40),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                      yaxis_title='Value', xaxis_title='Position', font=dict(family='Arial, sans-serif'))
    return fig


# Prepare home page table data
home_value_cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'total_points', 'points_per_million', 'form', 'ownership']
home_table_data = prepare_table_data(df_active[df_active['minutes'] > 1350].nlargest(15, 'points_per_million'), home_value_cols)

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.Span("Fantasy Premier League", style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                        'padding': '6px 12px', 'borderRadius': '6px', 'fontWeight': '800',
                                        'fontSize': '18px', 'marginRight': '12px'}),
                html.Span("Analytics Dashboard", style={'color': 'white', 'fontSize': '20px', 'fontWeight': '600'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Span(f"Data as of {current_gw['name'] if current_gw else 'N/A'}",
                          style={'color': 'rgba(255,255,255,0.8)', 'fontSize': '13px'})
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                  'maxWidth': '1400px', 'margin': '0 auto', 'padding': '0 20px'})
    ], style={'backgroundColor': COLORS['primary'], 'padding': '16px 0', 'position': 'sticky',
              'top': '0', 'zIndex': '1000', 'boxShadow': '0 2px 8px rgba(0,0,0,0.15)'}),

    # Main Content
    html.Div([
        dcc.Tabs(id='main-tabs', value='home', children=[

            # HOME TAB
            dcc.Tab(label='Home', value='home', children=[
                html.Div([
                    html.Div([
                        html.H2("Season Overview", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                        html.P("Key statistics from the 2025/26 FPL season", style={'color': COLORS['text_light']})
                    ], style={'marginBottom': '24px'}),

                    html.Div([
                        html.Div([build_stat_card("Total Managers", f"{total_managers:,}", "Competing worldwide")],
                                 style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                        html.Div([build_stat_card("Current Gameweek", current_gw['name'].replace('Gameweek ', 'GW') if current_gw else "N/A", f"Average: {avg_gw_score} pts")],
                                 style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                        html.Div([build_stat_card("Highest GW Score", f"{highest_gw_score}", "This gameweek", color=COLORS['success'])],
                                 style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                        html.Div([build_stat_card("Next Deadline", next_gw['name'].replace('Gameweek ', 'GW') if next_gw else "N/A",
                                                  datetime.fromisoformat(next_gw['deadline_time'].replace('Z', '+00:00')).strftime('%d %b, %H:%M') if next_gw else "")],
                                 style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}),

                    html.Div([
                        html.H2("Player Spotlights", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                        html.P("Top performers across key metrics", style={'color': COLORS['text_light']})
                    ], style={'marginBottom': '24px'}),

                    html.Div([
                        build_player_spotlight(top_scorer, "Top Scorer", "Total Points", f"{int(top_scorer['total_points'])}" if top_scorer is not None else "N/A"),
                        build_player_spotlight(most_selected, "Most Selected", "Ownership", f"{most_selected['ownership']:.1f}%" if most_selected is not None else "N/A"),
                        build_player_spotlight(best_value, "Best Value", "Points/£m", f"{best_value['points_per_million']:.2f}" if best_value is not None else "N/A"),
                        build_player_spotlight(top_form, "In Form", "Form Rating", f"{top_form['form']:.1f}" if top_form is not None else "N/A"),
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '40px'}),

                    html.Div([
                        html.H2("Position Breakdown", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                        html.P("Average points and value by position", style={'color': COLORS['text_light']})
                    ], style={'marginBottom': '24px'}),

                    html.Div([dcc.Graph(figure=build_position_breakdown_chart(), config={'displayModeBar': False})], style=CARD_STYLE),

                    html.Div([
                        html.H2("Value Watch", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                        html.P("Top 15 value picks (min. 1350 minutes)", style={'color': COLORS['text_light']})
                    ], style={'marginBottom': '24px'}),

                    html.Div([
                        dash_table.DataTable(
                            data=home_table_data,
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric'},
                                {'name': 'Points', 'id': 'total_points', 'type': 'numeric'},
                                {'name': 'Pts/£m', 'id': 'points_per_million', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # DEFCON BONUS TAB
            dcc.Tab(label='Def Con Bonus', value='defcon-bonus', children=[
                html.Div([
                    html.Div([
                        html.H3("Understanding Defensive Contribution Bonuses", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P(["Players earn ", html.Strong("2 bonus points"), " when they achieve ",
                                html.Strong("10+ defensive contributions"), " in a single match."],
                               style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([html.Span("🎯 Target: 10.0 defcon per 90", style={'backgroundColor': COLORS['secondary'],
                                  'color': COLORS['primary'], 'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='bonus-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='bonus-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='bonus-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='bonus-minutes', type='number', value=450, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Defcon Per 90 vs Bonus Threshold", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Red line = 10-point bonus threshold. Players above consistently earn defcon bonuses.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='bonus-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Distance from Bonus Threshold", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("How far above or below the 10-point threshold each player averages.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='bonus-bar')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Defcon Bonus Rankings", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='bonus-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric'},
                                {'name': 'Defcon', 'id': 'defcon', 'type': 'numeric'},
                                {'name': 'Defcon/90', 'id': 'defcon_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'vs Bonus', 'id': 'defcon_vs_bonus', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                                {'name': '% Target', 'id': 'bonus_rate', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{defcon_vs_bonus} >= 0', 'column_id': 'defcon_vs_bonus'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{defcon_vs_bonus} < 0', 'column_id': 'defcon_vs_bonus'}, 'backgroundColor': '#ffebee'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # BONUS CONSISTENCY TAB
            dcc.Tab(label='Def Con Bonus: Consistency', value='bonus-consistency', children=[
                html.Div([
                    # Explanation Card
                    html.Div([
                        html.H3("DefCon Bonus Consistency Analysis", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "This shows how ", html.Strong("consistently"), " players hit the 10+ defcon bonus threshold in individual matches. ",
                            "A player averaging 10 defcon per 90 minutes might be inconsistent (20 one week, 0 the next) vs someone who reliably hits 9-11 every game."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("📊 Based on games with 60+ minutes played", style={'backgroundColor': COLORS['secondary'],
                                      'color': COLORS['primary'], 'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Filters
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='consistency-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='consistency-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='consistency-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Games", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='consistency-games', type='number', value=5, min=1, step=1,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='consistency-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Chart - Hit Rate Distribution
                    html.Div([
                        html.H3("Bonus Hit Rate by Player", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Percentage of qualifying games (60+ mins) where player achieved 10+ defcon.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='consistency-bar')
                    ], style=CARD_STYLE),

                    # Scatter - Hit Rate vs Avg Defcon
                    html.Div([
                        html.H3("Consistency vs Average Output", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Compare hit rate (consistency) against average defcon in qualifying games. Top-right = high output AND consistent.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='consistency-scatter')
                    ], style=CARD_STYLE),

                    # Table
                    html.Div([
                        html.H4("Bonus Consistency Rankings", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='consistency-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric'},
                                {'name': 'Games', 'id': 'qualifying_games', 'type': 'numeric'},
                                {'name': 'Bonus Games', 'id': 'bonus_games', 'type': 'numeric'},
                                {'name': 'Hit Rate %', 'id': 'hit_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Avg Defcon', 'id': 'avg_defcon_qualifying', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Max', 'id': 'max_defcon_game', 'type': 'numeric'},
                                {'name': 'Min', 'id': 'min_defcon_game', 'type': 'numeric'},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{hit_rate} >= 50', 'column_id': 'hit_rate'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{hit_rate} >= 25 && {hit_rate} < 50', 'column_id': 'hit_rate'}, 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{hit_rate} < 25', 'column_id': 'hit_rate'}, 'backgroundColor': '#ffebee'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # DEFENSIVE CONTRIBUTIONS TAB
            dcc.Tab(label='Defensive Contributions', value='defcon', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='defcon-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='defcon-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='defcon-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='defcon-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Actual vs Expected Defensive Contributions", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players above the diagonal are outperforming expectations.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='defcon-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Top Defensive Contributors", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='defcon-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric'},
                                {'name': 'Defcon', 'id': 'defcon', 'type': 'numeric'},
                                {'name': 'Defcon/90', 'id': 'defcon_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xDefcon', 'id': 'expected_defcon', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Diff', 'id': 'defcon_diff', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{defcon_diff} > 0', 'column_id': 'defcon_diff'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{defcon_diff} < 0', 'column_id': 'defcon_diff'}, 'backgroundColor': '#ffebee'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # XG TAB
            dcc.Tab(label='Expected Goals & Assists', value='xg', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xg-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xg-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='xg-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='xg-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Goals Scored vs Expected Goals", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("See who is outperforming and underperforming their expected goals.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='xg-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("xG Differentials", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='xg-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Goals', 'id': 'goals_scored', 'type': 'numeric'},
                                {'name': 'xG', 'id': 'expected_goals', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xG Diff', 'id': 'xg_diff', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Assists', 'id': 'assists', 'type': 'numeric'},
                                {'name': 'xA', 'id': 'expected_assists', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xA Diff', 'id': 'xa_diff', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{xg_diff} < -1', 'column_id': 'xg_diff'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{xg_diff} > 1', 'column_id': 'xg_diff'}, 'backgroundColor': '#fff8e1'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # VALUE TAB
            dcc.Tab(label='Value Analysis', value='value', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='value-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='value-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='value-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='value-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Points vs Price", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Find the best value players by points returned per £1m invested.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='value-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Best Value Picks", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='value-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Points', 'id': 'total_points', 'type': 'numeric'},
                                {'name': 'Pts/£m', 'id': 'points_per_million', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # FORM TAB
            dcc.Tab(label='Form Tracker', value='form', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='form-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='form-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='form-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='form-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Form vs Season Average", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players trending up or down from their season average.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='form-chart')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Form Differentials", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='form-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Season PPG', 'id': 'ppg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Form Diff', 'id': 'form_vs_season', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{form_vs_season} > 1', 'column_id': 'form_vs_season'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{form_vs_season} < -1', 'column_id': 'form_vs_season'}, 'backgroundColor': '#ffebee'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # CLEAN SHEETS TAB
            dcc.Tab(label='Clean Sheets', value='cs', children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cs-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cs-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='cs-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='cs-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Clean Sheets vs Goals Conceded", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Best assets appear in the top-left quadrant (high CS, low GC).", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='cs-chart')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Defensive Asset Rankings", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='cs-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric'},
                                {'name': 'CS', 'id': 'clean_sheets', 'type': 'numeric'},
                                {'name': 'CS/90', 'id': 'cs_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'GC', 'id': 'goals_conceded', 'type': 'numeric'},
                                {'name': 'GC/90', 'id': 'gc_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # FIXTURE DIFFICULTY TAB
            dcc.Tab(label='Fixture Difficulty', value='fixtures', children=[
                html.Div([
                    # Explanation Card
                    html.Div([
                        html.H3("Fixture Difficulty Rating (FDR) Analysis", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Rank players by their team's upcoming fixture difficulty. ",
                            html.Strong("Lower FDR = easier fixtures"), ". ",
                            "FDR ranges from 1 (very easy) to 5 (very hard)."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("📅 Next 5 Gameweeks", style={'backgroundColor': COLORS['secondary'],
                                      'color': COLORS['primary'], 'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Filters
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fdr-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fdr-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max Price", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='fdr-price', min=4, max=16, step=0.5, value=16, marks={i: f'£{i}' for i in [4,6,8,10,12,14,16]})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min Minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='fdr-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Team FDR Chart
                    html.Div([
                        html.H3("Team Fixture Difficulty (Next 5 GWs)", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Teams sorted by average FDR. Green = easy run, Red = tough run.", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='fdr-team-bar')
                    ], style=CARD_STYLE),

                    # Player scatter
                    html.Div([
                        html.H3("Player Value vs Fixture Difficulty", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Find high-value players with easy fixtures. Best picks are top-left (high points, low FDR).", style={'color': COLORS['text_light']}),
                        dcc.Graph(id='fdr-scatter')
                    ], style=CARD_STYLE),

                    # Table
                    html.Div([
                        html.H4("Players Ranked by Fixture Difficulty", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='fdr-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Points', 'id': 'total_points', 'type': 'numeric'},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Avg FDR', 'id': 'avg_fdr_5', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Next 5 Fixtures', 'id': 'fixture_string'},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{avg_fdr_5} <= 2.5', 'column_id': 'avg_fdr_5'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{avg_fdr_5} > 2.5 && {avg_fdr_5} <= 3.5', 'column_id': 'avg_fdr_5'}, 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{avg_fdr_5} > 3.5', 'column_id': 'avg_fdr_5'}, 'backgroundColor': '#ffebee'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),
        ])
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '24px 20px',
              'backgroundColor': COLORS['background'], 'minHeight': 'calc(100vh - 60px)'}),

    # Footer
    html.Div([
        html.P(["Built for analytical Fantasy Premier League decision making • Data from ",
                html.A("Official FPL API", href="https://fantasy.premierleague.com", target="_blank",
                       style={'color': COLORS['secondary']})],
               style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '13px', 'margin': '0'})
    ], style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'textAlign': 'center'})

], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': COLORS['background'], 'margin': '0', 'padding': '0'})


# =============================================================================
# CALLBACKS
# =============================================================================

def filter_data(position, team, max_price, min_minutes, positions_allowed=None):
    filtered = df_active.copy()
    if positions_allowed:
        filtered = filtered[filtered['position'].isin(positions_allowed)]
    if position != 'All':
        filtered = filtered[filtered['position'] == position]
    if team != 'All':
        filtered = filtered[filtered['team_name'] == team]
    filtered = filtered[filtered['price'] <= max_price]
    filtered = filtered[filtered['minutes'] >= min_minutes]
    return filtered


# DEFCON BONUS
@callback(
    [Output('bonus-scatter', 'figure'), Output('bonus-bar', 'figure'), Output('bonus-table', 'data')],
    [Input('bonus-position', 'value'), Input('bonus-team', 'value'), Input('bonus-price', 'value'), Input('bonus-minutes', 'value')]
)
def update_bonus(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=['DEF', 'MID', 'FWD'])
    filtered = filtered.dropna(subset=['defcon_per_90'])

    scatter_fig = px.scatter(filtered, x='price', y='defcon_per_90', color='position', size='minutes',
                             hover_name='web_name', hover_data=['team_name', 'defcon', 'defcon_vs_bonus'],
                             color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    scatter_fig.add_hline(y=10, line_dash="dash", line_color=COLORS['danger'],
                          annotation_text="Bonus Threshold (10)", annotation_position="top right")
    scatter_fig.update_layout(template='plotly_white', height=400, xaxis_title='Price (£m)', yaxis_title='Defcon per 90')

    top_25 = filtered.nlargest(25, 'defcon_per_90')
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_25['web_name'], y=top_25['defcon_vs_bonus'],
        marker_color=[COLORS['success'] if x >= 0 else COLORS['danger'] for x in top_25['defcon_vs_bonus']],
        text=top_25['defcon_vs_bonus'].round(2), textposition='outside'
    ))
    bar_fig.add_hline(y=0, line_color='#333', line_width=2)
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45, yaxis_title='Distance from Threshold', showlegend=False)

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'defcon_vs_bonus', 'bonus_rate', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'defcon_per_90'), cols)

    return scatter_fig, bar_fig, table_data


# BONUS CONSISTENCY
@callback(
    [Output('consistency-bar', 'figure'), Output('consistency-scatter', 'figure'), Output('consistency-table', 'data')],
    [Input('consistency-position', 'value'), Input('consistency-team', 'value'), Input('consistency-price', 'value'), Input('consistency-games', 'value'), Input('consistency-minutes', 'value')]
)
def update_consistency(position, team, max_price, min_games, min_minutes):
    # Filter to players with consistency data
    filtered = df_active[df_active['qualifying_games'].notna()].copy()

    # Apply position filter
    if position != 'All':
        filtered = filtered[filtered['position'] == position]
    else:
        filtered = filtered[filtered['position'].isin(['DEF', 'MID', 'FWD'])]

    # Apply team filter
    if team != 'All':
        filtered = filtered[filtered['team_name'] == team]

    # Apply price filter
    filtered = filtered[filtered['price'] <= max_price]

    # Apply min minutes filter
    filtered = filtered[filtered['minutes'] >= min_minutes]

    # Apply min games filter
    filtered = filtered[filtered['qualifying_games'] >= min_games]

    # Bar chart - Top 25 by hit rate
    top_25 = filtered.nlargest(25, 'hit_rate')

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_25['web_name'],
        y=top_25['hit_rate'],
        marker_color=[COLORS['success'] if x >= 50 else (COLORS['warning'] if x >= 25 else COLORS['danger']) for x in top_25['hit_rate']],
        text=[f"{x:.0f}%" for x in top_25['hit_rate']],
        textposition='outside',
        hovertemplate='%{x}<br>%{customdata[2]} • %{customdata[3]}<br>Hit Rate: %{y:.1f}%<br>Bonus Games: %{customdata[0]}/%{customdata[1]}<extra></extra>',
        customdata=top_25[['bonus_games', 'qualifying_games', 'position', 'team_name']].values
    ))
    bar_fig.add_hline(y=50, line_dash="dash", line_color=COLORS['success'], annotation_text="50% threshold", annotation_position="right")
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Bonus Hit Rate (%)', showlegend=False,
                          yaxis=dict(range=[0, max(top_25['hit_rate'].max() * 1.15, 55) if len(top_25) > 0 else 100]))

    # Scatter - Hit Rate vs Avg Defcon
    scatter_fig = px.scatter(
        filtered,
        x='avg_defcon_qualifying',
        y='hit_rate',
        color='position',
        size='qualifying_games',
        hover_name='web_name',
        hover_data=['team_name', 'price', 'bonus_games', 'qualifying_games'],
        color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    scatter_fig.add_hline(y=50, line_dash="dash", line_color='#999', annotation_text="50% hit rate")
    scatter_fig.add_vline(x=10, line_dash="dash", line_color='#999', annotation_text="Bonus threshold")
    scatter_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title='Avg Defcon (in 60+ min games)',
        yaxis_title='Bonus Hit Rate (%)'
    )

    # Table data
    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'qualifying_games', 'bonus_games', 'hit_rate', 'avg_defcon_qualifying', 'max_defcon_game', 'min_defcon_game', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'hit_rate'), cols)

    return bar_fig, scatter_fig, table_data


# DEFCON
@callback(
    [Output('defcon-scatter', 'figure'), Output('defcon-table', 'data')],
    [Input('defcon-position', 'value'), Input('defcon-team', 'value'), Input('defcon-price', 'value'), Input('defcon-minutes', 'value')]
)
def update_defcon(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=['DEF', 'MID', 'FWD'])
    filtered = filtered.dropna(subset=['defcon_per_90', 'expected_defcon'])

    fig = px.scatter(filtered, x='expected_defcon', y='defcon', color='position', size='minutes',
                     hover_name='web_name', hover_data=['team_name', 'price', 'defcon_per_90'],
                     color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['defcon'].max(), filtered['expected_defcon'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='#999'), name='Expected'))
    fig.update_layout(template='plotly_white', height=400)

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'expected_defcon', 'defcon_diff', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'defcon_per_90'), cols)

    return fig, table_data


# XG
@callback(
    [Output('xg-scatter', 'figure'), Output('xg-table', 'data')],
    [Input('xg-position', 'value'), Input('xg-team', 'value'), Input('xg-price', 'value'), Input('xg-minutes', 'value')]
)
def update_xg(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['expected_goals'])

    fig = px.scatter(filtered, x='expected_goals', y='goals_scored', color='position', size='minutes',
                     hover_name='web_name', hover_data=['team_name', 'price', 'xg_diff'],
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['goals_scored'].max(), filtered['expected_goals'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='#999'), name='Expected'))
    fig.update_layout(template='plotly_white', height=400)

    cols = ['web_name', 'team_name', 'position', 'price', 'goals_scored', 'expected_goals', 'xg_diff', 'assists', 'expected_assists', 'xa_diff', 'ownership']
    table_data = prepare_table_data(filtered.sort_values('xg_diff').head(50), cols)

    return fig, table_data


# VALUE
@callback(
    [Output('value-scatter', 'figure'), Output('value-table', 'data')],
    [Input('value-position', 'value'), Input('value-team', 'value'), Input('value-price', 'value'), Input('value-minutes', 'value')]
)
def update_value(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['points_per_million'])

    fig = px.scatter(filtered, x='price', y='total_points', color='position', size='ownership',
                     hover_name='web_name', hover_data=['team_name', 'points_per_million', 'form'],
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    fig.update_layout(template='plotly_white', height=400)

    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'points_per_million', 'form', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'points_per_million'), cols)

    return fig, table_data


# FORM
@callback(
    [Output('form-chart', 'figure'), Output('form-table', 'data')],
    [Input('form-position', 'value'), Input('form-team', 'value'), Input('form-price', 'value'), Input('form-minutes', 'value')]
)
def update_form(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['form_vs_season'])
    top_form = filtered.nlargest(20, 'form_vs_season')

    fig = px.bar(top_form, x='web_name', y='form_vs_season', color='form_vs_season',
                 color_continuous_scale=['#dc3545', '#ffc107', '#28a745'], hover_data=['team_name', 'form', 'ppg'])
    fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45, coloraxis_showscale=False)

    cols = ['web_name', 'team_name', 'position', 'price', 'form', 'ppg', 'form_vs_season', 'ownership']
    table_data = prepare_table_data(filtered.sort_values('form_vs_season', ascending=False).head(50), cols)

    return fig, table_data


# CLEAN SHEETS
@callback(
    [Output('cs-chart', 'figure'), Output('cs-table', 'data')],
    [Input('cs-position', 'value'), Input('cs-team', 'value'), Input('cs-price', 'value'), Input('cs-minutes', 'value')]
)
def update_cs(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=['GKP', 'DEF'])
    filtered = filtered.dropna(subset=['cs_per_90', 'gc_per_90'])

    fig = px.scatter(filtered, x='gc_per_90', y='cs_per_90', color='team_name', size='minutes',
                     hover_name='web_name', hover_data=['price', 'clean_sheets', 'goals_conceded'])
    fig.update_layout(template='plotly_white', height=400)

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'clean_sheets', 'cs_per_90', 'goals_conceded', 'gc_per_90', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'cs_per_90'), cols)

    return fig, table_data


# FIXTURE DIFFICULTY
@callback(
    [Output('fdr-team-bar', 'figure'), Output('fdr-scatter', 'figure'), Output('fdr-table', 'data')],
    [Input('fdr-position', 'value'), Input('fdr-team', 'value'), Input('fdr-price', 'value'), Input('fdr-minutes', 'value')]
)
def update_fdr(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['avg_fdr_5'])

    # Team bar chart - aggregate by team
    team_fdr = filtered.groupby('team_name').agg({
        'avg_fdr_5': 'first',
        'fixture_string': 'first'
    }).reset_index().sort_values('avg_fdr_5')

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=team_fdr['team_name'],
        y=team_fdr['avg_fdr_5'],
        marker_color=[COLORS['success'] if x <= 2.6 else (COLORS['warning'] if x <= 3.5 else COLORS['danger']) for x in team_fdr['avg_fdr_5']],
        text=[f"{x:.2f}" for x in team_fdr['avg_fdr_5']],
        textposition='outside',
        hovertemplate='%{x}<br>Avg FDR: %{y:.2f}<br>Fixtures: %{customdata}<extra></extra>',
        customdata=team_fdr['fixture_string']
    ))
    bar_fig.add_hline(y=3.0, line_dash="dash", line_color='#999', annotation_text="Avg (3.0)", annotation_position="right")
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Average FDR (Next 5 GWs)', showlegend=False,
                          yaxis=dict(range=[0, 5.5]))

    # Scatter - Total Points vs FDR
    scatter_fig = px.scatter(
        filtered,
        x='avg_fdr_5',
        y='total_points',
        color='position',
        size='form',
        hover_name='web_name',
        hover_data=['team_name', 'price', 'fixture_string'],
        color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    scatter_fig.add_vline(x=3.0, line_dash="dash", line_color='#999')
    scatter_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title='Avg Fixture Difficulty (lower = easier)',
        yaxis_title='Total Points'
    )

    # Table - sorted by FDR (ascending = easiest first)
    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'form', 'ownership', 'avg_fdr_5', 'fixture_string']
    table_data = prepare_table_data(filtered.nsmallest(50, 'avg_fdr_5'), cols)

    return bar_fig, scatter_fig, table_data


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Fantasy Premier League Analytics Dashboard")
    print("="*60)
    print(f"  Current Gameweek: {current_gw['name'] if current_gw else 'N/A'}")
    print(f"  Players loaded: {len(df)}")
    print(f"  Active players: {len(df_active)}")
    print("="*60)
    print("\n  Starting server...")
    print("  Open http://127.0.0.1:8053 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, port=8053)