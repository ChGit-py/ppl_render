"""
Fantasy Premier League Analytics Dashboard
Track Defensive Contributions, Expected Metrics, and Value Picks
"""

import requests
import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, ctx, clientside_callback, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import pickle
import os
import pulp

# =============================================================================
# DATA FETCHING
# =============================================================================

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


def adaptive_min_minutes(base, gw_elapsed):
    """
    Season-aware minutes threshold: early in the season nobody can have the
    minutes a mid-season filter assumes (after GW1 the maximum is ~90, so a
    450-minute default filters out the entire league). Ramp at ~60 useful
    minutes per elapsed GW, floored at 45, capped at the mid-season value.
    Reaches base=200 by GW4 and base=450 by GW8.
    """
    if not gw_elapsed or gw_elapsed <= 0:
        return base  # pre-season: carried full-season stats, keep full filter
    return min(base, max(45, int(gw_elapsed) * 60))


def fetch_bootstrap_data():
    response = requests.get(f"{FPL_BASE_URL}/bootstrap-static/")
    response.raise_for_status()
    return response.json()


def fetch_fixtures():
    """Fetch all fixtures for the season."""
    response = requests.get(f"{FPL_BASE_URL}/fixtures/")
    response.raise_for_status()
    return response.json()


# =============================================================================
# SEASON CONFIG — derived from the API instead of hardcoded
# =============================================================================
#
# The FPL API added a `game_config` block for 2026/27. It exposes:
#   game_config.settings.static_content_url  -> ".../plfpl-production/2026_27/"
#   game_config.scoring                      -> full points table by position
#
# We read the season label, the player-photo path and the set of positions that
# actually earn Defensive Contribution points from there, rather than hardcoding.
#
# NOTE: the DefCon *thresholds* (10 for DEF, 12 for MID/FWD) are still NOT in
# the API anywhere. They remain hardcoded below — this is the only defcon value
# that has to be maintained by hand.

DEFCON_THRESHOLDS = {'DEF': 10, 'MID': 12, 'FWD': 12}

# Fallbacks, used only if the API drops a field we depend on.
_FALLBACK_SEASON_LABEL = '2026/27'
_FALLBACK_PHOTO_PREFIX = 'premierleague26'

# Optional escape hatch: set FPL_PHOTO_PREFIX in the environment (e.g. on
# Render) to force a bucket without redeploying code, if the PL CDN moves.
_PHOTO_PREFIX_OVERRIDE = os.environ.get('FPL_PHOTO_PREFIX', '').strip()

# Shown when every candidate URL 404s, so a missing headshot degrades to a
# neutral silhouette rather than the browser's broken-image icon.
PHOTO_PLACEHOLDER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 110 140'>"
    "<rect width='110' height='140' rx='8' fill='%23efeaf3'/>"
    "<circle cx='55' cy='54' r='23' fill='%23cabdd5'/>"
    "<path d='M16 140c0-24 17-41 39-41s39 17 39 41z' fill='%23cabdd5'/>"
    "</svg>"
)


def _photo_templates(prefix):
    """
    Ordered candidate URL templates for a given season prefix.

    The PL CDN has used several layouts over the years and does not always have
    the new season's bucket populated on day one, so we try in order rather than
    betting the UI on a single guess:
      1. this season's bucket        premierleague26/.../{code}.png
      2. last season's bucket        premierleague25/.../{code}.png
      3. the legacy unversioned path premierleague/.../p{code}.png
    """
    base = "https://resources.premierleague.com"
    templates = []
    if _PHOTO_PREFIX_OVERRIDE:
        templates.append(f"{base}/{_PHOTO_PREFIX_OVERRIDE}/photos/players/110x140/{{code}}.png")
    templates.append(f"{base}/{prefix}/photos/players/110x140/{{code}}.png")
    try:
        prev = f"premierleague{int(prefix.replace('premierleague', '')) - 1:02d}"
        templates.append(f"{base}/{prev}/photos/players/110x140/{{code}}.png")
    except (TypeError, ValueError):
        pass
    templates.append(f"{base}/premierleague/photos/players/110x140/p{{code}}.png")
    # De-duplicate, preserving order
    return list(dict.fromkeys(templates))


SEASON = {
    'label': _FALLBACK_SEASON_LABEL,
    'photo_base': f"https://resources.premierleague.com/{_FALLBACK_PHOTO_PREFIX}/photos/players/110x140/",
    'photo_templates': _photo_templates(_FALLBACK_PHOTO_PREFIX),
    'thresholds': dict(DEFCON_THRESHOLDS),
    'defcon_positions': sorted(DEFCON_THRESHOLDS),
    'positions': ['GKP', 'DEF', 'MID', 'FWD'],
    'outfield_positions': ['DEF', 'MID', 'FWD'],
}


def build_season_config(data):
    """
    Populate the module-level SEASON dict from the bootstrap payload.

    Called once at the top of every core refresh so a season rollover is picked
    up automatically rather than needing a code edit.
    """
    cfg = data.get('game_config', {}) or {}

    # --- Season label, e.g. "2026_27" -> "2026/27" ---
    static_url = (cfg.get('settings', {}) or {}).get('static_content_url', '') or ''
    season_slug = ''
    for part in static_url.rstrip('/').split('/'):
        if len(part) == 7 and part[:4].isdigit() and part[4] == '_':
            season_slug = part
            break
    if season_slug:
        start_year, end_yy = season_slug.split('_')
        SEASON['label'] = f"{start_year}/{end_yy}"
        # Photo bucket follows the season start year: 2026/27 -> premierleague26
        _prefix = f"premierleague{start_year[2:]}"
        SEASON['photo_base'] = (
            f"https://resources.premierleague.com/{_prefix}/photos/players/110x140/"
        )
        SEASON['photo_templates'] = _photo_templates(_prefix)

    # --- Positions, straight from element_types ---
    pos_short = [p['singular_name_short'] for p in data.get('element_types', [])]
    if pos_short:
        SEASON['positions'] = pos_short
        SEASON['outfield_positions'] = [p for p in pos_short if p != 'GKP']

    # --- Which positions actually earn DefCon points ---
    # 2026/27 scoring: {"DEF": 2, "MID": 2, "FWD": 2, "GKP": 0}
    dc_scoring = (cfg.get('scoring', {}) or {}).get('defensive_contribution')
    if isinstance(dc_scoring, dict):
        earners = [p for p, pts in dc_scoring.items() if pts and p in pos_short]
        if earners:
            SEASON['defcon_positions'] = sorted(earners)
            SEASON['thresholds'] = {
                p: DEFCON_THRESHOLDS.get(p, 12) for p in earners
            }

    print(f"  Player photos: {SEASON['photo_templates'][0]}")
    print(f"  Season config: {SEASON['label']} | "
          f"defcon positions {SEASON['defcon_positions']} | "
          f"thresholds {SEASON['thresholds']}")
    return SEASON


def _photo_code(player_or_code):
    """
    Resolve the numeric asset id. Prefers the element's `photo` field (which is
    what FPL itself builds image URLs from, e.g. "223094.jpg") and falls back to
    `code`. They are currently identical, but `photo` is the authoritative one.
    """
    value = player_or_code
    if hasattr(player_or_code, 'get'):
        value = player_or_code.get('photo') or player_or_code.get('code')
    if value is None:
        return None
    try:
        return int(str(value).split('.')[0])
    except (TypeError, ValueError):
        return None


def player_photo_candidates(player_or_code):
    """Ordered list of headshot URLs to try, ending in the local placeholder."""
    code = _photo_code(player_or_code)
    if code is None:
        return [PHOTO_PLACEHOLDER]
    urls = [t.format(code=code) for t in SEASON['photo_templates']]
    urls.append(PHOTO_PLACEHOLDER)
    return urls


def player_photo_url(player_or_code):
    """First-choice headshot URL. Kept for callers that just want a string."""
    return player_photo_candidates(player_or_code)[0]


def player_photo_img(player_or_code, style=None, **kwargs):
    """
    An <img> that walks its fallback list client-side if the CDN 404s.
    The remaining candidates ride along in data-fallbacks; a delegated error
    handler in index_string advances through them.
    """
    urls = player_photo_candidates(player_or_code)
    return html.Img(
        src=urls[0],
        className='player-photo',
        style=style or {},
        **{'data-fallbacks': '||'.join(urls[1:])},
        **kwargs
    )


def calculate_fixture_difficulty(fixtures, teams_df, current_gw, num_gameweeks=None):
    """
    Calculate average fixture difficulty for each team over remaining season fixtures.
    If num_gameweeks is None, uses all remaining fixtures in the season.
    Returns dict of team_id -> {fixtures: [...], avg_fdr: float}
    """
    # Get upcoming gameweeks - all remaining if num_gameweeks is None
    if num_gameweeks is None:
        upcoming_fixtures = [f for f in fixtures if f.get('event') is not None and f['event'] > current_gw]
    else:
        upcoming_gws = list(range(current_gw + 1, current_gw + num_gameweeks + 1))
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

        home_name = teams_df[teams_df['id'] == home_team]['name'].values[0] if home_team in teams_df[
            'id'].values else 'Unknown'
        away_name = teams_df[teams_df['id'] == away_team]['name'].values[0] if away_team in teams_df[
            'id'].values else 'Unknown'

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
        team_fixtures[team_id]['fixture_string'] = ', '.join(team_fixtures[team_id]['opponents'])

    return team_fixtures


def fetch_player_summary(player_id):
    """Full element-summary payload: this-season history AND past seasons."""
    try:
        response = requests.get(f"{FPL_BASE_URL}/element-summary/{player_id}/", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching player {player_id}: {e}")
        return {}


def fetch_player_history(player_id):
    """Fetch individual player's match-by-match history."""
    return fetch_player_summary(player_id).get('history', []) or []


def extract_last_season_prior(summary):
    """
    Per-90 attacking prior from the most recent PAST season in an
    element-summary payload. Only trusted with 900+ minutes (10 full games) —
    below that, last season is itself noise and the position prior is safer.
    Returns {'xg90', 'xa90', 'mins'} or None.
    """
    past = summary.get('history_past') or []
    if not past:
        return None
    last = past[-1]
    try:
        mins = float(last.get('minutes') or 0)
        if mins < 900:
            return None
        xg = float(last.get('expected_goals') or 0)
        xa = float(last.get('expected_assists') or 0)
        if xg == 0 and xa == 0:
            return None  # xG columns absent for that season
        return {'xg90': round(xg / mins * 90, 3),
                'xa90': round(xa / mins * 90, 3),
                'mins': int(mins)}
    except (TypeError, ValueError):
        return None


def calculate_bonus_consistency(player_ids, player_thresholds, min_minutes=1):
    """
    Defensive-contribution hit rate: how often a player actually banks the
    2 DEFCON points (10+ for DEF, 12+ for MID/FWD in a single match).

    MINUTES FILTER — was 60, now 1 (any appearance).
    The 60-minute cutoff was borrowed from clean sheets, where FPL really
    does require 60 minutes. Defensive contribution has NO such rule: hit
    the threshold in 40 minutes off the bench and the points are yours. So
    the old filter applied a scoring rule that does not exist for this stat,
    and silently deleted whole appearances from the denominator — a player
    with three games and one short outing was displayed as 2 games.

    Worse, it biased upward. A short appearance is far more likely to be a
    miss (fewer minutes, fewer defensive actions), so filtering at 60 removed
    misses preferentially and left the hits. Every hit rate in the table was
    too high, and those rates feed p_hit in the projection engine.

    `appearances` and `starts_60` are returned alongside so the split between
    "played" and "played a full game" stays visible rather than hidden inside
    the denominator.
    """
    results = {}
    priors = {}

    def process_player(player_id):
        summary = fetch_player_summary(player_id)
        prior = extract_last_season_prior(summary)
        history = summary.get('history', []) or []
        if not history:
            return player_id, None, prior

        # Every appearance is an opportunity — DEFCON points carry no
        # minutes requirement, unlike clean sheets.
        qualifying_games = [g for g in history if (g.get('minutes') or 0) >= min_minutes]

        if not qualifying_games:
            return player_id, None, prior

        # Count games hitting bonus threshold (position-aware)
        threshold = player_thresholds.get(player_id, 10)
        bonus_games = [g for g in qualifying_games if g.get('defensive_contribution', 0) >= threshold]

        # Calculate stats
        defcon_values = [g.get('defensive_contribution', 0) for g in qualifying_games]
        total_mins = sum((g.get('minutes') or 0) for g in qualifying_games)

        # Full-90 games are the clean sample for "would he hit it if he
        # started", which is what the projection needs — the engine applies
        # minutes risk separately via p60.
        full_games = [g for g in qualifying_games if (g.get('minutes') or 0) >= 60]
        full_hits = [g for g in full_games if g.get('defensive_contribution', 0) >= threshold]
        _mean = (sum(defcon_values) / len(defcon_values)) if defcon_values else 0.0
        _var = (sum((v - _mean) ** 2 for v in defcon_values) / (len(defcon_values) - 1)
                ) if len(defcon_values) > 1 else 0.0

        stats = {
            'qualifying_games': len(qualifying_games),
            'appearances': len(qualifying_games),
            'starts_60': len(full_games),
            'bonus_games_60': len(full_hits),
            'defcon_per_90_games': (sum(defcon_values) / total_mins * 90) if total_mins > 0 else 0,
            'defcon_var': _var,
            'bonus_games': len(bonus_games),
            'hit_rate': (len(bonus_games) / len(qualifying_games)) * 100 if qualifying_games else 0,
            'avg_defcon': sum(defcon_values) / len(defcon_values) if defcon_values else 0,
            'max_defcon': max(defcon_values) if defcon_values else 0,
            'min_defcon': min(defcon_values) if defcon_values else 0,
            'threshold': threshold,
        }
        return player_id, stats, prior

    # Use threading for faster fetching (capped at 8 to limit memory on free tier)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_player, pid): pid for pid in player_ids}
        for future in as_completed(futures):
            player_id, stats, prior = future.result()
            if stats:
                results[player_id] = stats
            if prior:
                priors[player_id] = prior

    return results, priors


def fetch_player_history_batch(player_ids, max_workers=8):
    """
    Fetch match-by-match history for multiple players in parallel.
    Returns dict of player_id -> list of match dicts.
    """
    results = {}

    def _fetch(pid):
        return pid, fetch_player_history(pid)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, pid): pid for pid in player_ids}
        for future in as_completed(futures):
            pid, history = future.result()
            if history:
                results[pid] = history
    return results


# =============================================================================
# MY SQUAD: TEAM ENTRY & PICKS FETCHING
# =============================================================================

def fetch_team_entry(team_id):
    """Fetch manager entry data: name, overall rank, bank, team value, total points."""
    try:
        response = requests.get(f"{FPL_BASE_URL}/entry/{team_id}/", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching team entry {team_id}: {e}")
        return None


def fetch_team_picks(team_id, gw):
    """Fetch a manager's 15 picks for a specific gameweek."""
    try:
        response = requests.get(f"{FPL_BASE_URL}/entry/{team_id}/event/{gw}/picks/", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching picks for team {team_id}, GW {gw}: {e}")
        return None


# =============================================================================
# MINI-LEAGUE RIVAL INTELLIGENCE
# =============================================================================

RIVALS_MAX_ENTRIES = 20  # keep API load and memory bounded for big leagues


def fetch_league_standings(league_id, max_entries=RIVALS_MAX_ENTRIES):
    """
    Standings for any classic mini-league. Returns (league_name, entries)
    where entries is a list of dicts (entry, entry_name, player_name, rank,
    total, event_total), capped at max_entries by rank.
    """
    try:
        r = requests.get(
            f"{FPL_BASE_URL}/leagues-classic/{int(league_id)}/standings/",
            timeout=15)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"Error fetching league {league_id}: {e}")
        return None, []
    name = (payload.get('league') or {}).get('name', f'League {league_id}')
    results = ((payload.get('standings') or {}).get('results') or [])
    if results:
        entries = [{
            'entry': e['entry'], 'entry_name': e.get('entry_name', ''),
            'player_name': e.get('player_name', ''), 'rank': e.get('rank'),
            'total': e.get('total', 0), 'event_total': e.get('event_total', 0),
        } for e in results[:max_entries]]
        return name, entries

    # Pre-season (and brand-new leagues): the API returns an EMPTY standings
    # list until a gameweek has completed — members are listed under
    # new_entries instead, with names split into first/last and no rank yet.
    newbies = ((payload.get('new_entries') or {}).get('results') or [])
    entries = [{
        'entry': e['entry'], 'entry_name': e.get('entry_name', ''),
        'player_name': (f"{e.get('player_first_name', '')} "
                        f"{e.get('player_last_name', '')}").strip(),
        'rank': None, 'total': 0, 'event_total': 0,
    } for e in newbies[:max_entries]]
    return name, entries


def fetch_entry_history(entry_id):
    """Season history: per-gameweek points, overall rank, transfers, chips."""
    try:
        r = requests.get(f"{FPL_BASE_URL}/entry/{int(entry_id)}/history/", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching history for entry {entry_id}: {e}")
        return None


def build_rank_card(history, overall_rank):
    """
    Overall rank with an interactive sparkline of every gameweek so far.
    Better ranks plot higher, so the line rises when you climb. Hover a
    point for that week's rank and points.
    """
    rows = [r for r in ((history or {}).get('current') or []) if r.get('overall_rank')]
    last_gw = rows[-1]['event'] if rows else None
    delta = None
    if len(rows) >= 2:
        moved = rows[-2]['overall_rank'] - rows[-1]['overall_rank']
        delta = html.Span(f"{'up' if moved >= 0 else 'down'} {abs(moved):,}",
                          style={'color': '#00ff87' if moved >= 0 else '#ff5c93', 'fontWeight': '700'})
    best = min(rows, key=lambda r: r['overall_rank']) if rows else None

    spark = None
    if len(rows) >= 2:
        fig = go.Figure(go.Scatter(
            x=[r['event'] for r in rows], y=[r['overall_rank'] for r in rows],
            mode='lines+markers',
            line=dict(color='#00ff87', width=3, shape='spline', smoothing=0.6),
            marker=dict(size=7, color='#04f5ff', line=dict(width=0)),
            customdata=[[r.get('points', 0), r.get('rank') or 0] for r in rows],
            hovertemplate=('GW%{x}<br>Overall rank %{y:,}<br>%{customdata[0]} pts '
                           '(GW rank %{customdata[1]:,})<extra></extra>')))
        fig.update_layout(
            height=90, margin=dict(l=4, r=4, t=6, b=4),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
            xaxis=dict(visible=False, fixedrange=True),
            yaxis=dict(visible=False, fixedrange=True, autorange='reversed'),
            hoverlabel=dict(bgcolor='#ffffff', font=dict(color='#37003c', family=FONT_FAMILY)))
        spark = dcc.Graph(figure=fig, config={'displayModeBar': False},
                          style={'height': '90px', 'width': '100%'})

    return html.Div([
        html.Div([
            html.Div("Overall rank", style={'fontSize': '14px', 'color': 'rgba(255,255,255,0.75)',
                                            'fontWeight': '600'}),
            html.Div(f"{overall_rank:,}" if overall_rank else '\u2014',
                     style={'fontSize': '44px', 'fontWeight': '800', 'lineHeight': '1.05',
                            'color': '#ffffff', 'letterSpacing': '-0.02em'}),
            html.Div([f"After GW{last_gw}" if last_gw else '', ' | ' if delta is not None else '',
                      delta if delta is not None else ''],
                     style={'fontSize': '14px', 'color': 'rgba(255,255,255,0.85)'}),
            html.Div(f"Season best:  {best['overall_rank']:,} (GW{best['event']})" if best else '',
                     style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.6)', 'marginTop': '2px'}),
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px', 'flex': '0 1 auto'}),
        html.Div(spark, style={'flex': '1 1 220px', 'minWidth': '180px', 'alignSelf': 'center'}),
    ], style={'background': COLORS['primary'], 'borderRadius': '12px', 'padding': '22px 24px',
              'marginBottom': '20px', 'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap',
              'alignItems': 'center', 'backgroundImage':
                  'linear-gradient(135deg, #37003c 60%, #4a0a52 100%)'})


def fetch_entry_transfers(entry_id):
    """Every transfer a manager has made this season (public endpoint)."""
    try:
        r = requests.get(f"{FPL_BASE_URL}/entry/{int(entry_id)}/transfers/", timeout=10)
        r.raise_for_status()
        return r.json() or []
    except Exception as e:
        print(f"Error fetching transfers for entry {entry_id}: {e}")
        return []


def estimate_selling_prices(squad_df, transfers, chips_used):
    """
    Selling price for each player, rebuilt from public data.

    The public picks endpoint has no prices at all (they only come from the
    logged-in /my-team/ endpoint), which is why this column read £0.0m. FPL's
    rule: you keep half of any rise, rounded DOWN to the nearest £0.1m; a
    fall comes off in full. The purchase price is the cost on your last
    transfer in for that player; players you've had since your first squad
    were bought at their season-start price. Free Hit transfers are skipped
    because those squads revert. Returns {player_id: (selling, purchase)}.
    """
    fh_gws = {c.get('event') for c in (chips_used or []) if c.get('name') == 'freehit'}
    bought = {}
    for t in sorted(transfers or [], key=lambda t: t.get('time') or ''):
        if t.get('event') in fh_gws:
            continue
        bought[t['element_in']] = int(t['element_in_cost'])          # tenths of £1m
    out = {}
    for _, r in squad_df.iterrows():
        now = int(round(float(r['price']) * 10))
        start = now - int(round(float(r.get('cost_change_start', 0) or 0) * 10))
        buy = bought.get(int(r['id']), start)
        sell = buy + (now - buy) // 2 if now > buy else now
        out[int(r['id'])] = (sell / 10, buy / 10)
    return out


def fetch_entry_chips(entry_id):
    """Chips a manager has already played: list of {name, event}."""
    try:
        r = requests.get(f"{FPL_BASE_URL}/entry/{entry_id}/history/", timeout=10)
        r.raise_for_status()
        return r.json().get('chips', []) or []
    except Exception as e:
        print(f"Error fetching history for entry {entry_id}: {e}")
        return []


def summarise_chip_usage(bootstrap_data):
    """
    SEASON-TO-DATE chip usage across the whole game, from bootstrap-static.

    Each event carries `chip_plays` — plays in THAT gameweek only. The Home
    tab was reading the current gameweek's figures and calling it chip usage,
    which is a weekly snapshot, not a season total. Summing across every
    event gives the cumulative number, and `total_players` is the denominator.

    Note on percentages: with two chip sets per season a manager can play the
    same chip twice, so a share can legitimately exceed the share of managers.
    Read it as plays per team, not as a proportion of teams.

    Returns {'total_players': int, 'rows': [{chip, played, pct}], 'gws': int}.
    """
    if not bootstrap_data:
        return {'total_players': 0, 'rows': [], 'gws': 0}

    totals, gws = Counter(), 0
    for ev in bootstrap_data.get('events', []) or []:
        plays = ev.get('chip_plays') or []
        if plays:
            gws += 1
        for c in plays:
            name = c.get('chip_name')
            if name:
                totals[name] += int(c.get('num_played') or 0)

    total_players = int(bootstrap_data.get('total_players') or 0)
    rows = []
    for name, played in totals.most_common():
        rows.append({
            'chip': chip_name_map.get(name, name),
            'raw_name': name,
            'played': played,
            'pct': round(played / total_players * 100, 1) if total_players else 0.0,
        })
    return {'total_players': total_players, 'rows': rows, 'gws': gws}


def chip_windows(bootstrap_data, current_gw_num):
    """
    Start/stop gameweek for each chip in the CURRENT half of the season.

    Read from bootstrap-static's `chips` array (start_event / stop_event) so
    the two-set structure and any mid-season change are picked up without a
    code edit. Falls back to a 1-19 / 20-38 split if the array is absent.
    """
    windows = {}
    for c in (bootstrap_data or {}).get('chips', []) or []:
        nm = c.get('name') or c.get('chip_name')
        se, ee = c.get('start_event'), c.get('stop_event')
        if nm and se and ee:
            # keep the window that contains, or next follows, the current GW
            prev = windows.get(nm)
            if prev is None or (ee >= current_gw_num and ee < prev[1]):
                windows[nm] = (int(se), int(ee))
    if not windows:
        half = (1, 19) if current_gw_num <= 19 else (20, 38)
        for nm in ('wildcard', 'freehit', 'bboost', '3xc'):
            windows[nm] = half
    return windows


def summarise_league_chips(snapshots, entries, bootstrap_data, current_gw_num, my_id=None):
    """
    Who in YOUR league still holds each chip — the version that affects rank.

    Global usage tells you nothing about your position: a mini-league is
    scored in gaps, so what matters is whether the people above you can
    answer your Free Hit in a blank, not whether 4.7% of the game has used
    one. A rival who has already burned Free Hit is defenceless that week and
    your chip becomes a swing rather than a like-for-like.

    Returns (summary_rows, matrix_rows) — per-chip availability counts, and a
    per-manager grid of held vs spent-in-GW.
    """
    chips = ['wildcard', 'freehit', 'bboost', '3xc']
    windows = chip_windows(bootstrap_data, current_gw_num)

    matrix, held_counts = [], {c: 0 for c in chips}
    for e in entries:
        snap = snapshots.get(e['entry'])
        row = {'manager': e['player_name'] + (' (you)' if my_id and e['entry'] == my_id else ''),
               'rank': e['rank']}
        used = (snap or {}).get('chips_used') or []
        for c in chips:
            lo, hi = windows.get(c, (1, 38))
            played_in_window = [u for u in used
                                if u.get('name') == c and lo <= (u.get('event') or 0) <= hi]
            if played_in_window:
                row[c] = f"GW{played_in_window[-1]['event']}"
            else:
                row[c] = 'held'
                held_counts[c] += 1
        matrix.append(row)

    n = max(len(entries), 1)
    summary = [{
        'chip': chip_name_map.get(c, c),
        'held': held_counts[c],
        'spent': n - held_counts[c],
        'held_pct': round(held_counts[c] / n * 100, 1),
        'window': f"GW{windows.get(c, (1, 38))[0]}-{windows.get(c, (1, 38))[1]}",
    } for c in chips]
    return summary, matrix


def fetch_rival_snapshots(entries, gw, max_workers=8):
    """
    For each league entry, fetch current-GW picks and chip history in
    parallel. Returns dict entry_id -> {picks, entry_history, active_chip,
    chips_used}. Entries whose picks fail (e.g. joined late) are skipped.
    """
    snapshots = {}

    def _one(entry_id):
        picks_data = fetch_team_picks(entry_id, gw)
        chips = fetch_entry_chips(entry_id)
        return entry_id, picks_data, chips

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, e['entry']) for e in entries]
        for future in as_completed(futures):
            entry_id, picks_data, chips = future.result()
            if picks_data and 'picks' in picks_data:
                snapshots[entry_id] = {
                    'picks': picks_data['picks'],
                    'entry_history': picks_data.get('entry_history', {}) or {},
                    'active_chip': picks_data.get('active_chip'),
                    'chips_used': chips,
                }
    return snapshots


def build_league_ownership(snapshots):
    """
    Per-player league-effective-ownership from pick multipliers across all
    sampled squads, plus a plain owner count. Returns (eo_pct, owner_count)
    dicts keyed by player id. Same EO definition as the top-100 sampling:
    bench 0, start 1, captain 2, TC 3 — so a player captained by half the
    league carries more swing than one benched by half of it.
    """
    eo = Counter()
    owners = Counter()
    n = len(snapshots)
    if n == 0:
        return {}, {}
    for snap in snapshots.values():
        for pk in snap['picks']:
            eo[pk['element']] += pk.get('multiplier', 1)
            owners[pk['element']] += 1
    return ({pid: round(v / n * 100, 1) for pid, v in eo.items()},
            dict(owners))


# =============================================================================
# CHIP PLANNER — per-gameweek squad projection
# =============================================================================

def build_gw_fixture_lookup(fixtures_data, teams_df, gws):
    """
    For each gameweek in `gws`, per team: fixture count, average
    attack/defence difficulty (1-5), fixture-specific ATTACKING GOAL
    ENVIRONMENT (form + strengths blended; 1.0 = neutral), and the
    opponent string. Returns
    {gw: {team_id: {'count', 'att_fdr', 'def_fdr', 'att_env', 'opp'}}}.
    """
    lookup = {}
    for gw in gws:
        per_gw = calculate_custom_fdr(fixtures_data, teams_df,
                                      anchor_gw=gw - 1, num_gameweeks=1)
        env_gw = calculate_goal_environment(fixtures_data, teams_df,
                                            anchor_gw=gw - 1, num_gws=1)
        counts = Counter()
        for f in fixtures_data:
            if f.get('event') == gw:
                counts[f['team_h']] += 1
                counts[f['team_a']] += 1
        lookup[gw] = {}
        for tid in set(list(per_gw.keys()) + list(counts.keys()) + list(env_gw.keys())):
            v = per_gw.get(tid, {})
            e = env_gw.get(tid, {})
            lookup[gw][tid] = {
                'count': counts.get(tid, 0),
                'att_fdr': (v.get('att_fdr') if v.get('att_fdr') is not None else 3.0),
                'def_fdr': (v.get('def_fdr') if v.get('def_fdr') is not None else 3.0),
                'att_env': e.get('att_env_avg', 1.0),
                'opp': (f"{e.get('opp_next', '')} ({e.get('venue_next', '')})"
                        if e.get('opp_next') else ''),
            }
    return lookup


def project_player_gw(neutral_base, position, team_id, gw_lookup_for_gw):
    """
    One player's projected points in one specific future GW: the neutral
    (FDR-3, single-fixture) per-GW base scaled by that GW's fixture and
    multiplied by fixture count (0 for a blank, 2 for a double).

    MID/FWD scale by the ATTACKING GOAL ENVIRONMENT (0.55-1.80) — so a
    striker at home to a promoted side gets the real ~1.5x his fixture
    deserves, not a ±12% nudge, and recency alone can no longer outrank the
    fixture of the season. GKP/DEF stay on defensive difficulty (their
    points are CS-driven).
    """
    info = gw_lookup_for_gw.get(team_id)
    if not info or info['count'] == 0:
        return 0.0
    if position in ('GKP', 'DEF'):
        mult = FDR_STEP_RATIO ** (3.0 - info['def_fdr'])
    else:
        mult = float(np.clip(info.get('att_env', 1.0) or 1.0, *ATT_ENV_CLIP))
    return round(neutral_base * mult * info['count'], 2)


FORMATIONS = ((3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
              (4, 5, 1), (5, 3, 2), (5, 4, 1))
SQUAD_QUOTA = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}


def project_pool_for_gw(pool, gw_lookup_for_gw):
    """
    Vectorised version of project_player_gw across a whole player frame —
    the Free Hit optimiser has to price every player in the game for every
    gameweek in the horizon, which is far too many scalar calls.

    `pool` needs columns: position, team, neutral_base. Returns a Series.
    """
    counts = pool['team'].map(lambda t: (gw_lookup_for_gw.get(t) or {}).get('count', 0))
    def_fdr = pool['team'].map(
        lambda t: (gw_lookup_for_gw.get(t) or {}).get('def_fdr', 3.0)).astype(float)
    att_env = pool['team'].map(
        lambda t: (gw_lookup_for_gw.get(t) or {}).get('att_env', 1.0) or 1.0).astype(float)

    is_def_unit = pool['position'].isin(['GKP', 'DEF'])
    mult = np.where(is_def_unit,
                    FDR_STEP_RATIO ** (3.0 - def_fdr),
                    att_env.clip(*ATT_ENV_CLIP))
    return (pool['neutral_base'].fillna(0) * mult * counts).clip(lower=0)


def optimise_free_hit_xi(pool, budget, max_per_club=3, pool_depth=60):
    """
    Best formation-legal XI buildable from the WHOLE player pool for one
    gameweek, inside `budget` and the 3-per-club limit.

    BENCH POLICY: on a Free Hit the bench is filler that should never play,
    so the four bench slots are reserved at the ABSOLUTE cheapest prices in
    the game that satisfy the squad quota for the formation being tried, and
    every remaining penny goes into the XI. (Autosubs do still fire on a
    Free Hit, so a 4.0m bench is a deliberate trade: maximum XI strength in
    exchange for no cover if a starter is a surprise omission.)

    METHOD — price-penalty (Lagrangian) search, then a polish pass.
    The previous version seeded the cheapest legal XI and took the single
    best affordable swap until none improved. That can never buy a premium:
    affording one usually needs TWO simultaneous downgrades, and a
    one-at-a-time search cannot see that move, so it converged on a flat
    mid-priced team every time. Scoring players on (proj - lambda x price)
    and binary-searching lambda until spend meets budget weighs the whole
    team's trade-offs at once, so premiums enter whenever they genuinely
    earn their cost. A short single-swap polish then spends any change left.

    Returns (best_total, best_xi_list, meta) where meta carries xi_spend,
    bench_reserve and formation.
    """
    need = ['position', 'team', 'price', 'proj_gw']
    if pool is None or pool.empty or any(c not in pool.columns for c in need):
        return 0.0, [], {}

    p = pool[pool['proj_gw'].notna() & pool['price'].notna()].copy()
    if p.empty:
        return 0.0, [], {}

    # Bench reserve comes from the FULL pool (cheapest bodies in the game),
    # not from a projection-ranked shortlist.
    cheap_by_pos = {pos: sorted(p[p['position'] == pos]['price'].tolist())
                    for pos in ('GKP', 'DEF', 'MID', 'FWD')}

    def _reserve_for(slots):
        if not cheap_by_pos['GKP']:
            return None
        total = cheap_by_pos['GKP'][0]           # the second keeper
        for pos in ('DEF', 'MID', 'FWD'):
            spare = SQUAD_QUOTA[pos] - slots[pos]
            if spare:
                prices = cheap_by_pos[pos]
                if len(prices) < spare:
                    return None
                total += sum(prices[:spare])
        return round(total, 1)

    cands = {}
    for pos in ('GKP', 'DEF', 'MID', 'FWD'):
        sub = p[p['position'] == pos]
        if sub.empty:
            return 0.0, [], {}
        keep = pd.concat([sub.nlargest(pool_depth, 'proj_gw'),
                          sub.nsmallest(10, 'price')]).drop_duplicates(subset='id')
        cands[pos] = keep.to_dict('records')

    def _greedy(lam, slots, xi_budget):
        """Best XI at a given price penalty: one global pass, respecting
        positional slots and the club cap."""
        allc = [r for pos in slots for r in cands[pos]]
        allc.sort(key=lambda r: -(r['proj_gw'] - lam * r['price']))
        filled = {pos: 0 for pos in slots}
        club, xi, spend = Counter(), [], 0.0
        for r in allc:
            pos = r['position']
            if filled.get(pos, 0) >= slots.get(pos, 0):
                continue
            if club[r['team']] >= max_per_club:
                continue
            xi.append(r); spend += r['price']; club[r['team']] += 1; filled[pos] += 1
            if len(xi) == 11:
                break
        if len(xi) < 11:
            return None
        return xi, round(spend, 1), club

    best_total, best_xi, best_meta = 0.0, [], {}
    formation_table = []   # every shape's score, not just the winner's

    for n_def, n_mid, n_fwd in FORMATIONS:
        slots = {'GKP': 1, 'DEF': n_def, 'MID': n_mid, 'FWD': n_fwd}
        reserve = _reserve_for(slots)
        if reserve is None:
            continue
        xi_budget = budget - reserve
        if xi_budget <= 0:
            continue

        # Binary search the price penalty. lam=0 buys the best XI regardless
        # of cost (usually over budget); a high lam buys the cheapest.
        lo, hi, found = 0.0, 8.0, None
        r0 = _greedy(0.0, slots, xi_budget)
        if r0 and r0[1] <= xi_budget:
            found = r0                       # everything affordable outright
        else:
            for _ in range(40):
                mid = (lo + hi) / 2.0
                res = _greedy(mid, slots, xi_budget)
                if res and res[1] <= xi_budget:
                    found = res
                    hi = mid                 # cheaper than needed: relax
                else:
                    lo = mid                 # still too dear: penalise harder
        if not found:
            continue

        xi, spend, club = found
        xi = list(xi)
        chosen = {r['id'] for r in xi}

        # Polish: spend leftover change on single upgrades.
        for _ in range(25):
            best_swap, best_gain = None, 1e-9
            for i, out_p in enumerate(xi):
                for in_p in cands[out_p['position']]:
                    if in_p['id'] in chosen:
                        continue
                    cost = spend - out_p['price'] + in_p['price']
                    if cost > xi_budget:
                        continue
                    if in_p['team'] != out_p['team'] and club[in_p['team']] >= max_per_club:
                        continue
                    gain = in_p['proj_gw'] - out_p['proj_gw']
                    if gain > best_gain:
                        best_gain, best_swap = gain, (i, in_p, round(cost, 1))
            if best_swap is None:
                break
            i, in_p, cost = best_swap
            out_p = xi[i]
            club[out_p['team']] -= 1; club[in_p['team']] += 1
            chosen.discard(out_p['id']); chosen.add(in_p['id'])
            xi[i] = in_p; spend = cost

        total = sum(r['proj_gw'] for r in xi)
        formation_table.append({'formation': f"{n_def}-{n_mid}-{n_fwd}",
                                'total': round(total, 1),
                                'xi_spend': round(spend, 1),
                                'bench_reserve': reserve})
        if total > best_total:
            best_total, best_xi = total, list(xi)
            best_meta = {'xi_spend': round(spend, 1), 'bench_reserve': reserve,
                         'formation': f"{n_def}-{n_mid}-{n_fwd}"}

    # The margin over the runner-up is the point of returning all of these.
    # A 0.3-pt win means the shape is arbitrary and you should pick whichever
    # you prefer for reasons the optimiser can't see (ceiling, rotation risk,
    # a rival's squad). A 6-pt win is a real read on that week's fixtures.
    formation_table.sort(key=lambda r: -r['total'])
    if best_meta:
        best_meta['formation_table'] = formation_table
        best_meta['margin'] = (round(formation_table[0]['total'] - formation_table[1]['total'], 1)
                               if len(formation_table) > 1 else 0.0)

    return round(best_total, 1), best_xi, best_meta


def estimate_autosub_points(xi, bench):
    """
    Points the bench would have contributed ANYWAY, through autosubs, if you
    did NOT play Bench Boost.

    Bench Boost's real value is bench total MINUS this. The planner used to
    count the full bench total, which overstates the chip by a few points
    every week and nudges you into playing it earlier than you should.

    Approximation: each XI player's chance of not appearing comes from
    expected minutes (p_blank = 1 - min(1, exp_mins/60)); the expected
    number of autosubs is their sum, capped at three outfield subs, and each
    one is credited with the mean projection of the bench players actually
    capable of coming on (i.e. those with a fixture).
    """
    eligible = [b for b in bench if b.get('proj', 0) > 0 and b.get('position') != 'GKP']
    if not eligible:
        return 0.0
    exp_subs = 0.0
    for p in xi:
        mins = p.get('exp_mins')
        if mins is None or (isinstance(mins, float) and np.isnan(mins)):
            continue
        exp_subs += 1.0 - min(1.0, max(float(mins), 0.0) / 60.0)
    exp_subs = min(exp_subs, 3.0, float(len(eligible)))
    mean_bench = float(np.mean([b['proj'] for b in eligible]))
    total_bench = sum(b.get('proj', 0) for b in bench)
    return round(min(exp_subs * mean_bench, total_bench), 2)


def pick_best_xi(players):
    """
    Formation-legal best XI from a 15-man squad list of dicts with keys
    position and proj: exactly 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total.
    Greedy: satisfy minimums with the best available, then fill by
    projection within positional maximums. Returns (xi, bench) lists.
    """
    by_pos = {pos: sorted([p for p in players if p['position'] == pos],
                          key=lambda x: -x['proj'])
              for pos in ('GKP', 'DEF', 'MID', 'FWD')}

    xi = (by_pos['GKP'][:1] + by_pos['DEF'][:3] +
          by_pos['MID'][:2] + by_pos['FWD'][:1])
    chosen = {id(p) for p in xi}
    maxima = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}

    pool = sorted([p for p in players if id(p) not in chosen],
                  key=lambda x: -x['proj'])
    for p in pool:
        if len(xi) >= 11:
            break
        pos_count = sum(1 for q in xi if q['position'] == p['position'])
        if pos_count < maxima[p['position']]:
            xi.append(p)
            chosen.add(id(p))

    bench = [p for p in players if id(p) not in chosen]
    return xi, bench


def get_squad_fixture_flags(team_ids, fixtures_data, current_gw_num, num_gws=5):
    """
    For each team_id, return a string listing any BGW/DGW in the next N GWs.
    Returns dict of team_id -> flag string (e.g. 'GW32: BGW, GW34: DGW')
    """
    upcoming_gws = list(range(current_gw_num + 1, current_gw_num + num_gws + 1))
    result = {}
    for tid in team_ids:
        flags = []
        for gw in upcoming_gws:
            count = sum(
                1 for f in fixtures_data
                if f.get('event') == gw and (f['team_h'] == tid or f['team_a'] == tid)
            )
            if count == 0:
                flags.append(f'GW{gw}: BGW')
            elif count >= 2:
                flags.append(f'GW{gw}: DGW')
        result[tid] = ', '.join(flags) if flags else ''
    return result


# =============================================================================
# RANK GAINS: CAPTAIN & TRANSFER HELPER FUNCTIONS
# =============================================================================

def calculate_home_away_splits(player_histories):
    """
    From match-by-match history, compute home/away PPG for each player.
    Returns dict of player_id -> {home_ppg, away_ppg, home_games, away_games}
    """
    splits = {}
    for pid, matches in player_histories.items():
        home_pts = [m['total_points'] for m in matches if m.get('was_home') and m.get('minutes', 0) >= 60]
        away_pts = [m['total_points'] for m in matches if not m.get('was_home') and m.get('minutes', 0) >= 60]

        splits[pid] = {
            'home_ppg': sum(home_pts) / len(home_pts) if home_pts else 0,
            'away_ppg': sum(away_pts) / len(away_pts) if away_pts else 0,
            'home_games': len(home_pts),
            'away_games': len(away_pts),
        }
    return splits


def _safe_val(v):
    """Convert None / NaN to 0."""
    if v is None:
        return 0
    try:
        if np.isnan(v):
            return 0
    except (TypeError, ValueError):
        pass
    return float(v)


CAPTAIN_WEIGHTS = {
    'form': 0.25,
    'xgi90': 0.20,
    'ppg': 0.15,
    'fdr_inv': 0.15,
    'bps90': 0.10,
    'venue_ppg': 0.10,
    'own_inv': 0.05,
}


def _minmax_norm(series, index):
    """
    Min-max normalize a series to 0-1. NaN inputs become 0 (no evidence = no
    credit). If the column is entirely NaN or constant, return a neutral 0.5
    so it neither helps nor hurts anyone.
    """
    s = pd.to_numeric(series, errors='coerce')
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=index)
    return ((s - lo) / (hi - lo)).fillna(0.0)


def compute_captain_distribution(df, n_sims=4000, seed=17):
    """
    Full points DISTRIBUTION for the next gameweek, per player, by simulation.

    Why this replaces the old haul metric: `haul_pct` was P(2+ goal
    involvements) from a single pooled Poisson. That counts goals and assists
    and nothing else, so a defender who returns clean sheet + DEFCON +
    appearance + 3 bonus — 11 points, a genuine captain haul — read as ~0%.
    Chase mode was structurally blind to defenders, keepers and bonus-heavy
    players. The threshold was wrong too: one goal plus bonus already puts a
    midfielder near 10, so "2 involvements" was never the right bar.

    This simulates every scoring route instead, drawing each component from
    its own distribution and summing:

      appearance   deterministic given minutes drawn
      goals        Poisson, assists Poisson
      clean sheet  Bernoulli
      DEFCON       Bernoulli (negative-binomial hit prob, computed upstream)
      conceded     Poisson, -1 per 2 for GKP/DEF
      saves        Poisson / 3 for GKP
      bonus        Binomial(3, m/3) — mean-matched, correct 0-3 support

    The rates are INVERTED FROM THE PROJECTION'S OWN COMPONENTS (xp_goals,
    xp_cs, ...), so the simulated mean reconciles with proj_pts_next by
    construction. There is no second model to drift out of step.

    Returns a DataFrame with p_10, p_15 (percentages) and sim_mean.
    """
    idx = df.index
    need = ['xp_goals', 'xp_assists', 'xp_cs', 'xp_defcon', 'xp_saves', 'xp_bonus', 'xp_gc']
    if any(c not in df.columns for c in need):
        return pd.DataFrame({'p_10': pd.Series(np.nan, index=idx),
                             'p_15': pd.Series(np.nan, index=idx),
                             'sim_mean': pd.Series(np.nan, index=idx)})

    rng = np.random.default_rng(seed)
    pos = df['position']
    num = lambda c, d=0.0: pd.to_numeric(df[c], errors='coerce').fillna(d).values

    gpts = pos.map(GOAL_POINTS).fillna(4).values.astype(float)
    cpts = pos.map(CS_POINTS).fillna(0).values.astype(float)

    # Invert expected component points back into rates
    lam_g = np.clip(num('xp_goals') / np.maximum(gpts, 1e-9), 0, 5)
    lam_a = np.clip(num('xp_assists') / ASSIST_POINTS, 0, 5)
    p_cs = np.clip(np.where(cpts > 0, num('xp_cs') / np.maximum(cpts, 1e-9), 0.0), 0, 1)
    p_dc = np.clip(num('xp_defcon') / DEFCON_POINTS, 0, 1)
    lam_sv = np.clip(num('xp_saves') * 3.0, 0, 15)
    m_bon = np.clip(num('xp_bonus'), 0, 3)
    lam_gc = np.clip(-num('xp_gc') * 2.0, 0, 8)

    mins = np.clip(num('exp_mins_next', 60.0), 0, 90)
    p_start = np.clip(mins / 90.0, 0, 1)          # P(plays 60+)
    p_cameo = np.clip((1 - p_start) * 0.35, 0, 1)  # some of the rest appear briefly
    is_def_unit = pos.isin(['GKP', 'DEF']).values
    is_gk = (pos == 'GKP').values

    p_played = np.clip(p_start + p_cameo, 1e-6, 1.0)
    p_st = np.clip(p_start, 1e-6, 1.0)

    # The xp_* components are UNCONDITIONAL expectations — expected minutes
    # are already baked into each one. The draws below are gated on actually
    # playing, so the rates must first be made CONDITIONAL by dividing out
    # the gate probability. Without this, minutes get counted twice and a
    # player on 45 expected minutes keeps only half his component value.
    lam_g_c = lam_g / p_played
    lam_a_c = lam_a / p_played
    m_bon_c = np.clip(m_bon / p_played, 0, 3)
    p_cs_c = np.clip(p_cs / p_st, 0, 1)
    p_dc_c = np.clip(p_dc / p_st, 0, 1)
    lam_sv_c = lam_sv / p_st
    lam_gc_c = lam_gc / p_st

    N, P = n_sims, len(idx)
    started = rng.random((N, P)) < p_start
    cameo = (~started) & (rng.random((N, P)) < p_cameo)
    played = started | cameo

    pts = np.where(started, 2.0, np.where(cameo, 1.0, 0.0))
    pts += rng.poisson(np.broadcast_to(lam_g_c, (N, P))) * gpts * played
    pts += rng.poisson(np.broadcast_to(lam_a_c, (N, P))) * ASSIST_POINTS * played
    pts += (rng.random((N, P)) < p_cs_c) * cpts * started
    pts += (rng.random((N, P)) < p_dc_c) * DEFCON_POINTS * started
    pts += np.where(is_gk, rng.poisson(np.broadcast_to(np.maximum(lam_sv_c, 1e-9),
                                                      (N, P))) // 3, 0) * started
    conceded = rng.poisson(np.broadcast_to(np.maximum(lam_gc_c, 1e-9), (N, P)))
    pts -= np.where(is_def_unit, conceded // 2, 0) * started
    pts += rng.binomial(3, np.clip(m_bon_c / 3.0, 0, 1), size=(N, P)) * played

    return pd.DataFrame({
        'p_10': pd.Series((pts >= 10).mean(axis=0) * 100, index=idx).round(1),
        'p_15': pd.Series((pts >= 15).mean(axis=0) * 100, index=idx).round(1),
        'sim_mean': pd.Series(pts.mean(axis=0), index=idx).round(2),
    })


def compute_captain_gain(df):
    """
    Expected points gained on the AVERAGE MANAGER by captaining a player.

    Captaincy is the one decision where raw expected points is the wrong
    ranking. Doubling is a constant multiplier, so ranking by 2 x xP gives
    exactly the same order as xP — the armband changes nothing about who is
    "best". What it changes is what you gain RELATIVE to everyone else, and
    that depends on effective ownership.

    If EO% of the field effectively owns a player (captaincy counted twice),
    your net from captaining him is (2 - EO/100) x his points. Captain a
    90% EO player and you bank 1.1x his score against the field; captain a
    15% differential and you bank 1.85x. That is the mini-league lever, and
    top_eo was sitting in the table unused by any ranking.
    """
    proj = pd.to_numeric(df.get('proj_pts_next'), errors='coerce')
    eo = pd.to_numeric(df.get('top_eo'), errors='coerce')
    if eo is None or eo.isna().all():
        eo = pd.to_numeric(df.get('ownership'), errors='coerce')
    eo = eo.fillna(0).clip(0, 200)
    return ((2.0 - eo / 100.0).clip(lower=0.1) * proj).round(2)


def compute_captain_scores(df):
    """
    Vectorized, normalized captain score on a 0-100 scale.

    The old model summed raw values on wildly different scales (season-total
    xGI vs 0-10 form vs a 0-1 ownership term) with fudge multipliers trying to
    rebalance them — so the stated weights didn't mean what they said. Here
    every input is min-max normalized across the pool first, so the weights
    are true relative importances:

      Form 25% | xGI per 90 20% | PPG 15% | Fixture ease 15%
      BPS/90 10% | Venue-specific PPG 10% | Differential 5%

    Two multipliers then discount the weighted sum:
      - availability (chance_of_playing_next_round; 100% when unflagged)
      - start security (recent start rate, floored at 0.5 so squad rotation
        dampens rather than erases a score; neutral until histories load)

    Per-90 xGI replaces season-total xGI so minutes played no longer
    dominates the attacking-threat term. Attack-specific FDR is used when
    available, falling back to FPL's generic FDR.
    """
    idx = df.index

    fdr_source = df['next_att_fdr'] if 'next_att_fdr' in df.columns else df['next_fdr']
    fdr_inv = 6 - pd.to_numeric(fdr_source, errors='coerce').fillna(3)

    components = {
        'form': _minmax_norm(df['form'], idx),
        'xgi90': _minmax_norm(df['xgi_per_90'], idx),
        'ppg': _minmax_norm(df['ppg'], idx),
        'fdr_inv': _minmax_norm(fdr_inv, idx),
        'bps90': _minmax_norm(df['bps_per_90'], idx),
        'venue_ppg': _minmax_norm(df['venue_ppg'], idx) if 'venue_ppg' in df.columns
                     else pd.Series(0.5, index=idx),
        'own_inv': _minmax_norm(100 - pd.to_numeric(df['ownership'], errors='coerce').fillna(50), idx),
    }

    raw = sum(CAPTAIN_WEIGHTS[k] * v for k, v in components.items())
    score = raw * 100

    # Availability multiplier — a 25% flag quarters the score
    if 'avail_pct' in df.columns:
        avail = pd.to_numeric(df['avail_pct'], errors='coerce').fillna(100) / 100
        score = score * avail

    # Start-security damping — rotation risks get discounted, not erased.
    # NaN (no history yet / not a candidate) stays neutral at 1.0.
    if 'start_rate' in df.columns:
        sec = pd.to_numeric(df['start_rate'], errors='coerce') / 100
        sec = sec.clip(lower=0.5, upper=1.0).fillna(1.0)
        score = score * sec

    return score.round(1)


# =============================================================================
# EXPECTED POINTS PROJECTION ENGINE
# =============================================================================

# FPL 2025/26 scoring values used by the projection (position-keyed)
GOAL_POINTS = {'GKP': 10, 'DEF': 6, 'MID': 5, 'FWD': 4}
CS_POINTS   = {'GKP': 4,  'DEF': 4, 'MID': 1, 'FWD': 0}
ASSIST_POINTS = 3
DEFCON_POINTS = 2

# How strongly fixture difficulty scales output. FDR 3 is neutral; each step
# away moves output by this fraction (FDR 1 → ×1.24 attack, FDR 5 → ×0.76).
# Geometric ratio per FDR step (was linear ±12%, which compressed fixture
# differences so hard a dream fixture could never overturn a modest baseline
# gap). FDR 1 → ×1.39, 2 → ×1.18, 3 → ×1.00, 4 → ×0.85, 5 → ×0.72.
# Tunable via environment without code changes — the Model Lab's parameter
# sweep tells you what to set them to once enough gameweeks are logged.
FDR_STEP_RATIO = float(os.environ.get('FPL_FDR_RATIO', '1.18'))
SHRINK_K = float(os.environ.get('FPL_SHRINK_K', '450'))
FDR_SENSITIVITY = FDR_STEP_RATIO - 1.0  # legacy alias

# Team-level form shrinkage (matches, not minutes): the weight given to a
# team's observed recent rate is n / (n + k), so k is the number of matches
# at which form and the static strength rating carry equal weight. k=6 gives
# 2 games ~25%, 6 games 50%, 20 games ~77% (capped by form_weight). Replaces
# a ramp that hit full weight after three games and let tiny samples swing
# expected goals conceded by ~40%.
TEAM_FORM_SHRINK_K = float(os.environ.get('FPL_TEAM_FORM_K', '6'))

# Backtest fetch limits. Render terminates long HTTP requests, so the
# history fetch has to fit inside one — hence a wall-clock budget, a
# chunk size that lets partial results survive, and a minutes floor that
# keeps the player set small. Press the button again to fetch more.
BACKTEST_FETCH_BUDGET = float(os.environ.get('FPL_BACKTEST_BUDGET', '45'))
BACKTEST_CHUNK = int(os.environ.get('FPL_BACKTEST_CHUNK', '120'))
BACKTEST_MIN_MINUTES = int(os.environ.get('FPL_BACKTEST_MIN_MINS', '90'))


def _fdr_mult(fdr_series, index, invert=False):
    """Geometric fixture multiplier centred on FDR 3 (ratio per step).
    invert=True for 'harder = more' quantities (goals conceded). Robust to
    a missing column (None / scalar) — e.g. replay frames that only carry
    next-GW fixture data — which resolves to a neutral 1.0 multiplier."""
    if fdr_series is None or np.isscalar(fdr_series):
        return pd.Series(1.0, index=index)
    fdr = pd.to_numeric(fdr_series, errors='coerce').fillna(3.0)
    delta = (fdr - 3.0) if invert else (3.0 - fdr)
    return pd.Series(np.power(FDR_STEP_RATIO, delta), index=fdr.index).reindex(index).fillna(1.0)


def _shrink_per90(obs90, minutes, prior90, k=None):
    """
    Empirical-Bayes shrinkage of a per-90 rate toward a prior:
    shrunk = (observed_total + prior_rate x k) / (minutes + k), in per-90
    space. With 90 minutes played the estimate is ~85% prior; by ~450
    minutes the observed data dominates; by 1000+ shrinkage is negligible.
    This is what stops one hot opening game projecting like a season.
    """
    if k is None:
        k = SHRINK_K
    obs = pd.to_numeric(obs90, errors='coerce')
    mins = pd.to_numeric(minutes, errors='coerce').fillna(0).clip(lower=0)
    pri = pd.to_numeric(prior90, errors='coerce').fillna(0)
    return ((obs.fillna(pri) * mins + pri * k) / (mins + k)).fillna(pri)


def _position_prior_per90(df, stat_total_col):
    """
    Pooled position-level per-90 prior computed from the whole league's
    season totals: sum(stat) / sum(minutes) x 90 per position. Pooling
    across ~150 players per position makes even opening-weekend data a
    stable prior.
    """
    mins = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
    stat = pd.to_numeric(df[stat_total_col], errors='coerce').fillna(0) \
        if stat_total_col in df.columns else pd.Series(0.0, index=df.index)
    grp = pd.DataFrame({'pos': df['position'], 'stat': stat, 'mins': mins}).groupby('pos')
    agg = grp.sum()
    rate = (agg['stat'] / agg['mins'].replace(0, np.nan) * 90).fillna(0)
    return df['position'].map(rate).fillna(0)


def compute_expected_points(df, gw_elapsed=38, priors=None, components_out=None):
    """
    Per-player projected FPL points for the next gameweek (`proj_pts_next`),
    the next five (`proj_pts_5`) and next eight (`proj_pts_8`), plus haul
    probability, built strictly from data already in the frame.

    SHRINKAGE (v2): every attacking per-90 rate is shrunk toward a prior
    before use — the player's OWN last-season per-90s when available (900+
    minutes last season, harvested from the same element-summary calls Phase
    2 already makes), otherwise the pooled position-level rate. Weighting is
    minutes-proportional (k=450), so early-season projections stop being
    one-game extrapolations and the shrinkage fades to nothing by
    mid-season automatically. Defensive rates (CS, GC) shrink toward
    position priors only — team context changes too much across seasons for
    individual defensive priors to be safe.

    Components (per GW, under real FPL scoring): appearance, goals (xG/90 x
    position value), assists, clean sheets, goals-conceded penalty, DEFCON
    (measured hit rate when present), keeper saves, realised bonus rate —
    all gated by expected minutes (recent start rate x availability) and
    scaled by attack/defence-specific fixture difficulty.

    `proj_pts_5` / `proj_pts_8` re-use the per-GW model with horizon-average
    FDRs x fixture counts (DGW/BGW aware). `haul_pct` is the Poisson
    probability of 2+ goal involvements next GW — the ceiling metric for
    chase-mode captaincy.

    Returns (exp_mins, proj_next, proj_5, proj_8, haul_pct).
    """
    idx = df.index
    n = lambda col: pd.to_numeric(df[col], errors='coerce') if col in df.columns \
        else pd.Series(np.nan, index=idx)

    pos = df['position']
    priors = priors or {}
    mins_raw = n('minutes').fillna(0)

    # --- Shrunk attacking rates ------------------------------------------
    pos_xg90 = _position_prior_per90(df, 'expected_goals')
    pos_xa90 = _position_prior_per90(df, 'expected_assists')
    ind_xg90 = df['id'].map(lambda i: (priors.get(i) or {}).get('xg90')) \
        if 'id' in df.columns else pd.Series(np.nan, index=idx)
    ind_xa90 = df['id'].map(lambda i: (priors.get(i) or {}).get('xa90')) \
        if 'id' in df.columns else pd.Series(np.nan, index=idx)
    prior_xg90 = pd.to_numeric(ind_xg90, errors='coerce').fillna(pos_xg90)
    prior_xa90 = pd.to_numeric(ind_xa90, errors='coerce').fillna(pos_xa90)

    xg90s = _shrink_per90(n('xg_per_90'), mins_raw, prior_xg90)

    # Penalty duty uplift: a first-choice taker carries ~0.10 xG/90 of
    # near-deterministic volume (~0.13 team pens/match x 0.79 xG each) that
    # position-prior shrinkage otherwise dilutes to nothing. Deliberately
    # conservative at +0.06 (not the full 0.10) because established takers'
    # observed and prior rates already contain the pens they took —
    # the uplift mainly credits NEW takers the data hasn't caught up with.
    if 'pen_rank' in df.columns:
        pen_rank = pd.to_numeric(df['pen_rank'], errors='coerce')
        pen_uplift = pen_rank.map({1: 0.06, 2: 0.015}).fillna(0.0)
        xg90s = xg90s + pen_uplift
    xa90s = _shrink_per90(n('xa_per_90'), mins_raw, prior_xa90)
    bonus90s = _shrink_per90(n('bonus_per_90'), mins_raw, _position_prior_per90(df, 'bonus'))

    # Defensive rates: position priors only
    cs90s = _shrink_per90(n('cs_per_90'), mins_raw, _position_prior_per90(df, 'clean_sheets')).clip(0, 0.8)
    gc90s = _shrink_per90(n('gc_per_90'), mins_raw, _position_prior_per90(df, 'goals_conceded'))

    saves_obs90 = (n('saves').fillna(0) / mins_raw.replace(0, np.nan) * 90)
    saves90s = _shrink_per90(saves_obs90, mins_raw, _position_prior_per90(df, 'saves'))

    # --- Expected minutes -------------------------------------------------
    avail = n('avail_pct').fillna(100).clip(0, 100) / 100

    season_share = (mins_raw / (max(int(gw_elapsed), 1) * 90)).clip(0, 1)
    recent_share = (n('recent_minutes_pct') / 100).clip(0, 1)
    share = recent_share.fillna(season_share)

    p60_fallback = (share * 1.05).clip(0, 1)
    p60 = (n('start_rate') / 100).clip(0, 1).fillna(p60_fallback) * avail
    p_any = (share * 1.15 + 0.05).clip(0, 1).where(share > 0, 0) * avail
    p_any = pd.concat([p_any, p60], axis=1).max(axis=1)

    exp90 = share * avail
    exp_mins = (exp90 * 90).round(0)

    appearance_pts = 2 * p60 + 1 * (p_any - p60)

    # --- Per-GW component model ------------------------------------------
    def per_gw(att_fdr_col, def_fdr_col, att_env_col=None, cs_prob_col=None, collect=False):
        # Attacking side: prefer the goal-environment ratio (expected goals
        # for this fixture / league average) — it spreads real fixtures
        # (promoted side at home ~1.5x, top defence away ~0.7x) where the
        # old linear FDR multiplier compressed everything into ±12%.
        att_mult = None
        if att_env_col and att_env_col in df.columns:
            env = pd.to_numeric(df[att_env_col], errors='coerce')
            if env.notna().any():
                att_mult = env.clip(*ATT_ENV_CLIP).fillna(1.0)
        if att_mult is None:
            att_mult = _fdr_mult(df.get(att_fdr_col), idx)
        def_mult_cs = _fdr_mult(df.get(def_fdr_col), idx)
        def_mult_gc = _fdr_mult(df.get(def_fdr_col), idx, invert=True)

        goal_pts = xg90s * exp90 * att_mult * pos.map(GOAL_POINTS).fillna(4)
        assist_pts = xa90s * exp90 * att_mult * ASSIST_POINTS

        # Clean sheets: prefer the team-level Poisson probability
        # (P(CS) = exp(-lambda), lambda from opponent attack x own defence,
        # blended with market odds when a key is set) computed by
        # calculate_expected_clean_sheets and mapped onto the frame.
        #
        # The fallback below is the ORIGINAL linear FDR ramp, kept only for
        # frames that lack the column — notably Model Lab replay frames built
        # from FEATURE_COLS logged before this change. Previously the ramp was
        # the only path, which meant the xCS page and the Proj column were
        # quoting two different models and disagreeing with each other.
        cs_prob = None
        if cs_prob_col and cs_prob_col in df.columns:
            _team_cs = pd.to_numeric(df[cs_prob_col], errors='coerce')
            if _team_cs.notna().any():
                cs_prob = _team_cs.clip(0, 0.90)
        if cs_prob is None:
            cs_fixture = (0.50 - 0.08 * (n(def_fdr_col).fillna(3.0) - 1)
                          ).clip(0.05, 0.55) * def_mult_cs.clip(0.8, 1.2)
            cs_prob = (0.5 * cs90s + 0.5 * cs_fixture).clip(0, 0.75)
        else:
            # Fill any team the CS model couldn't price with the old blend
            cs_fixture = (0.50 - 0.08 * (n(def_fdr_col).fillna(3.0) - 1)
                          ).clip(0.05, 0.55) * def_mult_cs.clip(0.8, 1.2)
            cs_prob = cs_prob.fillna((0.5 * cs90s + 0.5 * cs_fixture).clip(0, 0.75))
        cs_pts = cs_prob * p60 * pos.map(CS_POINTS).fillna(0)

        is_def_unit = pos.isin(['GKP', 'DEF'])
        gc_pts = (-0.5 * gc90s * exp90 * def_mult_gc).where(is_def_unit, 0)

        # --- DEFCON: modelled as a COUNT, not a coin flip --------------
        # Defensive contribution is a tally of tackles, blocks, clearances,
        # interceptions and recoveries. The old code counted binary hits and
        # shrank that toward a positional average, which threw away the one
        # thing that matters — how close a player's actual volume is to his
        # threshold. A midfielder averaging 5.0 with a best of 6 against a
        # threshold of 12 was still credited ~57% of the positional rate,
        # despite never having come close in any game.
        #
        # Instead: take his shrunk DEFCON rate per 90 as lambda and read the
        # probability straight off a count distribution. Negative binomial
        # rather than Poisson because these counts are overdispersed — game
        # state clusters defensive actions — with the dispersion estimated
        # from the pooled variance-to-mean ratio of the squad itself.
        #
        # p_hit is deliberately conditional on PLAYING A FULL GAME. Minutes
        # risk is applied once, by the p60 multiplier below; folding it into
        # p_hit as well would penalise rotation players twice.
        threshold = n('bonus_threshold').fillna(10)

        dc90_obs = n('defcon_per_90_games')
        dc90 = dc90_obs.fillna(n('defcon_per_90')).fillna(0).clip(lower=0)
        dc90 = _shrink_per90(dc90, mins_raw, _position_prior_per90(df, 'defensive_contribution'))

        # Pooled overdispersion: var/mean across players with a real sample.
        _vm = pd.DataFrame({'v': n('defcon_var'), 'm': dc90,
                            'q': n('qualifying_games').fillna(0)})
        _vm = _vm[(_vm['q'] >= 3) & (_vm['m'] > 0) & _vm['v'].notna() & (_vm['v'] > 0)]
        vmr = float(np.clip((_vm['v'] / _vm['m']).median(), 1.0, 6.0)) if len(_vm) >= 10 else 2.0

        def _p_at_least(lam, thr, ratio):
            """P(X >= thr) for X negative-binomial with mean lam and
            variance ratio x lam. Falls back to Poisson at ratio ~ 1."""
            lam = np.asarray(lam, dtype=float).clip(1e-6, 60)
            thr = np.asarray(thr, dtype=float).clip(1, 60)
            if ratio <= 1.02:
                r = np.full_like(lam, 1e6)
            else:
                r = lam / (ratio - 1.0)
            pr = r / (r + lam)                      # P(success) per NB trial
            k = np.floor(thr).astype(int)
            # P(X < k) by summing the pmf up to k-1, vectorised over players
            cdf = np.zeros_like(lam)
            term = pr ** r                          # pmf at 0
            kmax = int(k.max())
            for i in range(kmax):
                cdf = np.where(i < k, cdf + term, cdf)
                term = term * (r + i) / (i + 1.0) * (1.0 - pr)
            return np.clip(1.0 - cdf, 0.0, 1.0)

        p_model = pd.Series(_p_at_least(dc90.values, threshold.values, vmr), index=idx)

        # Empirical check, from FULL games only — that is the sample which
        # actually answers "does he hit it when he starts".
        g60 = n('starts_60').fillna(0).clip(lower=0)
        h60 = n('bonus_games_60').fillna(0).clip(lower=0)
        # Empirical Bayes with the player's OWN rate-based estimate as the
        # prior (k=4 games), rather than a positional average that ignores
        # whether he is anywhere near the threshold.
        p_hit = ((h60 + p_model * 4.0) / (g60 + 4.0)).clip(0, 1)
        p_hit = p_hit.where(dc90 > 0, 0.0).fillna(0.0)

        defcon_pts = DEFCON_POINTS * p_hit * p60

        save_pts = (saves90s / 3 * exp90).where(pos == 'GKP', 0)
        bonus_pts = (bonus90s * exp90).clip(0, 3)

        total = (appearance_pts + goal_pts + assist_pts + cs_pts +
                 gc_pts + defcon_pts + save_pts + bonus_pts)
        # The eight components are summed and thrown away, which makes a
        # projection impossible to interrogate from outside: an 8.8 built
        # from a modest rate x a 1.6 fixture multiplier is indistinguishable
        # from an 8.5 rate barely scaled, and they need opposite fixes.
        if collect and components_out is not None:
            components_out.update({
                'xp_appear': appearance_pts.round(2),
                'xp_goals': goal_pts.round(2),
                'xp_assists': assist_pts.round(2),
                'xp_cs': cs_pts.round(2),
                'xp_gc': gc_pts.round(2),
                'xp_defcon': defcon_pts.round(2),
                'xp_saves': save_pts.round(2),
                'xp_bonus': bonus_pts.round(2),
                'xp_att_mult': pd.Series(att_mult, index=idx).round(3)
                               if att_mult is not None else pd.Series(1.0, index=idx),
                'xp_cs_prob_used': cs_prob.round(3),
                'xp_p60': p60.round(3),
            })
        return total

    proj_next = per_gw('next_att_fdr', 'next_def_fdr', att_env_col='att_env_next',
                       cs_prob_col='cs_prob_next', collect=True).round(2)

    horizon_per_gw = per_gw('att_fdr_5', 'def_fdr_5', att_env_col='att_env_5',
                            cs_prob_col='cs_prob_5')
    fixture_count = n('fixture_count').fillna(5).clip(0, 10)
    proj_5 = (horizon_per_gw * fixture_count).round(1)
    fixture_count_8 = n('fixture_count_8').fillna(8).clip(0, 16)
    proj_8 = (horizon_per_gw * fixture_count_8).round(1)

    # --- Haul probability: P(2+ goal involvements) next GW ----------------
    if 'att_env_next' in df.columns and pd.to_numeric(df['att_env_next'], errors='coerce').notna().any():
        att_mult_next = pd.to_numeric(df['att_env_next'], errors='coerce').clip(*ATT_ENV_CLIP).fillna(1.0)
    else:
        att_mult_next = _fdr_mult(df.get('next_att_fdr'), idx)
    lam_neutral = ((xg90s + xa90s) * exp90).clip(lower=0)
    lam_i = (lam_neutral * att_mult_next).clip(lower=0)
    haul_pct = ((1 - np.exp(-lam_i) * (1 + lam_i)) * 100).round(1)

    return exp_mins, proj_next, proj_5, proj_8, haul_pct, lam_neutral.round(3)


def calculate_minutes_security(player_histories, window=6):
    """
    From match-by-match history, compute recent start rate and share of
    available minutes over the last `window` matches. The biggest source of
    FPL point loss isn't bad picks — it's benched/rotated picks.
    Returns dict of player_id -> {start_rate, recent_minutes_pct, recent_games}
    """
    security = {}
    for pid, matches in player_histories.items():
        recent = sorted(matches, key=lambda m: (m.get('round') or 0))[-window:]
        if not recent:
            continue
        total_mins = sum(m.get('minutes', 0) for m in recent)
        # The API ships a `starts` field; fall back to a 60-minute heuristic
        starts = sum(
            1 for m in recent
            if (m.get('starts') if m.get('starts') is not None else (m.get('minutes', 0) >= 60))
        )
        n = len(recent)
        security[pid] = {
            'start_rate': round(starts / n * 100, 1),
            'recent_minutes_pct': round(total_mins / (n * 90) * 100, 1),
            'recent_games': n,
        }
    return security


def calculate_schedule_adjusted_form(player_histories, teams_df, window=4,
                                     min_match_minutes=45):
    """
    'True form': rolling xGI per 90 over the last `window` matches, with each
    match's xGI re-weighted by the quality of the defence it came against
    (venue-adjusted strength, relative to league mean). Three involvements
    against promoted sides and one against Arsenal are different assets —
    raw form treats them identically; this doesn't.

    factor > 1 means the run came against tough defences (production is
    UNDERSTATED by raw form); factor < 1 means it was farmed against weak
    ones (OVERSTATED — regression candidate one level deeper than the
    xGI-vs-GI watchlist).

    Returns {pid: {'adj_xgi90', 'raw_xgi90', 'sched_factor', 'matches'}}.
    """
    strength_cols = ['strength_defence_home', 'strength_defence_away']
    if teams_df.empty or not all(c in teams_df.columns for c in strength_cols):
        return {}
    st = teams_df.set_index('id')[strength_cols].apply(pd.to_numeric, errors='coerce')
    st = st.where(np.isfinite(st) & (st > 100))
    if st.notna().sum().sum() == 0:
        return {}
    mean_def = float(np.nanmean(st.values))
    st = st.fillna(mean_def)
    defs = st.to_dict('index')

    out = {}
    for pid, matches in player_histories.items():
        recent = sorted(matches, key=lambda m: (m.get('round') or 0))
        recent = [m for m in recent if m.get('minutes', 0) >= min_match_minutes][-window:]
        if not recent:
            continue
        tot_mins, tot_xgi, tot_adj, factors = 0.0, 0.0, 0.0, []
        for m in recent:
            mins = m.get('minutes', 0)
            try:
                xgi = float(m.get('expected_goal_involvements') or 0)
            except (TypeError, ValueError):
                xgi = 0.0
            opp = m.get('opponent_team')
            # The opponent defended at THEIR venue: player home => opp away
            key = 'strength_defence_away' if m.get('was_home') else 'strength_defence_home'
            opp_def = defs.get(opp, {}).get(key, mean_def)
            factor = opp_def / mean_def if mean_def else 1.0
            tot_mins += mins
            tot_xgi += xgi
            tot_adj += xgi * factor
            factors.append(factor)
        if tot_mins < min_match_minutes:
            continue
        out[pid] = {
            'raw_xgi90': round(tot_xgi / tot_mins * 90, 3),
            'adj_xgi90': round(tot_adj / tot_mins * 90, 3),
            'sched_factor': round(float(np.mean(factors)), 3),
            'matches': len(recent),
        }
    return out


def calculate_custom_fdr(fixtures, teams_df, anchor_gw, num_gameweeks=5):
    """
    Attack- and defence-specific fixture difficulty built from FPL's team
    strength ratings, scaled to the familiar 1-5 range.

    FPL's generic FDR gives one number per fixture, but "easy fixture for a
    defender" and "easy fixture for a forward" are different questions:
      - attack difficulty = opponent's (venue-specific) DEFENCE strength
      - defence difficulty = opponent's (venue-specific) ATTACK strength

    Returns dict of team_id -> {att_fdr, def_fdr, next_att_fdr, next_def_fdr}
    """
    strength_cols = ['strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']
    if not all(c in teams_df.columns for c in strength_cols):
        return {}

    strengths = teams_df.set_index('id')[strength_cols].to_dict('index')

    att_pool = ([s['strength_attack_home'] for s in strengths.values()] +
                [s['strength_attack_away'] for s in strengths.values()])
    def_pool = ([s['strength_defence_home'] for s in strengths.values()] +
                [s['strength_defence_away'] for s in strengths.values()])

    def scale(value, pool):
        lo, hi = min(pool), max(pool)
        if hi == lo:
            return 3.0
        return round(1 + 4 * (value - lo) / (hi - lo), 2)

    upcoming_gws = set(range(anchor_gw + 1, anchor_gw + num_gameweeks + 1))
    per_team = {tid: {'att': [], 'def': []} for tid in strengths}

    for f in sorted([f for f in fixtures if f.get('event') in upcoming_gws],
                    key=lambda f: (f.get('event') or 0, f.get('kickoff_time') or '')):
        home, away = f['team_h'], f['team_a']
        if home in strengths and away in strengths:
            # Home team attacks the away side's away-defence, defends their away-attack
            per_team[home]['att'].append(scale(strengths[away]['strength_defence_away'], def_pool))
            per_team[home]['def'].append(scale(strengths[away]['strength_attack_away'], att_pool))
            # Away team faces the home side's home strengths
            per_team[away]['att'].append(scale(strengths[home]['strength_defence_home'], def_pool))
            per_team[away]['def'].append(scale(strengths[home]['strength_attack_home'], att_pool))

    result = {}
    for tid, vals in per_team.items():
        result[tid] = {
            'att_fdr': round(sum(vals['att']) / len(vals['att']), 2) if vals['att'] else None,
            'def_fdr': round(sum(vals['def']) / len(vals['def']), 2) if vals['def'] else None,
            'next_att_fdr': vals['att'][0] if vals['att'] else None,
            'next_def_fdr': vals['def'][0] if vals['def'] else None,
        }
    return result


# =============================================================================
# EXPECTED CLEAN SHEETS MODEL
# =============================================================================

def build_team_xg_ledger(df_active, gks_per_team=2):
    """
    Per-team, per-match EXPECTED goals for and against, for the whole season
    so far — the input that makes defensive ratings usable weeks earlier than
    actual goals do.

    The trick: a goalkeeper's `expected_goals_conceded` in a match IS his
    team's xGC for that match. So one (rotation-safe: two) keeper per team
    gives a complete xGC ledger for ~40 element-summary calls, and a team's
    xGF in any match is simply the OPPONENT's xGC in the same match. No new
    data source, no scraping.

    Matches are only accepted when the fetched keepers cover 80+ minutes, so
    a split match isn't recorded as half a game's worth of chances.

    Returns {team_id: {fixture_id: {'xgc': float, 'xgf': float|None,
                                    'opponent': int, 'was_home': bool}}}.
    """
    if df_active is None or df_active.empty:
        return {}

    gks = df_active[(df_active['position'] == 'GKP') & (df_active['minutes'] > 0)]
    if gks.empty:
        return {}

    keeper_team = {}
    for tid, grp in gks.groupby('team'):
        for pid in grp.nlargest(gks_per_team, 'minutes')['id'].tolist():
            keeper_team[int(pid)] = int(tid)

    if not keeper_team:
        return {}

    print(f"  Building team xG ledger from {len(keeper_team)} keepers...")
    histories = fetch_player_history_batch(list(keeper_team.keys()))

    # Accumulate xGC and minutes per (team, fixture)
    raw = {}
    for pid, matches in histories.items():
        tid = keeper_team.get(int(pid))
        if tid is None:
            continue
        for m in matches:
            fid = m.get('fixture')
            mins = m.get('minutes') or 0
            if fid is None or mins <= 0:
                continue
            try:
                xgc = float(m.get('expected_goals_conceded') or 0.0)
            except (TypeError, ValueError):
                continue
            slot = raw.setdefault((tid, fid), {'xgc': 0.0, 'mins': 0,
                                               'was_home': bool(m.get('was_home')),
                                               'opponent': m.get('opponent_team')})
            slot['xgc'] += xgc
            slot['mins'] += mins

    # Keep only matches with near-full keeper coverage
    ledger = {}
    for (tid, fid), slot in raw.items():
        if slot['mins'] < 80:
            continue
        ledger.setdefault(tid, {})[fid] = {
            'xgc': slot['xgc'], 'xgf': None,
            'opponent': slot['opponent'], 'was_home': slot['was_home'],
        }

    # A team's xGF in a match is the opponent's xGC in that same match
    for tid, fixtures in ledger.items():
        for fid, rec in fixtures.items():
            opp = rec.get('opponent')
            opp_rec = ledger.get(opp, {}).get(fid) if opp else None
            if opp_rec:
                rec['xgf'] = opp_rec['xgc']

    covered = sum(len(v) for v in ledger.values())
    print(f"  xG ledger: {covered} team-matches across {len(ledger)} teams")
    return ledger


def calculate_team_recent_form(fixtures_data, window=6, xg_ledger=None):
    """
    Per-team attacking/defensive form from FINISHED fixtures this season:
    goals scored and conceded per match plus clean sheets kept over the last
    `window` games. This is the component that makes clean-sheet expectations
    react to form — it re-derives from results at every data refresh.

    When `xg_ledger` (see build_team_xg_ledger) is supplied, scored_pm and
    conceded_pm are taken from EXPECTED goals rather than actual ones. Goals
    are a tiny, high-variance sample over a 6-game window: a promoted side
    with two clean sheets reads as elite on goals and as ordinary on xGC.
    Actual goals and clean sheets are still returned for display.

    Returns {team_id: {scored_pm, conceded_pm, cs, played, basis}}.
    """
    # `finished` only flips once bonus points are applied, which can lag the
    # final whistle by most of a day — long enough for a completed gameweek
    # to be missing from the form window. Scores are final much earlier, so
    # `finished_provisional` is safe for goals and clean sheets.
    finished = sorted(
        [f for f in fixtures_data
         if (f.get('finished') or f.get('finished_provisional'))
         and f.get('team_h_score') is not None
         and f.get('team_a_score') is not None],
        key=lambda f: (f.get('event') or 0, f.get('kickoff_time') or '')
    )
    per_team = {}
    for f in finished:
        h, a = f['team_h'], f['team_a']
        hs, as_ = f['team_h_score'], f['team_a_score']
        per_team.setdefault(h, []).append((hs, as_))
        per_team.setdefault(a, []).append((as_, hs))

    # Fixture ids in chronological order, for aligning the xG ledger
    order = {}
    for f in finished:
        for tid in (f['team_h'], f['team_a']):
            order.setdefault(tid, []).append(f['id'])

    form = {}
    for tid, matches in per_team.items():
        recent = matches[-window:]
        n = len(recent)
        scored = sum(m[0] for m in recent)
        conceded = sum(m[1] for m in recent)
        scored_pm, conceded_pm, basis = scored / n, conceded / n, 'goals'

        # Prefer expected goals when the ledger covers enough of this window
        if xg_ledger:
            team_led = xg_ledger.get(tid, {})
            recent_fids = order.get(tid, [])[-window:]
            xgf_vals = [team_led[fid]['xgf'] for fid in recent_fids
                        if fid in team_led and team_led[fid].get('xgf') is not None]
            xgc_vals = [team_led[fid]['xgc'] for fid in recent_fids
                        if fid in team_led and team_led[fid].get('xgc') is not None]
            if len(xgc_vals) >= max(1, len(recent_fids) // 2):
                conceded_pm = float(np.mean(xgc_vals))
                basis = 'xg'
            if len(xgf_vals) >= max(1, len(recent_fids) // 2):
                scored_pm = float(np.mean(xgf_vals))

        form[tid] = {
            'scored_pm': scored_pm,
            'conceded_pm': conceded_pm,          # what the model uses
            'goals_conceded_pm': conceded / n,   # actual goals, for display
            'xgc_pm': conceded_pm if basis == 'xg' else None,
            'cs': sum(1 for m in recent if m[1] == 0),
            'played': n,
            'basis': basis,
        }
    return form


def calculate_expected_clean_sheets(fixtures_data, teams_df, anchor_gw,
                                    num_gws=5, form_weight=0.6, form_window=6,
                                    odds_lambdas=None, xg_ledger=None):
    """
    Expected clean sheets per team over the next `num_gws` gameweeks.

    Per fixture, expected goals conceded is a multiplicative Poisson rate

        lambda = league_avg_goals x opponent-attack factor x own-defence factor

    and P(clean sheet) = P(Poisson(lambda) = 0) = exp(-lambda). Expected CS
    over the horizon is the SUM of per-fixture probabilities — which makes it
    DGW/BGW aware for free (two fixtures add two chances; a blank adds none).

    STATIC COMPONENT — two sources, in order of preference:
      1. FPL's venue-specific team strength ratings, when they contain real
         values. Pre-season the API can ship these as null/0, which would
         poison every division — so values are coerced to numeric and any
         non-finite or sub-100 entry is treated as missing.
      2. If fewer than half the teams have usable strengths, fall back to the
         per-fixture difficulty ratings (team_h_difficulty/team_a_difficulty),
         which are always present. Difficulty d in 1..5 maps to a base rate
         lambda_static = 0.70 + 0.35 x (d - 1)  →  CS ~50% at d=1, ~25% at
         d=3, ~12% at d=5.

    DYNAMIC COMPONENT — actual goals scored/conceded per match over each
    team's last `form_window` FINISHED games, re-derived at every refresh.
    Form's share ramps with games played (full weight after 3), so
    pre-season it is honestly 100% static and the numbers start moving with
    the first real results.

    Returns {team_id: {xcs, avg_cs_prob, fixtures: [(gw, opp_short, venue,
    cs_prob)], fixture_string, count, recent_conceded_pm, recent_cs,
    recent_played}}.
    """
    if teams_df.empty:
        return {}

    strength_cols = ['strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']

    # --- Sanitise strengths: numeric, finite, plausibly scaled (FPL uses
    # ~1000-1400; null/0 placeholders mean "not set yet") ---
    st = teams_df.set_index('id').reindex(columns=strength_cols)
    st = st.apply(pd.to_numeric, errors='coerce')
    st = st.where(np.isfinite(st) & (st > 100))

    valid_teams = int(st.notna().all(axis=1).sum())
    strengths_ok = valid_teams >= max(2, len(teams_df) // 2)
    if not strengths_ok:
        print(f"  xCS: only {valid_teams}/{len(teams_df)} teams have usable strength "
              f"ratings — falling back to fixture difficulty ratings")

    # Partial gaps in an otherwise-usable table: fill with column means
    if strengths_ok:
        st = st.fillna(st.mean())
        strengths = st.to_dict('index')
        mean_att = float(np.mean([[v['strength_attack_home'], v['strength_attack_away']]
                                  for v in strengths.values()]))
        mean_def = float(np.mean([[v['strength_defence_home'], v['strength_defence_away']]
                                  for v in strengths.values()]))
    else:
        strengths, mean_att, mean_def = {}, 1.0, 1.0

    short = dict(zip(teams_df['id'], teams_df.get('short_name', teams_df['name'])))
    all_ids = list(teams_df['id'])

    recent = calculate_team_recent_form(fixtures_data, window=form_window,
                                        xg_ledger=xg_ledger)
    played_vals = [v['scored_pm'] for v in recent.values() if v['played'] > 0]
    league_avg_goals = float(np.mean(played_vals)) if played_vals else 1.40
    if not np.isfinite(league_avg_goals) or league_avg_goals <= 0:
        league_avg_goals = 1.40

    def _w(tid):
        """
        Effective form weight: n / (n + k), capped at `form_weight_max`.

        The old rule ramped to full weight after THREE games, which let a
        two-match sample move a team's expected goals conceded by ~40%. That
        is the promoted-side trap: Hull keeping two clean sheets read as an
        elite defence when their underlying process said otherwise. Shrinkage
        means 2 games carry ~25% weight, 6 carry ~50%, 20 carry the cap — and
        it needs no promoted-team flag, because FPL's static strength ratings
        already price those sides low and now stay in control early on.
        """
        played = recent.get(tid, {}).get('played', 0)
        if played <= 0:
            return 0.0
        return min(form_weight, played / (played + TEAM_FORM_SHRINK_K))

    def _att_factor(opp_id, opp_venue):
        static = (strengths[opp_id][f'strength_attack_{opp_venue}'] / mean_att
                  if strengths_ok else 1.0)
        w = _w(opp_id)
        form_f = recent.get(opp_id, {}).get('scored_pm', league_avg_goals) / league_avg_goals
        return (1 - w) * static + w * form_f

    def _def_factor(tid, venue):
        # Stronger defence => factor below 1 => lower lambda
        static = (mean_def / strengths[tid][f'strength_defence_{venue}']
                  if strengths_ok else 1.0)
        w = _w(tid)
        form_f = recent.get(tid, {}).get('conceded_pm', league_avg_goals) / league_avg_goals
        return (1 - w) * static + w * form_f

    upcoming_gws = set(range(anchor_gw + 1, anchor_gw + num_gws + 1))
    upcoming = sorted([f for f in fixtures_data if f.get('event') in upcoming_gws],
                      key=lambda f: (f.get('event') or 0, f.get('kickoff_time') or ''))

    result = {tid: {'xcs': 0.0, 'fixtures': [], 'count': 0} for tid in all_ids}

    def _add(tid, opp_id, venue, opp_venue, gw, own_difficulty):
        if tid not in result:
            return
        if strengths_ok:
            lam = league_avg_goals * _att_factor(opp_id, opp_venue) * _def_factor(tid, venue)
        else:
            # Difficulty-based static rate, still scaled by form on both sides
            d = own_difficulty if own_difficulty in (1, 2, 3, 4, 5) else 3
            lam_static = 0.70 + 0.35 * (d - 1)
            wT, wO = _w(tid), _w(opp_id)
            own_form = recent.get(tid, {}).get('conceded_pm', league_avg_goals) / league_avg_goals
            opp_form = recent.get(opp_id, {}).get('scored_pm', league_avg_goals) / league_avg_goals
            lam = lam_static * ((1 - wT) + wT * own_form) * ((1 - wO) + wO * opp_form)
        if not np.isfinite(lam):
            lam = league_avg_goals
        # Market blend: when the bookmakers have priced THIS specific
        # fixture (this team vs this opponent — could be next GW or a
        # couple out, whatever fetch_market_lambdas found priced), average
        # our model with the market's goals-against rate — the market
        # prices in team news hours before any stats feed does.
        if odds_lambdas:
            mk = odds_lambdas.get(tid, {}).get(opp_id)
            if mk:
                lam = 0.5 * lam + 0.5 * mk['lam_against']
        lam = float(np.clip(lam, 0.25, 3.5))
        p_cs = float(np.exp(-lam))
        result[tid]['xcs'] += p_cs
        result[tid]['count'] += 1
        result[tid]['fixtures'].append((gw, short.get(opp_id, '???'),
                                        'H' if venue == 'home' else 'A', p_cs))

    for f in upcoming:
        _add(f['team_h'], f['team_a'], 'home', 'away', f.get('event'),
             f.get('team_h_difficulty'))
        _add(f['team_a'], f['team_h'], 'away', 'home', f.get('event'),
             f.get('team_a_difficulty'))

    for tid, v in result.items():
        v['xcs'] = round(v['xcs'], 2)
        v['avg_cs_prob'] = round(v['xcs'] / v['count'] * 100, 1) if v['count'] else 0.0
        v['fixture_string'] = ', '.join(
            f"{opp} ({ven}) {p * 100:.0f}%" for _gw, opp, ven, p in v['fixtures'])
        rec = recent.get(tid, {})
        has_form = bool(rec) and rec.get('played', 0) > 0
        v['recent_conceded_pm'] = round(rec['conceded_pm'], 2) if has_form else None
        v['recent_goals_conceded_pm'] = round(rec.get('goals_conceded_pm', 0), 2) if has_form else None
        v['recent_xgc_pm'] = (round(rec['xgc_pm'], 2)
                              if has_form and rec.get('xgc_pm') is not None else None)
        v['form_basis'] = rec.get('basis') if has_form else None
        v['recent_cs'] = rec.get('cs') if has_form else None
        v['recent_played'] = rec.get('played', 0) if rec else 0

    return result


# =============================================================================
# SNAPSHOT STORE — daily state for ownership/price trend deltas
# =============================================================================

SNAPSHOT_DB_PATH = os.environ.get(
    'FPL_SNAPSHOT_DB',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fpl_snapshots.db')
)
SNAPSHOT_COLS = ['id', 'price', 'ownership', 'total_points', 'form',
                 'transfers_in_gw', 'transfers_out_gw']


def _snapshot_conn():
    conn = sqlite3.connect(SNAPSHOT_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_snapshots (
            snap_date        TEXT NOT NULL,
            player_id        INTEGER NOT NULL,
            gw               INTEGER,
            price            REAL,
            ownership        REAL,
            total_points     INTEGER,
            form             REAL,
            transfers_in_gw  INTEGER,
            transfers_out_gw INTEGER,
            PRIMARY KEY (snap_date, player_id)
        )
    """)
    return conn


def save_daily_snapshot(df_active, gw):
    """
    Append today's player state (one row per player per day, idempotent).
    Bootstrap-static is a point-in-time snapshot — without this, every refresh
    throws away the previous state and trends are unknowable.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    rows = []
    for _, r in df_active[SNAPSHOT_COLS].iterrows():
        rows.append((
            today, int(r['id']), gw,
            None if pd.isna(r['price']) else float(r['price']),
            None if pd.isna(r['ownership']) else float(r['ownership']),
            None if pd.isna(r['total_points']) else int(r['total_points']),
            None if pd.isna(r['form']) else float(r['form']),
            None if pd.isna(r['transfers_in_gw']) else int(r['transfers_in_gw']),
            None if pd.isna(r['transfers_out_gw']) else int(r['transfers_out_gw']),
        ))
    conn = _snapshot_conn()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO player_snapshots VALUES (?,?,?,?,?,?,?,?,?)", rows
        )
        # Retention: keep 90 days
        conn.execute(
            "DELETE FROM player_snapshots WHERE snap_date < date('now', '-90 days')"
        )
    conn.close()
    print(f"  Snapshot saved for {len(rows)} players ({today})")


def load_snapshot_baseline(days=7):
    """
    Oldest snapshot within the window (excluding today), per player.
    Returns (dict of player_id -> {price, ownership}, baseline_date or None).
    """
    conn = _snapshot_conn()
    row = conn.execute(
        """SELECT MIN(snap_date) FROM player_snapshots
           WHERE snap_date >= date('now', ?) AND snap_date < date('now')""",
        (f'-{int(days)} days',)
    ).fetchone()
    baseline_date = row[0] if row else None
    if not baseline_date:
        conn.close()
        return {}, None
    baseline = {
        pid: {'price': price, 'ownership': own}
        for pid, price, own in conn.execute(
            "SELECT player_id, price, ownership FROM player_snapshots WHERE snap_date = ?",
            (baseline_date,)
        )
    }
    conn.close()
    return baseline, baseline_date


# =============================================================================
# BOOKMAKER ODDS — market-implied goal expectations (optional)
# =============================================================================
# Set ODDS_API_KEY (free tier at the-odds-api.com) to enable. Without a key
# everything below is a silent no-op and the models use their own maths.

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '').strip()
ODDS_API_URL = ("https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
                "?regions=uk&markets=h2h,totals&oddsFormat=decimal")

_ODDS_TEAM_ALIASES = {
    'spurs': 'tottenham', 'tottenham hotspur': 'tottenham',
    'manchester united': 'man utd', 'manchester utd': 'man utd',
    'manchester city': 'man city',
    'nottingham forest': "nott'm forest",
    'wolverhampton wanderers': 'wolves', 'wolverhampton': 'wolves',
    'brighton and hove albion': 'brighton', 'brighton & hove albion': 'brighton',
    'west ham united': 'west ham', 'newcastle united': 'newcastle',
    'leeds united': 'leeds', 'afc bournemouth': 'bournemouth',
    'leicester city': 'leicester', 'ipswich': 'ipswich town',
    'sheffield united': 'sheffield utd', 'luton town': 'luton',
}


def _norm_team_name(name):
    n = (name or '').lower().replace(' fc', '').replace('.', '').strip()
    return _ODDS_TEAM_ALIASES.get(n, n)


def match_odds_team_to_fpl(odds_name, fpl_names_by_id):
    """Map a bookmaker team name to an FPL team id (exact, containment,
    then fuzzy). Returns team id or None."""
    target = _norm_team_name(odds_name)
    normed = {tid: _norm_team_name(nm) for tid, nm in fpl_names_by_id.items()}
    for tid, nm in normed.items():
        if nm == target:
            return tid
    for tid, nm in normed.items():
        if nm and (nm in target or target in nm):
            return tid
    import difflib
    best = difflib.get_close_matches(target, list(normed.values()), n=1, cutoff=0.75)
    if best:
        for tid, nm in normed.items():
            if nm == best[0]:
                return tid
    return None


def derive_match_lambdas(h2h_probs, total_line, over_prob):
    """
    Expected goals per side from market prices.

    h2h_probs: de-vigged (p_home, p_draw, p_away). total_line: the main
    over/under line (e.g. 2.5). over_prob: de-vigged P(over).

    Total goals: nudge the line by how the market leans (a 60% over at 2.5
    implies a true total nearer 2.8). Supremacy: s ≈ 1.3 x (pH − pA) is a
    standard first-order mapping from outcome probabilities to goal
    difference for football scorelines. Split total ± supremacy and clip to
    sane per-side rates.
    """
    p_home, _p_draw, p_away = h2h_probs
    lam_total = max(0.5, float(total_line) + (float(over_prob) - 0.5) * 1.2)
    supremacy = 1.3 * (p_home - p_away)
    lam_home = float(np.clip((lam_total + supremacy) / 2, 0.2, 4.0))
    lam_away = float(np.clip((lam_total - supremacy) / 2, 0.2, 4.0))
    return lam_home, lam_away


def _devig(prices):
    """Decimal odds -> normalised implied probabilities."""
    inv = [1.0 / p for p in prices if p and p > 1.0]
    if len(inv) != len(prices) or not inv:
        return None
    total = sum(inv)
    return [v / total for v in inv]


def fetch_market_lambdas(teams_df):
    """
    One call per refresh cycle: for every upcoming EPL match with h2h+totals
    prices, expected goals for and against per team. Returns
    {team_id: {opp_id: {'lam_for': x, 'lam_against': y}}} — every priced
    fixture for a team, not just its nearest one, keyed by opponent so a
    consumer looks up its own known next-N-gameweek opponents against this
    dict rather than assuming only one entry exists. {} when no key / any
    failure.

    How far ahead this actually reaches depends on the bookmakers, not on
    this code: EPL markets are usually posted and liquid for the next
    fixture, sometimes the one after, rarely much beyond that — a market
    can't price news (an injury, a rotation call) that hasn't happened yet.
    So most refreshes this will cover 1-2 fixtures per team, occasionally
    more; consumers should treat a missing (team, opponent) pair as "not
    priced yet", not as an error.
    """
    if not ODDS_API_KEY:
        return {}
    try:
        r = requests.get(f"{ODDS_API_URL}&apiKey={ODDS_API_KEY}", timeout=15)
        r.raise_for_status()
        events = r.json()
        remaining = r.headers.get('x-requests-remaining')
        used = r.headers.get('x-requests-last')
        if remaining is not None:
            print(f"  Odds API credits: {used or '?'} used this call, "
                  f"{remaining} remaining this month")
            if remaining.isdigit() and int(remaining) < 20:
                print(f"  WARNING: odds API quota nearly exhausted "
                      f"({remaining} credits left) — market blend will "
                      f"silently stop once it hits 0")
    except Exception as e:
        print(f"  Odds fetch failed (non-fatal): {e}")
        return {}

    # Sorted purely so logs/behaviour are deterministic and the nearest
    # fixture for a team is the first one processed — no longer load-bearing
    # for correctness now that every fixture is kept, but still worth having.
    events = sorted(events, key=lambda ev: ev.get('commence_time') or '')

    fpl_names = dict(zip(teams_df['id'], teams_df['name']))
    out = {}
    n_fixtures = 0
    for ev in events:
        home_id = match_odds_team_to_fpl(ev.get('home_team'), fpl_names)
        away_id = match_odds_team_to_fpl(ev.get('away_team'), fpl_names)
        # Dedup by the (team, opponent) PAIR, not by team alone — this is
        # the change that lets a team carry more than one priced fixture.
        if not home_id or not away_id or away_id in out.get(home_id, {}):
            continue
        h2h, totals = None, None
        for bm in ev.get('bookmakers', []):
            for mk in bm.get('markets', []):
                if mk['key'] == 'h2h' and h2h is None and len(mk.get('outcomes', [])) == 3:
                    prices = {o['name']: o['price'] for o in mk['outcomes']}
                    ph = prices.get(ev.get('home_team'))
                    pa = prices.get(ev.get('away_team'))
                    pd_ = prices.get('Draw')
                    if ph and pa and pd_:
                        probs = _devig([ph, pd_, pa])
                        if probs:
                            h2h = (probs[0], probs[1], probs[2])
                if mk['key'] == 'totals' and totals is None:
                    overs = [o for o in mk.get('outcomes', []) if o.get('name') == 'Over']
                    unders = [o for o in mk.get('outcomes', []) if o.get('name') == 'Under']
                    if overs and unders and overs[0].get('point') is not None:
                        probs = _devig([overs[0]['price'], unders[0]['price']])
                        if probs:
                            totals = (overs[0]['point'], probs[0])
            if h2h and totals:
                break
        if not h2h or not totals:
            continue
        lam_h, lam_a = derive_match_lambdas(h2h, totals[0], totals[1])
        out.setdefault(home_id, {})[away_id] = {'lam_for': lam_h, 'lam_against': lam_a}
        out.setdefault(away_id, {})[home_id] = {'lam_for': lam_a, 'lam_against': lam_h}
        n_fixtures += 1
    if out:
        print(f"  Market lambdas derived for {n_fixtures} fixtures "
              f"across {len(out)} teams")
    return out


# =============================================================================
# GOAL ENVIRONMENT — fixture-specific expected-goals scaling
# =============================================================================

# Why this exists: a linear "±12% per FDR step" multiplier compresses fixture
# reality. Against a newly promoted side at home, an elite striker's goal
# expectation can be ~1.5-1.8x his average — a 1.18x cap lets one recent
# hattrick outrank the fixture of the season. This model scales attacking
# output by the actual goals a fixture should produce: Poisson-style
# lambda = league_avg x own-attack factor x opponent-defence weakness, with
# both factors blending static strengths and recent results (same machinery
# and same fallbacks as the Expected Clean Sheets model), optionally
# sharpened by bookmaker lambdas when ODDS_API_KEY is set.

ATT_ENV_CLIP = (0.55, 1.80)
_FDR_ENV_FALLBACK = {1: 1.40, 2: 1.18, 3: 1.00, 4: 0.85, 5: 0.72}


def calculate_goal_environment(fixtures_data, teams_df, anchor_gw, num_gws=5,
                               form_weight=0.6, form_window=6, odds_lambdas=None,
                               xg_ledger=None):
    """
    Per-team attacking environment over the next `num_gws` gameweeks.
    Returns {team_id: {'att_env_next', 'att_env_avg', 'opp_next',
    'venue_next', 'n_fixtures'}} where env is expected-goals-scored divided
    by the league average (1.0 = neutral fixture, 1.5 = a fixture that
    should produce 50% more goals than average for this team).
    """
    if teams_df.empty:
        return {}
    strength_cols = ['strength_attack_home', 'strength_attack_away',
                     'strength_defence_home', 'strength_defence_away']
    st = teams_df.set_index('id').reindex(columns=strength_cols)
    st = st.apply(pd.to_numeric, errors='coerce')
    st = st.where(np.isfinite(st) & (st > 100))
    valid_teams = int(st.notna().all(axis=1).sum())
    strengths_ok = valid_teams >= max(2, len(teams_df) // 2)
    if strengths_ok:
        st = st.fillna(st.mean())
        strengths = st.to_dict('index')
        mean_att = float(np.mean([[v['strength_attack_home'], v['strength_attack_away']]
                                  for v in strengths.values()]))
        mean_def = float(np.mean([[v['strength_defence_home'], v['strength_defence_away']]
                                  for v in strengths.values()]))
    else:
        strengths, mean_att, mean_def = {}, 1.0, 1.0

    short = dict(zip(teams_df['id'], teams_df.get('short_name', teams_df['name'])))
    recent = calculate_team_recent_form(fixtures_data, window=form_window,
                                        xg_ledger=xg_ledger)
    played_vals = [v['scored_pm'] for v in recent.values() if v['played'] > 0]
    league_avg = float(np.mean(played_vals)) if played_vals else 1.40
    if not np.isfinite(league_avg) or league_avg <= 0:
        league_avg = 1.40

    def _w(tid):
        # Same n/(n+k) shrinkage as the clean-sheet model — see _w there.
        played = recent.get(tid, {}).get('played', 0)
        if played <= 0:
            return 0.0
        return min(form_weight, played / (played + TEAM_FORM_SHRINK_K))

    def _att_factor(tid, venue):
        static = (strengths[tid][f'strength_attack_{venue}'] / mean_att) if strengths_ok else 1.0
        w = _w(tid)
        # Clip: a single 4-goal fluke shouldn't double a team's rating
        form_f = float(np.clip(
            recent.get(tid, {}).get('scored_pm', league_avg) / league_avg, 0.5, 1.7))
        return (1 - w) * static + w * form_f

    def _opp_def_weakness(opp_id, opp_venue):
        static = (mean_def / strengths[opp_id][f'strength_defence_{opp_venue}']) if strengths_ok else 1.0
        w = _w(opp_id)
        form_f = float(np.clip(
            recent.get(opp_id, {}).get('conceded_pm', league_avg) / league_avg, 0.5, 1.7))
        return (1 - w) * static + w * form_f

    upcoming_gws = set(range(anchor_gw + 1, anchor_gw + num_gws + 1))
    upcoming = sorted([f for f in fixtures_data if f.get('event') in upcoming_gws],
                      key=lambda f: (f.get('event') or 0, f.get('kickoff_time') or ''))

    envs = {tid: [] for tid in teams_df['id']}
    meta = {tid: {'opp_next': '', 'venue_next': ''} for tid in teams_df['id']}

    def _env_for(tid, opp_id, venue, opp_venue, own_difficulty):
        if strengths_ok:
            lam = league_avg * _att_factor(tid, venue) * _opp_def_weakness(opp_id, opp_venue)
        else:
            d = own_difficulty if own_difficulty in (1, 2, 3, 4, 5) else 3
            lam = league_avg * _FDR_ENV_FALLBACK[d]
            # form still bends the fallback
            wT, wO = _w(tid), _w(opp_id)
            own_f = recent.get(tid, {}).get('scored_pm', league_avg) / league_avg
            opp_f = recent.get(opp_id, {}).get('conceded_pm', league_avg) / league_avg
            lam = lam * ((1 - wT) + wT * own_f) * ((1 - wO) + wO * opp_f)
        # Market blend for whichever fixtures are actually priced
        if odds_lambdas:
            mk = odds_lambdas.get(tid, {}).get(opp_id)
            if mk:
                lam = 0.5 * lam + 0.5 * mk['lam_for']
        if not np.isfinite(lam):
            lam = league_avg
        return float(np.clip(lam / league_avg, *ATT_ENV_CLIP))

    # Per-fixture detail as well as the aggregate. The clean-sheet model
    # already emits this; the attacking side needs it too so a grid can show
    # gameweek-by-gameweek expected goals rather than one averaged multiplier.
    per_fixture = {tid: [] for tid in envs}

    for f in upcoming:
        h, a = f['team_h'], f['team_a']
        gw = f.get('event')
        if h in envs:
            env = _env_for(h, a, 'home', 'away', f.get('team_h_difficulty'))
            if not envs[h]:
                meta[h] = {'opp_next': short.get(a, '???'), 'venue_next': 'H'}
            envs[h].append(env)
            per_fixture[h].append((gw, short.get(a, '???'), 'H', env))
        if a in envs:
            env = _env_for(a, h, 'away', 'home', f.get('team_a_difficulty'))
            if not envs[a]:
                meta[a] = {'opp_next': short.get(h, '???'), 'venue_next': 'A'}
            envs[a].append(env)
            per_fixture[a].append((gw, short.get(h, '???'), 'A', env))

    out = {}
    for tid, vals in envs.items():
        out[tid] = {
            'att_env_next': round(vals[0], 3) if vals else 1.0,
            'att_env_avg': round(float(np.mean(vals)), 3) if vals else 1.0,
            'n_fixtures': len(vals),
            'fixtures': per_fixture.get(tid, []),
            'league_avg_goals': round(float(league_avg), 3),
            **meta[tid],
        }
    return out


# =============================================================================
# MODEL CALIBRATION — log projections, score them once results are in
# =============================================================================

def log_projections(df_active, target_gw):
    """
    Record this refresh's next-GW projections (ours and FPL's ep_next) for
    the gameweek being planned. INSERT OR REPLACE means the last write
    before the deadline is the one that gets scored — exactly the number a
    user would have acted on.
    """
    conn = _snapshot_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projection_log (
            gw        INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            proj      REAL,
            fpl_ep    REAL,
            PRIMARY KEY (gw, player_id)
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS actual_points (
            gw        INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            pts       INTEGER,
            PRIMARY KEY (gw, player_id)
        )""")
    rows = []
    for r in df_active.itertuples():
        proj = getattr(r, 'proj_pts_next', None)
        ep = getattr(r, 'ep_next', None)
        rows.append((int(target_gw), int(r.id),
                     None if pd.isna(proj) else float(proj),
                     None if pd.isna(ep) else float(ep)))
    with conn:
        conn.executemany("INSERT OR REPLACE INTO projection_log VALUES (?,?,?,?)", rows)
    conn.close()


def log_actual_points(df_active, current_gw_num):
    """
    Record realised points for the in-flight gameweek from bootstrap's
    event_points. Overwritten each refresh while the GW runs, so the stored
    values converge to finals without needing to detect 'finished'.
    """
    if 'event_points' not in df_active.columns:
        return
    conn = _snapshot_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS actual_points (
            gw INTEGER NOT NULL, player_id INTEGER NOT NULL, pts INTEGER,
            PRIMARY KEY (gw, player_id))""")
    rows = [(int(current_gw_num), int(r.id),
             int(r.event_points) if pd.notna(r.event_points) else None)
            for r in df_active.itertuples()]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO actual_points VALUES (?,?,?)", rows)
    conn.close()


def compute_calibration(min_actual_minutes_players=50):
    """
    Score logged projections against realised points on every completed
    gameweek: mean absolute error for this model vs FPL's ep_next, overall
    and per gameweek. The number that answers 'is my model worth obeying?'
    Returns dict or None if nothing is scoreable yet.
    """
    try:
        conn = _snapshot_conn()
        rows = conn.execute("""
            SELECT p.gw, p.proj, p.fpl_ep, a.pts
            FROM projection_log p JOIN actual_points a
              ON a.gw = p.gw AND a.player_id = p.player_id
            WHERE p.proj IS NOT NULL AND a.pts IS NOT NULL
        """).fetchall()
        conn.close()
    except Exception as e:
        print(f"  Calibration query failed: {e}")
        return None
    if len(rows) < min_actual_minutes_players:
        return None
    by_gw = {}
    for gw, proj, fpl_ep, pts in rows:
        by_gw.setdefault(gw, []).append((proj, fpl_ep, pts))
    per_gw = []
    for gw in sorted(by_gw):
        sample = by_gw[gw]
        mae_model = float(np.mean([abs(p - a) for p, _f, a in sample]))
        fp = [(f, a) for _p, f, a in sample if f is not None]
        mae_fpl = float(np.mean([abs(f - a) for f, a in fp])) if fp else None
        per_gw.append({'gw': gw, 'n': len(sample),
                       'mae_model': round(mae_model, 3),
                       'mae_fpl': round(mae_fpl, 3) if mae_fpl is not None else None})
    all_rows = [x for sample in by_gw.values() for x in sample]
    mae_model = round(float(np.mean([abs(p - a) for p, _f, a in all_rows])), 3)
    fp = [(f, a) for _p, f, a in all_rows if f is not None]
    mae_fpl = round(float(np.mean([abs(f - a) for f, a in fp])), 3) if fp else None
    return {'per_gw': per_gw, 'mae_model': mae_model, 'mae_fpl': mae_fpl,
            'n': len(all_rows), 'gws': len(per_gw)}


# =============================================================================
# WALK-FORWARD BACKTEST — projected vs actual, reconstructed from history
# =============================================================================

def _rate(total, minutes):
    """Per-90 rate, safe at zero minutes."""
    return (total / minutes * 90.0) if minutes and minutes > 0 else 0.0


def reconstruct_player_frame(histories, meta, upto_round, recent_window=6):
    """
    Rebuild every player's state EXACTLY as it stood before `upto_round`.

    This is what lets the model be scored without any stored logs. Each
    element-summary history row carries that player's minutes, xG, xA,
    bonus, saves, conceded, clean sheets and defensive contribution for one
    match, tagged with its round. Filtering to rounds strictly BELOW the
    target and re-deriving the per-90 rates reproduces the inputs the engine
    would have had at that deadline.

    The one rule that matters: `round < upto_round`, everywhere, with no
    exceptions. A single leaked row turns the whole exercise into nonsense
    that looks impressive.

    Not reconstructable: `avail_pct`. FPL does not archive
    chance_of_playing per round, so injury flags as they stood at the time
    are gone. Everyone is treated as fully available, which means the
    backtest is mildly pessimistic — it projects points for players who were
    flagged and did not play. That inflates error for every model equally,
    so comparisons between models stay fair even though the absolute MAE is
    a little worse than live performance would be.
    """
    rows = []
    for pid, hist in histories.items():
        prior = [h for h in hist if (h.get('round') or 0) < upto_round]
        if not prior:
            continue
        m = meta.get(pid)
        if not m:
            continue

        mins = sum((h.get('minutes') or 0) for h in prior)
        if mins <= 0:
            continue

        f = lambda k: sum(float(h.get(k) or 0) for h in prior)
        recent = prior[-recent_window:]
        rec_mins = sum((h.get('minutes') or 0) for h in recent)
        starts = sum(1 for h in recent if (h.get('minutes') or 0) >= 60)

        dc_vals = [float(h.get('defensive_contribution') or 0) for h in prior
                   if (h.get('minutes') or 0) > 0]
        dc_mean = (sum(dc_vals) / len(dc_vals)) if dc_vals else 0.0
        dc_var = (sum((v - dc_mean) ** 2 for v in dc_vals) / (len(dc_vals) - 1)
                  ) if len(dc_vals) > 1 else 0.0
        thr = 10 if m['position'] in ('GKP', 'DEF') else 12
        full = [h for h in prior if (h.get('minutes') or 0) >= 60]

        rows.append({
            'id': pid,
            'position': m['position'],
            'team': m['team'],
            'web_name': m['web_name'],
            'pen_rank': m.get('pen_rank'),
            'minutes': mins,
            'xg_per_90': _rate(f('expected_goals'), mins),
            'xa_per_90': _rate(f('expected_assists'), mins),
            'bonus_per_90': _rate(f('bonus'), mins),
            'cs_per_90': _rate(f('clean_sheets'), mins),
            'gc_per_90': _rate(f('goals_conceded'), mins),
            'saves': f('saves'),
            'defcon_per_90': _rate(f('defensive_contribution'), mins),
            'defcon_per_90_games': _rate(f('defensive_contribution'), mins),
            'defcon_var': dc_var,
            'qualifying_games': len(dc_vals),
            'starts_60': len(full),
            'bonus_games_60': sum(1 for h in full
                                  if (h.get('defensive_contribution') or 0) >= thr),
            'bonus_threshold': thr,
            # Availability is unknowable after the fact — see docstring.
            'avail_pct': 100.0,
            'recent_minutes_pct': (rec_mins / (90.0 * max(len(recent), 1))) * 100,
            'start_rate': (starts / max(len(recent), 1)) * 100,
        })
    return pd.DataFrame(rows)


def _fixtures_as_of(fixtures_data, upto_round):
    """
    Fixture list as it looked before `upto_round`: results from earlier
    rounds stay finished, everything from the target round onward is marked
    unplayed so team-form functions cannot see the future.
    """
    out = []
    for f in fixtures_data:
        g = dict(f)
        if (g.get('event') or 0) >= upto_round:
            g['finished'] = False
            g['finished_provisional'] = False
            g['team_h_score'] = None
            g['team_a_score'] = None
        out.append(g)
    return out


def run_projection_backtest(histories, meta, fixtures_data, teams_df,
                            gws, priors=None, recent_window=6):
    """
    Walk forward one gameweek at a time: rebuild the inputs, project, then
    score against what actually happened.

    Reported alongside the model, and this is the part that matters, are two
    naive baselines — predict each player's points-per-game to date, and
    predict the positional average. MAE on FPL scores is a weak test because
    most players score 1-3 points, so a flat guess near 2.2 already scores
    about 2.0. If the model cannot beat "just use his PPG", the modelling
    layer is not earning its place.

    Rank metrics are included for the same reason. Every decision the app
    makes is a ranking decision, not an absolute-value one, so Spearman
    correlation and top-20 precision say more than MAE does.
    """
    results, player_rows = [], []
    for g in gws:
        frame = reconstruct_player_frame(histories, meta, g, recent_window)
        if frame.empty or len(frame) < 30:
            continue

        hist_fx = _fixtures_as_of(fixtures_data, g)
        try:
            xcs = calculate_expected_clean_sheets(hist_fx, teams_df, g - 1, num_gws=1)
            genv = calculate_goal_environment(hist_fx, teams_df, g - 1, num_gws=1)
        except Exception:
            xcs, genv = {}, {}

        tgt = [f for f in fixtures_data if (f.get('event') or 0) == g]
        fdr_h = {f['team_h']: f.get('team_a_difficulty', 3) for f in tgt}
        fdr_a = {f['team_a']: f.get('team_h_difficulty', 3) for f in tgt}
        att_fdr = {**fdr_h, **fdr_a}
        counts = Counter()
        for f in tgt:
            counts[f['team_h']] += 1
            counts[f['team_a']] += 1

        frame['next_att_fdr'] = frame['team'].map(att_fdr).fillna(3.0)
        frame['next_def_fdr'] = frame['next_att_fdr']
        frame['att_env_next'] = frame['team'].map(
            lambda t: (genv.get(t, {}) or {}).get('avg_att_env', 1.0)).fillna(1.0)
        frame['cs_prob_next'] = frame['team'].map(
            lambda t: ((xcs.get(t, {}) or {}).get('avg_cs_prob') or 0) / 100.0).replace(0, np.nan)
        frame['fixture_count'] = frame['team'].map(counts).fillna(0)
        for c in ('att_fdr_5', 'def_fdr_5'):
            frame[c] = 3.0
        frame['att_env_5'] = 1.0
        frame['cs_prob_5'] = 0.25
        frame['fixture_count_8'] = 8

        try:
            _, proj, _, _, _, _ = compute_expected_points(
                frame, gw_elapsed=max(g - 1, 1), priors=priors or {})
        except Exception as e:
            print(f"  backtest GW{g} projection failed: {e}")
            continue

        actual = {pid: sum(h.get('total_points') or 0
                           for h in hist if (h.get('round') or 0) == g)
                  for pid, hist in histories.items()}
        played = {pid: any((h.get('round') or 0) == g for h in hist)
                  for pid, hist in histories.items()}

        frame = frame.assign(proj=proj.values)
        frame['actual'] = frame['id'].map(actual)
        frame['gw_label'] = f"GW{g}"
        frame['mins_played'] = frame['id'].map(
            {pid: sum(h.get('minutes') or 0 for h in hist if (h.get('round') or 0) == g)
             for pid, hist in histories.items()}).fillna(0)
        frame['in_gw'] = frame['id'].map(played).fillna(False)
        # Only players whose team actually had a fixture that week
        frame = frame[(frame['fixture_count'] > 0) & frame['actual'].notna()]
        if len(frame) < 30:
            continue

        # Naive baselines
        frame['base_ppg'] = frame.apply(
            lambda r: sum(h.get('total_points') or 0
                          for h in histories[r['id']] if (h.get('round') or 0) < g)
            / max(sum(1 for h in histories[r['id']] if (h.get('round') or 0) < g), 1), axis=1)
        # Positional baseline must come from PRIOR rounds. Averaging this
        # gameweek's actuals would hand the baseline the answers and make
        # the model look worse than it is by comparison.
        _prior_pos = {}
        for _pos in frame['position'].unique():
            _ids = set(frame[frame['position'] == _pos]['id'])
            _pts = [h.get('total_points') or 0 for pid in _ids
                    for h in histories.get(pid, []) if (h.get('round') or 0) < g]
            _prior_pos[_pos] = (sum(_pts) / len(_pts)) if _pts else 0.0
        frame['base_pos'] = frame['position'].map(_prior_pos)

        err = (frame['proj'] - frame['actual'])
        top20_proj = set(frame.nlargest(20, 'proj')['id'])
        top20_act = set(frame.nlargest(20, 'actual')['id'])
        spear = frame[['proj', 'actual']].corr(method='spearman').iloc[0, 1]

        # Per-player detail: what the model said, what he actually got.
        for _r in frame.itertuples():
            player_rows.append({
                'gw': _r.gw_label,
                'web_name': _r.web_name,
                'position': _r.position,
                'proj': round(float(_r.proj), 2),
                'actual': int(_r.actual),
                'diff': round(float(_r.actual) - float(_r.proj), 2),
                'minutes_played': int(_r.mins_played),
            })

        results.append({
            'gw': g,
            'players': len(frame),
            'mae': round(err.abs().mean(), 3),
            'bias': round(err.mean(), 3),
            'spearman': round(float(spear) if pd.notna(spear) else 0, 3),
            'top20': len(top20_proj & top20_act),
            'mae_ppg': round((frame['base_ppg'] - frame['actual']).abs().mean(), 3),
            'mae_pos': round((frame['base_pos'] - frame['actual']).abs().mean(), 3),
        })
    return results, player_rows


# =============================================================================
# MODEL LAB — feature store, replay engine, parameter sweep
# =============================================================================
# The engine's constants (FDR ratio, shrinkage k) are judgement calls. This
# stores the exact inputs behind every logged projection so they can be
# REPLAYED under different constants and scored against realised points —
# converting the model from "plausible" to "measured".

FEATURE_COLS = ['id', 'position', 'minutes', 'avail_pct', 'recent_minutes_pct',
                'start_rate', 'xg_per_90', 'xa_per_90', 'cs_per_90', 'gc_per_90',
                'bonus_threshold', 'defcon_per_90', 'hit_rate', 'qualifying_games',
                'cs_prob_next', 'cs_prob_5',
                'saves', 'bonus_per_90', 'next_att_fdr', 'next_def_fdr',
                'att_env_next', 'pen_rank', 'expected_goals', 'expected_assists',
                'clean_sheets', 'goals_conceded', 'bonus']


def log_model_features(df_active, target_gw, priors):
    """Store the engine's raw inputs (and priors) for the GW being planned —
    last write before the deadline wins, mirroring projection_log."""
    conn = _snapshot_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS feature_log (
        gw INTEGER NOT NULL, player_id INTEGER NOT NULL, feats TEXT,
        PRIMARY KEY (gw, player_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS priors_log (
        gw INTEGER PRIMARY KEY, priors TEXT)""")
    cols = [c for c in FEATURE_COLS if c in df_active.columns]
    sub = df_active[cols].replace([np.inf, -np.inf], np.nan)
    rows = [(int(target_gw), int(rec['id']), json.dumps(
                {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                 for k, v in rec.items()}))
            for rec in json.loads(sub.to_json(orient='records'))]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO feature_log VALUES (?,?,?)", rows)
        conn.execute("INSERT OR REPLACE INTO priors_log VALUES (?,?)",
                     (int(target_gw), json.dumps(priors or {})))
    conn.close()


def _load_replay_frames():
    """GWs with both stored features and realised points, as
    [(gw, features_df, priors_dict, actuals_dict)]."""
    conn = _snapshot_conn()
    try:
        gws = [r[0] for r in conn.execute(
            """SELECT DISTINCT f.gw FROM feature_log f
               JOIN actual_points a ON a.gw = f.gw LIMIT 20""")]
    except Exception:
        conn.close()
        return []
    frames = []
    for gw in gws:
        feats = [json.loads(r[0]) for r in conn.execute(
            "SELECT feats FROM feature_log WHERE gw = ?", (gw,))]
        pri_row = conn.execute("SELECT priors FROM priors_log WHERE gw = ?", (gw,)).fetchone()
        priors = {int(k): v for k, v in json.loads(pri_row[0]).items()} if pri_row else {}
        actuals = {pid: pts for pid, pts in conn.execute(
            "SELECT player_id, pts FROM actual_points WHERE gw = ? AND pts IS NOT NULL", (gw,))}
        if feats and actuals:
            frames.append((gw, pd.DataFrame(feats), priors, actuals))
    conn.close()
    return frames


def run_parameter_sweep(fdr_ratios=(1.10, 1.18, 1.26, 1.34),
                        shrink_ks=(250, 450, 700)):
    """
    Replay every scoreable gameweek under each parameter combination and
    score projection MAE against realised points. Single-threaded, mutates
    the module constants under a try/finally restore — do not call from
    concurrent contexts (Dash callbacks are fine; they serialise per worker).
    Returns sorted results + how the current live config ranks.
    """
    global FDR_STEP_RATIO, SHRINK_K
    frames = _load_replay_frames()
    if not frames:
        return None
    saved = (FDR_STEP_RATIO, SHRINK_K)
    results = []
    try:
        for ratio in fdr_ratios:
            for k in shrink_ks:
                FDR_STEP_RATIO, SHRINK_K = ratio, k
                errs = []
                for gw, fdf, priors, actuals in frames:
                    proj = compute_expected_points(fdf, gw_elapsed=max(gw - 1, 1),
                                                   priors=priors)[1]
                    for pid, p in zip(fdf['id'], proj):
                        a = actuals.get(int(pid))
                        if a is not None and pd.notna(p):
                            errs.append(abs(float(p) - a))
                if errs:
                    results.append({'fdr_ratio': ratio, 'shrink_k': k,
                                    'mae': round(float(np.mean(errs)), 4),
                                    'n': len(errs)})
    finally:
        FDR_STEP_RATIO, SHRINK_K = saved
    results.sort(key=lambda r: r['mae'])
    current = next((r for r in results
                    if r['fdr_ratio'] == saved[0] and r['shrink_k'] == saved[1]), None)
    return {'results': results, 'current': current,
            'gws': len(frames), 'live': {'fdr_ratio': saved[0], 'shrink_k': saved[1]}}


# =============================================================================
# EFFECTIVE OWNERSHIP — sampled from the top of the overall league
# =============================================================================

OVERALL_LEAGUE_ID = 314
EO_SAMPLE_SIZE = 100


def fetch_top_manager_entry_ids(sample_size=EO_SAMPLE_SIZE):
    """Entry IDs of the current overall-league leaders (paged, 50/page)."""
    ids = []
    page = 1
    while len(ids) < sample_size and page <= (sample_size // 50) + 1:
        try:
            r = requests.get(
                f"{FPL_BASE_URL}/leagues-classic/{OVERALL_LEAGUE_ID}/standings/"
                f"?page_standings={page}", timeout=15
            )
            r.raise_for_status()
            results = (r.json().get('standings') or {}).get('results') or []
        except Exception as e:
            print(f"  Error fetching overall standings page {page}: {e}")
            break
        if not results:
            break
        ids.extend(e['entry'] for e in results)
        page += 1
    return ids[:sample_size]


def calculate_effective_ownership(entry_ids, gw, max_workers=8):
    """
    Effective ownership among sampled top managers, using pick multipliers
    (0 = benched, 1 = starting, 2 = captain, 3 = triple captain). Overall
    `selected_by_percent` includes millions of abandoned teams; what moves
    rank is ownership among managers you're actually racing.
    Returns (dict of player_id -> EO%, number of squads sampled).
    """
    eo_counts = Counter()
    n_ok = 0

    def _fetch(eid):
        return fetch_team_picks(eid, gw)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch, eid) for eid in entry_ids]
        for future in as_completed(futures):
            picks_data = future.result()
            if not picks_data or 'picks' not in picks_data:
                continue
            n_ok += 1
            for p in picks_data['picks']:
                eo_counts[p['element']] += p.get('multiplier', 1)

    if n_ok == 0:
        return {}, 0
    return {pid: round(cnt / n_ok * 100, 1) for pid, cnt in eo_counts.items()}, n_ok


def estimate_price_change_likelihood(row, total_managers):
    """
    Heuristic price-change likelihood based on net transfers relative to ownership.
    Returns a score from -100 (very likely to fall) to +100 (very likely to rise).
    """
    ownership_count = (row.get('ownership', 0) / 100) * total_managers
    if ownership_count <= 0:
        ownership_count = 1

    net = row.get('net_transfers_gw', 0)
    ratio = net / ownership_count

    score = np.clip(ratio * 2000, -100, 100)
    return round(score, 1)


PRICE_TRACKER_INTERVAL = int(os.environ.get('FPL_PRICE_TRACKER_MINUTES', '20')) * 60
PRICE_TRACKER_COLS = ['price_progress', 'price_predicted', 'price_status', 'price_likelihood']


def _num_or_none(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _tonight_projection(proj):
    """
    FPL's price_change_projections: one entry per upcoming change window
    (offset 0 = tonight). The fields are new and undocumented and ship
    numbers as strings, so read defensively: accept a list of entries or a
    dict keyed by offset, and a few plausible key names for the percent.
    Returns (percent, likelihood) for tonight, or (None, None).
    """
    if not proj:
        return None, None
    entries = []
    if isinstance(proj, dict):
        for k, v in proj.items():
            entries.append({'offset': k, **v} if isinstance(v, dict) else {'offset': k, 'percent': v})
    elif isinstance(proj, list):
        entries = [e for e in proj if isinstance(e, dict)]
    for e in entries:
        off = _num_or_none(e.get('offset', e.get('day', 0)))
        if off is None or int(off) != 0:
            continue
        pct = None
        for key in ('percent', 'projected_percent', 'progress', 'value', 'projection'):
            pct = _num_or_none(e.get(key))
            if pct is not None:
                break
        if pct is None:
            continue
        return pct, _num_or_none(e.get('likelihood'))
    return None, None


# Status thresholds on FPL's predicted progress, matched to FPL's own Price
# Changes page: +97.8% and +96.6% show "Likely to rise", while +88.6%,
# +87.9%, +83.7% and +5.0% show "Unlikely to change". So "Likely" starts
# somewhere between 88.6 and 96.6; 95 is used. At 100 the change is due.
PRICE_LIKELY_PCT = float(os.environ.get('FPL_PRICE_LIKELY_PCT', '95'))
PRICE_VERY_LIKELY_PCT = 100.0


def _price_status(predicted, progress, likelihood, locked, calibrating):
    """
    FPL-style status label from FPL's predicted progress.

    The feed's 'likelihood' field is NOT FPL's status: it tracks the size of
    the projection in bands (+5% came through as 'likely', -19% as 'likely
    to drop', -33% as 'very likely'), so using it labelled Gakpo at +5% as
    'Likely to rise' while FPL's page said 'Unlikely to change'. It's kept
    in the data but no longer drives the label.
    """
    if locked:
        return 'Locked'
    if calibrating and predicted is None:
        return 'Calibrating'
    ref = predicted if predicted is not None else progress
    if ref is None:
        return ''
    if ref >= PRICE_VERY_LIKELY_PCT:
        return 'Very likely to rise'
    if ref >= PRICE_LIKELY_PCT:
        return 'Likely to rise'
    if ref <= -PRICE_VERY_LIKELY_PCT:
        return 'Very likely to drop'
    if ref <= -PRICE_LIKELY_PCT:
        return 'Likely to drop'
    return 'Unlikely to change'


def read_fpl_price_tracker(elements):
    """
    FPL's official price-change tracker, as published in bootstrap-static:
    price_change_percent (progress now; a change happens at +/-100),
    price_change_projections (tonight's projected progress and FPL's own
    likelihood), price_change_locked_until, price_change_calibrating.
    Returns a DataFrame indexed by player id, or None if the feed doesn't
    carry the fields.
    """
    if not elements or not any('price_change_percent' in e for e in elements[:50]):
        return None
    now = datetime.now().astimezone()
    rows = []
    for e in elements:
        progress = _num_or_none(e.get('price_change_percent'))
        predicted, likelihood = _tonight_projection(e.get('price_change_projections'))
        calibrating = bool(e.get('price_change_calibrating'))
        locked = False
        lu = e.get('price_change_locked_until')
        if lu:
            try:
                locked = datetime.fromisoformat(str(lu).replace('Z', '+00:00')) > now
            except ValueError:
                locked = False
        if calibrating:
            predicted = None
        rows.append({'id': e['id'], 'price_progress': progress, 'price_predicted': predicted,
                     'price_likelihood': likelihood,
                     'price_status': _price_status(predicted, progress, likelihood, locked, calibrating)})
    return pd.DataFrame(rows).set_index('id')


def apply_price_tracker(df, tracker):
    """Attach tracker columns. price_change_likelihood (also used by the
    Deadline Dashboard) becomes FPL's own tonight projection, falling back to
    current progress, then to the old transfer-based estimate."""
    if tracker is None or df is None or df.empty:
        return df
    df = df.drop(columns=[c for c in PRICE_TRACKER_COLS if c in df.columns])
    df = df.join(tracker, on='id')
    official = df['price_predicted'].fillna(df['price_progress'])
    if 'price_change_likelihood' in df.columns:
        df['price_change_likelihood'] = official.fillna(df['price_change_likelihood'])
    else:
        df['price_change_likelihood'] = official
    return df


def refresh_price_tracker():
    """Between the 3-hour full refreshes, re-read just FPL's price tracker
    (one bootstrap call) so the columns keep up with FPL's own updates."""
    with DATA_LOCK:
        if DATA.get('price_tracker_running'):
            return
        DATA['price_tracker_running'] = True
    try:
        tracker = read_fpl_price_tracker(fetch_bootstrap_data().get('elements', []))
        if tracker is not None and DATA.get('df_active') is not None:
            updated = apply_price_tracker(DATA['df_active'].copy(), tracker)
            with DATA_LOCK:
                DATA['df_active'] = updated
                DATA['price_tracker_at'] = time.time()
    except Exception as e:
        print(f"  Price tracker refresh failed (non-fatal): {e}")
    finally:
        with DATA_LOCK:
            DATA['price_tracker_running'] = False
            DATA['price_tracker_next'] = time.time() + PRICE_TRACKER_INTERVAL


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

    df['minutes_safe'] = df['minutes'].replace(0, np.nan)

    def per_90(api_col, numerator_col, decimals=2):
        """
        Prefer the per-90 value the API now ships; fall back to computing it.
        The API rounds to 2dp and returns 0 (not NaN) for zero-minute players,
        so we blank those out to keep the old NaN semantics.
        """
        if api_col in df.columns:
            vals = pd.to_numeric(df[api_col], errors='coerce')
            return vals.where(df['minutes'] > 0).round(decimals)
        return ((pd.to_numeric(df[numerator_col], errors='coerce')
                 / df['minutes_safe']) * 90).round(decimals)

    # Defensive Contribution calculations
    df['defcon'] = df['defensive_contribution'].fillna(0).astype(int)
    # API now supplies defensive_contribution_per_90 directly
    df['defcon_per_90'] = per_90('defensive_contribution_per_90', 'defcon')

    df['games_played'] = df['minutes_safe'] / 90
    # Threshold map comes from SEASON (GKP is excluded — GKPs score 0 for defcon)
    df['bonus_threshold'] = df['position'].map(SEASON['thresholds'])
    df['defcon_vs_bonus'] = df['defcon_per_90'] - df['bonus_threshold']
    df['bonus_rate'] = (df['defcon_per_90'] / df['bonus_threshold']) * 100

    # Season-aware baseline pool: 450 minutes is impossible in the first
    # weeks and an empty pool makes expected_defcon NaN for everyone
    _gws_done = sum(1 for e in data.get('events', []) if e.get('finished'))
    if _gws_done == 0 and any(e.get('is_current') for e in data.get('events', [])):
        _gws_done = 1
    _defcon_pool_mins = adaptive_min_minutes(450, _gws_done) if _gws_done else 450
    position_defcon_rates = df[df['minutes'] >= _defcon_pool_mins].groupby('position')['defcon_per_90'].mean()
    df['position_avg_defcon_rate'] = df['position'].map(position_defcon_rates)
    df['expected_defcon'] = (df['minutes_safe'] / 90) * df['position_avg_defcon_rate']
    df['defcon_diff'] = df['defcon'] - df['expected_defcon']

    df['points_per_million'] = df['total_points'] / df['price']
    df['xg_diff'] = df['goals_scored'] - df['expected_goals']
    df['xa_diff'] = df['assists'] - df['expected_assists']

    df['form'] = pd.to_numeric(df['form'], errors='coerce')
    df['ppg'] = pd.to_numeric(df['points_per_game'], errors='coerce')
    df['form_vs_season'] = (df['form'] - df['ppg']).round(1)
    df['ownership'] = pd.to_numeric(df['selected_by_percent'], errors='coerce')

    # API now supplies clean_sheets_per_90 / goals_conceded_per_90
    df['cs_per_90'] = per_90('clean_sheets_per_90', 'clean_sheets')
    df['gc_per_90'] = per_90('goals_conceded_per_90', 'goals_conceded')

    # Transfer columns
    df['transfers_in_gw'] = df['transfers_in_event']
    df['transfers_out_gw'] = df['transfers_out_event']
    df['net_transfers_gw'] = df['transfers_in_gw'] - df['transfers_out_gw']
    df['transfer_ratio'] = np.where(
        df['transfers_out_gw'] > 0,
        df['transfers_in_gw'] / df['transfers_out_gw'],
        df['transfers_in_gw']
    )
    df['cost_change_start'] = df['cost_change_start'] / 10
    df['cost_change_event'] = df['cost_change_event'] / 10
    df['cost_change_start_fall'] = df['cost_change_start_fall'] / 10

    # BPS columns
    df['bps_per_90'] = (df['bps'] / df['minutes_safe']) * 90
    df['bonus_per_90'] = (df['bonus'] / df['minutes_safe']) * 90

    # Underlying per-90 columns — xG/xA/xGI now come from the API
    df['xgi_per_90'] = per_90('expected_goal_involvements_per_90', 'expected_goal_involvements')
    df['xg_per_90'] = per_90('expected_goals_per_90', 'expected_goals')
    df['xa_per_90'] = per_90('expected_assists_per_90', 'expected_assists')
    # ICT components have no API per-90 equivalent — still computed
    df['threat_per_90'] = (df['threat'] / df['minutes_safe'] * 90).round(1)
    df['creativity_per_90'] = (df['creativity'] / df['minutes_safe'] * 90).round(1)
    df['ict_per_90'] = (df['ict_index'] / df['minutes_safe'] * 90).round(1)
    df['gi'] = df['goals_scored'] + df['assists']
    df['gi_per_90'] = (df['gi'] / df['minutes_safe'] * 90).round(2)
    df['xgi_diff'] = (df['gi'] - df['expected_goal_involvements']).round(2)
    df['xgi_diff_per_90'] = (df['gi_per_90'] - df['xgi_per_90']).round(2)

    # --- Availability & FPL's own expected points ---
    # chance_of_playing_next_round is None for unflagged players = fully fit
    if 'chance_of_playing_next_round' in df.columns:
        df['avail_pct'] = pd.to_numeric(
            df['chance_of_playing_next_round'], errors='coerce').fillna(100)
    else:
        df['avail_pct'] = 100.0
    df['news'] = df['news'].fillna('') if 'news' in df.columns else ''
    df['ep_next'] = pd.to_numeric(df['ep_next'], errors='coerce') if 'ep_next' in df.columns else np.nan
    df['event_points'] = pd.to_numeric(df['event_points'], errors='coerce') \
        if 'event_points' in df.columns else np.nan

    # --- Set-piece duties (P=penalties, C=corners/indirect FKs, F=direct FKs) ---
    # First or second in a duty order is a real xGI driver hiding in the payload.
    def _duty(col, tag):
        if col not in df.columns:
            return pd.Series('', index=df.index)
        order = pd.to_numeric(df[col], errors='coerce')
        return order.map(lambda v: f"{tag}{int(v)}" if pd.notna(v) and v <= 2 else '')

    # Numeric penalty rank for the projection engine (1 = first choice)
    df['pen_rank'] = pd.to_numeric(df['penalties_order'], errors='coerce') \
        if 'penalties_order' in df.columns else np.nan

    duties = pd.concat([
        _duty('penalties_order', 'P'),
        _duty('corners_and_indirect_freekicks_order', 'C'),
        _duty('direct_freekicks_order', 'F'),
    ], axis=1)
    df['set_pieces'] = duties.apply(lambda r: ' '.join(x for x in r if x), axis=1)

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


def season_has_started(data):
    """
    True once the season is live. IMPORTANT: FPL only marks a gameweek
    `finished` after ALL its matches are played and bonus is confirmed —
    often days after football has actually happened. `is_current` flips at
    the deadline, and player histories populate match by match from then on.
    Waiting for `finished` made Phase 2 skip the entire history fetch all
    weekend, blanking the consistency tab while GW1 was mid-flight.
    """
    return any(e.get('finished') or e.get('is_current')
               for e in data.get('events', []))


def get_target_gw_num(data):
    """
    The gameweek the user is planning FOR — i.e. the next deadline.

    Pre-season nothing is `is_current`, so the old `current_gw['id'] + 1`
    pattern silently resolved to GW2 and skipped GW1 entirely. FPL flips
    `is_next` correctly at every deadline, so read it directly.
    """
    nxt = get_next_gameweek(data)
    if nxt:
        return nxt['id']
    cur = get_current_gameweek(data)
    if cur:
        return cur['id']
    # Season over (or events missing) — fall back to the last event we know of
    events = data.get('events', [])
    return events[-1]['id'] if events else 1


# =============================================================================
# STYLING
# =============================================================================

# Fantasy Premier League's own palette: deep purple, electric green, cyan
# and their magenta-red. The bright colours are for fills (bars, pills,
# markers); the *_text variants are darker versions that stay readable as
# text on white, since #00ff87 or #04f5ff text on white is almost invisible.
COLORS = {
    'primary': '#37003c',       # FPL purple
    'secondary': '#00ff87',     # FPL green
    'accent': '#e90052',        # FPL magenta-red
    'cyan': '#04f5ff',          # FPL cyan
    'background': '#f5f5f5',
    'card_bg': '#ffffff',
    'text_dark': '#37003c',
    'text_light': '#6b5c70',
    'success': '#00ff87',
    'warning': '#ffab1b',
    'danger': '#e90052',
    'info': '#04f5ff',
    'success_text': '#00813f',
    'warning_text': '#8a5300',
    'danger_text': '#c4003f',
    'info_text': '#00717f',
}
import plotly.io as pio
_tpl = pio.templates['plotly_white']
_tpl.layout.font = dict(family="Outfit, Arial, sans-serif", color='#37003c')
pio.templates['plotly_white'] = _tpl

# FPL's typeface, Premier Sans, is a bespoke licensed font that can't be
# served from here; Outfit is the closest free geometric match.
FONT_FAMILY = "Outfit, Arial, sans-serif"

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
    'fontFamily': FONT_FAMILY,
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
# FETCH AND PROCESS DATA €” WRAPPED FOR AUTO-REFRESH
# =============================================================================

import time
import threading

# =============================================================================
# SHOT INTELLIGENCE — where chances are created and conceded (Understat)
# =============================================================================
# The FPL API has no location data, so this pulls shot-level data from
# Understat's JSON endpoints: every shot with pitch coordinates, xG, situation
# (open play / corner / free kick / penalty), body part and the action that
# set it up (cross, through ball, ...), plus per-match rosters for minutes and
# left/right positions.
#
# Fetch-once-per-match: a finished match never changes, so each one is
# downloaded once, stored in the snapshot SQLite DB and never re-requested.
# Current FPL season only. Understat is an unofficial source, so every fetch
# fails soft — nothing else in the app depends on it.

UNDERSTAT_BASE = 'https://understat.com/'
UNDERSTAT_HEADERS = {
    # Understat's data endpoints only answer JSON to requests that look like
    # its own page's XHR calls — without these the response is HTML or a 404.
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/126.0 Safari/537.36'),
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': UNDERSTAT_BASE,
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}
SHOTS_REFRESH_INTERVAL = float(os.environ.get('SHOTS_REFRESH_INTERVAL_HOURS', '6')) * 3600
# Cap per sync so a from-scratch catch-up late in the season is spread over a
# few runs (10 minutes apart) instead of one long burst of requests.
SHOTS_MAX_PER_RUN = int(os.environ.get('SHOTS_MAX_MATCHES_PER_RUN', '60'))
SHOTS_REQUEST_GAP = 1.5          # seconds between match requests — be polite
# Matches of league-average "prior" each team's zone profile is shrunk toward.
# Same idea as FPL_TEAM_FORM_K: early-season profiles lean on the league,
# and earn their way out as real matches accumulate.
SHOT_PROFILE_K = float(os.environ.get('FPL_SHOT_PROFILE_K', '6'))

# Pitch geometry on Understat's 0-1 grid: X runs towards the goal being
# attacked (1 = goal line), Y runs across the pitch.
SHOT_BOX_X, SHOT_SIX_X = 0.83, 0.945
SHOT_BOX_Y = (0.211, 0.789)
SHOT_CENTRE_Y = (0.368, 0.632)   # six-yard-box width = the central channel
SHOT_BIG_CHANCE_XG = 0.30        # "high-quality chance" threshold

SET_PIECE_SITUATIONS = {'FromCorner', 'SetPiece', 'DirectFreekick'}
# Mutually exclusive split of every non-penalty shot. Flanks are always in
# the ATTACKING team's frame: 'op_left' = attacking down their own left.
SHOT_COMPONENTS = ['set_piece', 'cross', 'op_centre', 'op_left', 'op_right']
COMP_LABEL_ATT = {
    'set_piece': 'Set pieces', 'cross': 'Crosses', 'op_centre': 'Open play: central',
    'op_left': 'Open play: down their left', 'op_right': 'Open play: down their right',
}
# A defence conceding down ITS left is being attacked down the attacker's right.
COMP_LABEL_DEF = {
    'set_piece': 'Set pieces', 'cross': 'Crosses', 'op_centre': 'Open play: central',
    'op_left': 'Open play: down their right', 'op_right': 'Open play: down their left',
}
COMP_TAG = {'set_piece': 'Set pcs', 'cross': 'Crosses', 'op_centre': 'Central',
            'op_left': 'Left', 'op_right': 'Right'}

# Understat roster positions carry a side, which gives each player's flank
# directly and lets the app check Understat's Y orientation from the data.
_US_POS_SIDE = {
    'DL': 'left', 'DML': 'left', 'ML': 'left', 'AML': 'left', 'FWL': 'left',
    'DR': 'right', 'DMR': 'right', 'MR': 'right', 'AMR': 'right', 'FWR': 'right',
}


def _season_start_year():
    """FPL season start year (2026/27 -> 2026), which is also how Understat
    keys its seasons."""
    try:
        return int(str(SEASON.get('label', ''))[:4])
    except (TypeError, ValueError):
        now = datetime.now()
        return now.year if now.month >= 7 else now.year - 1


def _us_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _us_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _shots_conn():
    conn = sqlite3.connect(SNAPSHOT_DB_PATH, timeout=30)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS us_matches (
            match_id INTEGER PRIMARY KEY, season INTEGER, kickoff TEXT,
            h_team TEXT, a_team TEXT, h_short TEXT, a_short TEXT,
            h_goals INTEGER, a_goals INTEGER, fetched_at REAL
        );
        CREATE TABLE IF NOT EXISTS us_shots (
            shot_id INTEGER PRIMARY KEY, match_id INTEGER, season INTEGER,
            minute INTEGER, result TEXT, x REAL, y REAL, xg REAL,
            player TEXT, player_id INTEGER, side TEXT, team TEXT, opponent TEXT,
            situation TEXT, shot_type TEXT, last_action TEXT, assisted_by TEXT
        );
        CREATE TABLE IF NOT EXISTS us_rosters (
            match_id INTEGER, player_id INTEGER, season INTEGER, team TEXT,
            player TEXT, position TEXT, minutes INTEGER, key_passes INTEGER,
            xa REAL, PRIMARY KEY (match_id, player_id)
        );
        CREATE INDEX IF NOT EXISTS ix_us_shots_season ON us_shots(season);
        CREATE INDEX IF NOT EXISTS ix_us_rosters_season ON us_rosters(season);
    """)
    return conn


_US_SESSION = None


def _understat_get(path):
    global _US_SESSION
    if _US_SESSION is None:
        _US_SESSION = requests.Session()
        _US_SESSION.headers.update(UNDERSTAT_HEADERS)
    r = _US_SESSION.get(UNDERSTAT_BASE + path, timeout=20)
    r.raise_for_status()
    return r.json()   # ValueError if Understat served HTML instead


def parse_understat_match(match_id, payload, season, meta):
    """getMatchData payload -> (shot rows, roster rows) for the DB. Own goals
    are dropped: they aren't a chance the scoring side created."""
    payload = payload or {}
    shot_rows, roster_rows = [], []
    teams = {'h': meta.get('h_team'), 'a': meta.get('a_team')}
    shots = payload.get('shots') or {}
    for side in ('h', 'a'):
        team = teams[side]
        opp = teams['a' if side == 'h' else 'h']
        for s in (shots.get(side) or []):
            if s.get('result') == 'OwnGoal':
                continue
            shot_rows.append((
                _us_int(s.get('id')), match_id, season, _us_int(s.get('minute')),
                s.get('result'), _us_float(s.get('X')), _us_float(s.get('Y')),
                _us_float(s.get('xG')), s.get('player'), _us_int(s.get('player_id')),
                side, team, opp, s.get('situation'), s.get('shotType'),
                s.get('lastAction'), s.get('player_assisted') or None,
            ))
    rosters = payload.get('rosters') or {}
    for side in ('h', 'a'):
        block = rosters.get(side) or {}
        entries = block.values() if isinstance(block, dict) else block
        for r in entries:
            roster_rows.append((
                match_id, _us_int(r.get('player_id')), season, teams[side],
                r.get('player'), r.get('position'), _us_int(r.get('time')),
                _us_int(r.get('key_passes')), _us_float(r.get('xA')),
            ))
    return shot_rows, roster_rows


def sync_understat_shots():
    """Download any newly finished matches, then rebuild the shot model.
    Runs in a background thread; never raises."""
    with DATA_LOCK:
        if DATA.get('shots_syncing'):
            return
        DATA['shots_syncing'] = True
    error, fetched, pending = None, 0, 0
    try:
        season = _season_start_year()
        conn = _shots_conn()
        try:
            try:
                league = _understat_get(f'getLeagueData/EPL/{season}')
                dates = league.get('dates') or []
            except Exception as e:
                dates = []
                error = f"Understat unreachable ({type(e).__name__})"
                print(f"  Shot data: {error}: {e}")
            finished = [d for d in dates if str(d.get('isResult')).lower() in ('true', '1')]
            have = {row[0] for row in conn.execute(
                "SELECT match_id FROM us_matches WHERE season = ?", (season,))}
            todo = sorted((d for d in finished if _us_int(d.get('id')) not in have),
                          key=lambda d: d.get('datetime') or '')
            pending = max(0, len(todo) - SHOTS_MAX_PER_RUN)
            if todo:
                print(f"  Shot data: fetching {min(len(todo), SHOTS_MAX_PER_RUN)} new "
                      f"match(es){f', {pending} queued for the next run' if pending else ''}")
            for d in todo[:SHOTS_MAX_PER_RUN]:
                mid = _us_int(d.get('id'))
                h, a = d.get('h') or {}, d.get('a') or {}
                meta = {'h_team': h.get('title'), 'a_team': a.get('title')}
                try:
                    payload = _understat_get(f'getMatchData/{mid}')
                except Exception as e:
                    error = f"Match fetch failed ({type(e).__name__}); will retry"
                    print(f"  Shot data: match {mid} failed: {e}")
                    pending += 1
                    break
                shot_rows, roster_rows = parse_understat_match(mid, payload, season, meta)
                goals = d.get('goals') or {}
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO us_shots VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", shot_rows)
                    conn.executemany(
                        "INSERT OR REPLACE INTO us_rosters VALUES (?,?,?,?,?,?,?,?,?)",
                        roster_rows)
                    conn.execute(
                        "INSERT OR REPLACE INTO us_matches VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (mid, season, d.get('datetime'), meta['h_team'], meta['a_team'],
                         h.get('short_title'), a.get('short_title'),
                         _us_int(goals.get('h')), _us_int(goals.get('a')), time.time()))
                fetched += 1
                time.sleep(SHOTS_REQUEST_GAP)

            shots = pd.read_sql_query("SELECT * FROM us_shots WHERE season = ?", conn,
                                      params=(season,))
            rosters = pd.read_sql_query("SELECT * FROM us_rosters WHERE season = ?", conn,
                                        params=(season,))
            matches = pd.read_sql_query("SELECT * FROM us_matches WHERE season = ?", conn,
                                        params=(season,))
        finally:
            conn.close()

        changed = fetched > 0 or DATA.get('shot_intel') is None
        intel = build_shot_intel(shots, rosters, matches, DATA.get('teams_df'),
                                 DATA.get('fixtures_data'), DATA.get('df_active'))
        with DATA_LOCK:
            DATA['shot_intel'] = intel
            if changed and intel is not None:
                DATA['shots_version'] = int(time.time())
            DATA['shots_status'] = {
                'synced_at': time.time(), 'matches': int(len(matches)),
                'pending': pending, 'error': error, 'fetched': fetched,
            }
        if intel is not None:
            print(f"  Shot data: {len(matches)} matches, {len(shots)} shots stored "
                  f"({fetched} new). Orientation: {intel['orientation_note']}")
            if intel.get('unmatched_teams'):
                print(f"  Shot data WARNING: unmatched Understat teams "
                      f"{intel['unmatched_teams']} — add them to _ODDS_TEAM_ALIASES")
    except Exception as e:
        error = f"Shot model failed ({type(e).__name__})"
        print(f"  Shot data ERROR: {e}")
        with DATA_LOCK:
            DATA['shots_status'] = {'synced_at': time.time(), 'error': error,
                                    'pending': pending, 'fetched': fetched}
    finally:
        with DATA_LOCK:
            DATA['shots_syncing'] = False
            retry_soon = pending > 0 or error is not None
            DATA['shots_next_due'] = time.time() + (600 if retry_soon else SHOTS_REFRESH_INTERVAL)


def check_shots_sync():
    """Kick off a background shot sync when due. Cheap to call often."""
    if DATA.get('shots_syncing') or time.time() < DATA.get('shots_next_due', 0):
        return
    if DATA.get('teams_df') is None:
        return   # FPL data not loaded yet — team mapping needs it
    with DATA_LOCK:
        DATA['shots_next_due'] = time.time() + 600   # debounce concurrent callers
    threading.Thread(target=sync_understat_shots, daemon=True).start()


# --- Model ------------------------------------------------------------------

def _classify_shots(shots, y0_is_right):
    """Add zone / channel / component columns in the attacking team's frame."""
    s = shots.copy()
    x, y = s['x'], s['y']
    in_centre = (y >= SHOT_CENTRE_Y[0]) & (y <= SHOT_CENTRE_Y[1])
    in_box_w = (y >= SHOT_BOX_Y[0]) & (y <= SHOT_BOX_Y[1])
    s['zone'] = np.select(
        [(x >= SHOT_SIX_X) & in_centre, (x >= SHOT_BOX_X) & in_centre,
         (x >= SHOT_BOX_X) & in_box_w],
        ['six_yard', 'central_box', 'wide_box'], 'outside_box')
    low, high = y < SHOT_CENTRE_Y[0], y > SHOT_CENTRE_Y[1]
    right = low if y0_is_right else high
    left = high if y0_is_right else low
    s['channel'] = np.select([left, right], ['left', 'right'], 'centre')
    # Horizontal plot coordinate with the attacked goal at the top: the
    # attacker's right is on the right of the picture.
    s['hx'] = (1 - y) if y0_is_right else y
    sit = s['situation'].fillna('')
    pen = sit == 'Penalty'
    sp = sit.isin(SET_PIECE_SITUATIONS)
    cross = (~sp) & (~pen) & (s['last_action'].fillna('') == 'Cross')
    s['component'] = np.select(
        [pen, sp, cross, s['channel'] == 'left', s['channel'] == 'right'],
        ['penalty', 'set_piece', 'cross', 'op_left', 'op_right'], 'op_centre')
    s['is_goal'] = s['result'].fillna('') == 'Goal'
    return s


def _detect_orientation(shots, rosters):
    """Which way does Understat's Y axis run? Measured, not assumed: players
    listed on the right (AMR, MR, DR, ...) should shoot from right of centre.
    Falls back to 'Y=0 is the attacker's right' until there's enough data."""
    if rosters is None or rosters.empty or shots.empty:
        return True, 'default (not enough data yet to check)'
    side = rosters[['match_id', 'player_id', 'position']].copy()
    side['pside'] = side['position'].map(_US_POS_SIDE)
    m = shots.merge(side[['match_id', 'player_id', 'pside']],
                    on=['match_id', 'player_id'], how='left')
    r_y = m.loc[m['pside'] == 'right', 'y']
    l_y = m.loc[m['pside'] == 'left', 'y']
    if len(r_y) < 25 or len(l_y) < 25:
        return True, 'default (not enough data yet to check)'
    y0_right = bool(r_y.mean() < l_y.mean())
    gap = abs(r_y.mean() - l_y.mean())
    return y0_right, f"measured from {len(r_y) + len(l_y)} wide-player shots (separation {gap:.2f})"


def _match_counts(matches, team_map, match_ids=None):
    mm = matches if match_ids is None else matches[matches['match_id'].isin(match_ids)]
    ids = pd.concat([mm['h_team'], mm['a_team']]).map(team_map).dropna().astype(int)
    return ids.value_counts().to_dict()


def _profiles_from(np_shots, n_by_team):
    """Shrunk per-match xG rates by component, created and conceded."""
    n_team_matches = sum(n_by_team.values())
    if n_team_matches == 0 or np_shots.empty:
        return None, None
    tot = np_shots.groupby('component')['xg'].sum()
    league = {c: float(tot.get(c, 0.0)) / n_team_matches for c in SHOT_COMPONENTS}
    made = np_shots.groupby(['team_id', 'component'])['xg'].sum()
    conc = np_shots.groupby(['opp_id', 'component'])['xg'].sum()
    K = SHOT_PROFILE_K
    profiles = {}
    for tid, n in n_by_team.items():
        cr, cc = {}, {}
        for c in SHOT_COMPONENTS:
            cr[c] = (float(made.get((tid, c), 0.0)) + K * league[c]) / (n + K)
            cc[c] = (float(conc.get((tid, c), 0.0)) + K * league[c]) / (n + K)
        profiles[tid] = {'n': int(n), 'created': cr, 'conceded': cc,
                         'created_total': sum(cr.values()), 'conceded_total': sum(cc.values())}
    return profiles, league


def shot_matchup(att, dfn, league):
    """
    Style-aware expected npxG for `att` attacking `dfn`, vs a strength-only
    baseline. Each component: attack rate x defence leak rate / league rate.
    If both sides have league-typical mixes the two agree exactly; the gap is
    the matchup — an attack built on crosses meeting a defence that leaks
    them. Returns (baseline, style_total, per-component xG, per-component
    edge vs what a league-typical mix would give).
    """
    lt = sum(league.values())
    if not att or not dfn or lt <= 0:
        return None
    base = att['created_total'] * dfn['conceded_total'] / lt
    comps, edges = {}, {}
    for c in SHOT_COMPONENTS:
        if league[c] <= 0:
            continue
        comps[c] = att['created'][c] * dfn['conceded'][c] / league[c]
        edges[c] = comps[c] - base * league[c] / lt
    return base, sum(comps.values()), comps, edges


def _validate_matchups(np_shots, matches, team_map, fixtures):
    """
    Out-of-sample check: for each gameweek, build profiles only from earlier
    gameweeks, predict each team's npxG in that gameweek both ways, and
    compare with what actually happened. Tells you whether the style model
    earns its keep before you lean on it.
    """
    if not fixtures or matches.empty:
        return None
    ev = {(f.get('team_h'), f.get('team_a')): f.get('event') for f in fixtures}
    mm = matches.copy()
    mm['h_id'] = mm['h_team'].map(team_map)
    mm['a_id'] = mm['a_team'].map(team_map)
    mm['gw'] = [ev.get((h, a)) for h, a in zip(mm['h_id'], mm['a_id'])]
    mm = mm.dropna(subset=['gw', 'h_id', 'a_id'])
    if mm.empty:
        return None
    actual = np_shots.groupby(['match_id', 'team_id'])['xg'].sum()
    err_b, err_s, closer = [], [], 0
    for g in sorted(mm['gw'].unique()):
        train = mm[mm['gw'] < g]
        test = mm[mm['gw'] == g]
        if train.empty:
            continue
        n_by = _match_counts(train, team_map)
        if len(n_by) < 16 or min(n_by.values()) < 3:
            continue   # too early: most teams need 3+ matches of history
        prof, league = _profiles_from(np_shots[np_shots['match_id'].isin(train['match_id'])], n_by)
        if not prof:
            continue
        for _, r in test.iterrows():
            for att_id, def_id in ((r['h_id'], r['a_id']), (r['a_id'], r['h_id'])):
                res = shot_matchup(prof.get(att_id), prof.get(def_id), league)
                if res is None:
                    continue
                base, style = res[0], res[1]
                act = float(actual.get((r['match_id'], att_id), 0.0))
                eb, es = abs(base - act), abs(style - act)
                err_b.append(eb)
                err_s.append(es)
                closer += es < eb
    if not err_b:
        return None
    n = len(err_b)
    return {'n': n, 'mae_base': float(np.mean(err_b)), 'mae_style': float(np.mean(err_s)),
            'closer_pct': 100.0 * closer / n}


def _norm_person(s):
    import unicodedata
    import re
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace('-', ' ').replace("'", ''))
    return ' '.join(s.split())


def _map_players_to_fpl(us_players, df_fpl):
    """Understat player -> FPL element id, matched within the same club.
    Returns {understat_player_id: fpl_id}. Ambiguous matches are left out
    rather than guessed."""
    import difflib
    if df_fpl is None or df_fpl.empty:
        return {}
    cands = {}
    for r in df_fpl.itertuples(index=False):
        first = _norm_person(getattr(r, 'first_name', '') or '')
        second = _norm_person(getattr(r, 'second_name', '') or '')
        web = _norm_person(r.web_name)
        full = f"{first} {second}".strip() or web
        cands.setdefault(r.team, []).append((r.id, full, web, first, second))
    out = {}
    for pid, name, tid in us_players[['player_id', 'player', 'team_id']].itertuples(index=False):
        if pd.isna(tid):
            continue
        pool = cands.get(int(tid), [])
        nm = _norm_person(name)
        toks = set(nm.split())
        if not nm or not pool:
            continue
        strategies = (
            lambda c: c[1] == nm,
            lambda c: c[2] == nm,
            lambda c: toks <= set(c[1].split()) or set(c[1].split()) <= toks,
            lambda c: bool(c[4]) and nm.split()[-1] == c[4].split()[-1]
                      and c[3][:1] == nm[:1],
            lambda c: bool(c[2]) and set(c[2].split()) <= toks,
        )
        hit = None
        for test in strategies:
            m = [c for c in pool if test(c)]
            if len(m) == 1:
                hit = m[0][0]
                break
            if len(m) > 1:
                break   # ambiguous at this level — don't guess
        if hit is None:
            best = difflib.get_close_matches(nm, [c[1] for c in pool], n=2, cutoff=0.85)
            if len(best) == 1:
                hit = next(c[0] for c in pool if c[1] == best[0])
        if hit is not None:
            out[int(pid)] = int(hit)
    return out


def _player_shot_table(s, rosters, team_map, df_fpl):
    """Per-player chance quality, chance creation and flank, FPL-linked."""
    if rosters is None or rosters.empty:
        return pd.DataFrame()
    ro = rosters.copy()
    ro['team_id'] = ro['team'].map(team_map)
    mins = ro.groupby('player_id')['minutes'].sum()
    latest = ro.sort_values('match_id').groupby('player_id').tail(1).set_index('player_id')
    started = ro[ro['position'].fillna('Sub') != 'Sub']
    side_mode = started.groupby('player_id')['position'].agg(
        lambda p: Counter(_US_POS_SIDE.get(x, 'centre') for x in p).most_common(1)[0][0])

    np_s = s[s['component'] != 'penalty']
    g = np_s.groupby('player_id')
    t = pd.DataFrame({
        'shots': g.size(),
        'npxg': g['xg'].sum(),
        'box_shots': g['zone'].apply(lambda z: (z != 'outside_box').sum()),
        'big_chances': g['xg'].apply(lambda x: (x >= SHOT_BIG_CHANCE_XG).sum()),
        'headers': g['shot_type'].apply(lambda x: (x == 'Head').sum()),
        'sp_npxg': g.apply(lambda d: d.loc[d['component'] == 'set_piece', 'xg'].sum()),
        'np_goals': g['is_goal'].sum(),
    })
    # Chances created: shots a player set up, linked back to his roster id
    ast = np_s.dropna(subset=['assisted_by'])[['match_id', 'team', 'assisted_by', 'xg',
                                                'component']]
    ast = ast.merge(ro[['match_id', 'team', 'player', 'player_id']].rename(
        columns={'player': 'assisted_by', 'player_id': 'creator_id'}),
        on=['match_id', 'team', 'assisted_by'], how='inner')
    ga = ast.groupby('creator_id')
    created = pd.DataFrame({
        'chances_created': ga.size(),
        'xa': ga['xg'].sum(),
        'crosses_created': ga['component'].apply(lambda c: (c == 'cross').sum()),
        'sp_created': ga['component'].apply(lambda c: (c == 'set_piece').sum()),
        'sp_xa': ga.apply(lambda d: d.loc[d['component'] == 'set_piece', 'xg'].sum()),
    })
    base = pd.DataFrame({'minutes': mins})
    base = base.join(latest[['player', 'team_id']]).join(t).join(created)
    base['side'] = side_mode.reindex(base.index).fillna('centre')
    base = base.reset_index().rename(columns={'index': 'player_id'})
    fill0 = ['shots', 'npxg', 'box_shots', 'big_chances', 'headers', 'sp_npxg', 'np_goals',
             'chances_created', 'xa', 'crosses_created', 'sp_created', 'sp_xa']
    base[fill0] = base[fill0].fillna(0)
    base = base[(base['shots'] > 0) | (base['chances_created'] > 0)]
    m90 = base['minutes'].where(base['minutes'] > 0) / 90
    base['shots_90'] = base['shots'] / m90
    base['npxg_90'] = base['npxg'] / m90
    base['xa_90'] = base['xa'] / m90
    base['chances_90'] = base['chances_created'] / m90
    base['big_90'] = base['big_chances'] / m90
    base['sp_threat_90'] = (base['sp_npxg'] + base['sp_xa']) / m90
    base['xg_per_shot'] = base['npxg'] / base['shots'].where(base['shots'] > 0)
    base['box_pct'] = 100 * base['box_shots'] / base['shots'].where(base['shots'] > 0)
    base['head_pct'] = 100 * base['headers'] / base['shots'].where(base['shots'] > 0)
    base['cross_pct'] = 100 * base['crosses_created'] / base['chances_created'].where(
        base['chances_created'] > 0)
    fmap = _map_players_to_fpl(base, df_fpl)
    base['fpl_id'] = base['player_id'].map(fmap)
    return base


def build_shot_intel(shots, rosters, matches, teams_df, fixtures, df_active):
    """Everything the Shot Intelligence pages need, computed once per sync."""
    if shots is None or shots.empty or matches is None or matches.empty \
            or teams_df is None or teams_df.empty:
        return None
    fpl_names = dict(zip(teams_df['id'], teams_df['name']))
    id_by_short = {str(v).upper(): k for k, v in zip(teams_df['id'], teams_df['short_name'])}
    shorts = dict(zip(matches['h_team'], matches['h_short']))
    shorts.update(zip(matches['a_team'], matches['a_short']))
    team_map, unmatched = {}, []
    for title in sorted(set(matches['h_team']) | set(matches['a_team'])):
        tid = match_odds_team_to_fpl(title, fpl_names)
        if tid is None:
            tid = id_by_short.get(str(shorts.get(title) or '').upper())
        if tid is None:
            unmatched.append(title)
        else:
            team_map[title] = int(tid)

    y0_right, orient_note = _detect_orientation(shots, rosters)
    s = _classify_shots(shots, y0_right)
    s['team_id'] = s['team'].map(team_map)
    s['opp_id'] = s['opponent'].map(team_map)
    s = s.dropna(subset=['team_id', 'opp_id'])
    s['team_id'] = s['team_id'].astype(int)
    s['opp_id'] = s['opp_id'].astype(int)
    np_s = s[s['component'] != 'penalty']

    n_by = _match_counts(matches, team_map)
    profiles, league = _profiles_from(np_s, n_by)
    if not profiles:
        return None

    # Raw (unshrunk) descriptors for the team table
    team_rows = []
    for tid, p in profiles.items():
        n = max(p['n'], 1)
        mine = np_s[np_s['team_id'] == tid]
        vs = np_s[np_s['opp_id'] == tid]
        op_vs = vs[vs['component'] != 'set_piece']
        op_mine = mine[mine['component'] != 'set_piece']
        tot_vs_op = op_vs['xg'].sum()
        def _pct(num, den):
            return round(100 * num / den, 1) if den > 0 else None
        team_rows.append({
            'team_id': tid, 'team': fpl_names.get(tid, str(tid)), 'matches': p['n'],
            'npxg_for_pm': round(mine['xg'].sum() / n, 2),
            'npxg_against_pm': round(vs['xg'].sum() / n, 2),
            'sp_for_pm': round(mine.loc[mine['component'] == 'set_piece', 'xg'].sum() / n, 2),
            'sp_against_pm': round(vs.loc[vs['component'] == 'set_piece', 'xg'].sum() / n, 2),
            'cross_pct_for': _pct(op_mine.loc[op_mine['component'] == 'cross', 'xg'].sum(),
                                  op_mine['xg'].sum()),
            'box_pct_for': _pct(mine.loc[mine['zone'] != 'outside_box', 'xg'].sum(),
                                mine['xg'].sum()),
            'box_pct_against': _pct(vs.loc[vs['zone'] != 'outside_box', 'xg'].sum(),
                                    vs['xg'].sum()),
            # Defender's perspective: attacker's right = the defence's left
            'conc_their_left_pct': _pct(op_vs.loc[op_vs['channel'] == 'right', 'xg'].sum(), tot_vs_op),
            'conc_centre_pct': _pct(op_vs.loc[op_vs['channel'] == 'centre', 'xg'].sum(), tot_vs_op),
            'conc_their_right_pct': _pct(op_vs.loc[op_vs['channel'] == 'left', 'xg'].sum(), tot_vs_op),
        })

    players = _player_shot_table(s, rosters, team_map, df_active)
    validation = _validate_matchups(np_s, matches, team_map, fixtures)
    last_kick = str(matches['kickoff'].dropna().max() or '')[:10]
    s = s.merge(matches[['match_id', 'kickoff']], on='match_id', how='left')
    return {
        'shots': s[['match_id', 'kickoff', 'team_id', 'opp_id', 'player', 'player_id', 'minute', 'x', 'hx',
                    'xg', 'result', 'is_goal', 'situation', 'shot_type', 'last_action',
                    'zone', 'channel', 'component']].reset_index(drop=True),
        'profiles': profiles, 'league': league, 'team_table': pd.DataFrame(team_rows),
        'players': players, 'validation': validation, 'team_map': team_map,
        'unmatched_teams': unmatched, 'y0_is_right': y0_right,
        'orientation_note': orient_note, 'n_matches': int(len(matches)),
        'last_match_date': last_kick,
    }


# Global data store
DATA = {
    'last_refresh': 0,
    'refreshing': False,
    'odds_last_refresh': 0,
}
DATA_LOCK = threading.Lock()
REFRESH_INTERVAL = 3 * 60 * 60  # 3 hours in seconds

# The-odds-api free tier is 500 credits/month, and an h2h+totals call costs
# markets x regions = 2 credits. Polling it on the same 3-hour cadence as
# core stats (8x/day) burns ~480-496 credits/month on the routine cycle
# alone, before a single Render restart or local dev run — i.e. it silently
# exhausts the free quota most months (fetch_market_lambdas fails soft, so
# this would never surface as an error, just as the market blend quietly
# stopping). Odds also don't move hour-to-hour the way stats do; the value
# is in catching team news that lands well before kickoff, not in polling
# every 3 hours. Override via ODDS_REFRESH_INTERVAL_HOURS if needed.
ODDS_REFRESH_INTERVAL = int(os.environ.get('ODDS_REFRESH_INTERVAL_HOURS', 6)) * 60 * 60
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fpl_cache.pkl')

# Keys to persist in cache (excludes transient flags like 'refreshing')
_CACHE_KEYS = [
    'bootstrap_data', 'df', 'df_active', 'current_gw', 'next_gw',
    'total_managers', 'fixtures_data', 'teams_df', 'fixture_difficulty',
    'player_histories', 'sorted_teams', 'next_gw_num', 'last_refresh',
    'heavy_loaded', 'fixture_anchor_gw', 'season_started', 'season_label',
    'last_season_priors', 'calibration',
    'xg_ledger', 'odds_lambdas', 'odds_last_refresh', 'xcs_next', 'delta_basis',
]

# Bump whenever the shape of cached data changes, or at a season rollover.
# A mismatch (or an over-age cache) forces a clean fetch instead of serving
# last season's teams and players from disk.
CACHE_VERSION = 10
# Render sets RENDER_GIT_COMMIT on every deploy. Stamping the cache with it
# means a new deploy never reuses data pickled by an older build (or a local
# run that ended up in the repo), so "Updated" always resets on deploy.
BUILD_ID = os.environ.get('RENDER_GIT_COMMIT', 'local')
MAX_CACHE_AGE = REFRESH_INTERVAL


def save_cache():
    """Persist current DATA to disk so next startup is instant."""
    try:
        payload = {k: DATA[k] for k in _CACHE_KEYS if k in DATA}
        payload['_cache_version'] = CACHE_VERSION
        payload['_season_label'] = SEASON['label']
        payload['_build'] = BUILD_ID
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(CACHE_PATH) / (1024 * 1024)
        print(f"  Cache saved ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Warning: failed to save cache: {e}")


def load_cache():
    """
    Load cached DATA from disk. Returns True if cache was loaded successfully.
    """
    if not os.path.exists(CACHE_PATH):
        print("  No cache file found — will fetch from API")
        return False
    try:
        with open(CACHE_PATH, 'rb') as f:
            cached = pickle.load(f)

        age = time.time() - cached.get('last_refresh', 0)
        age_hrs = age / 3600

        # Reject a cache written by an older build or a previous season —
        # otherwise startup silently serves last season's teams and players.
        if cached.get('_cache_version') != CACHE_VERSION:
            print(f"  Cache version mismatch "
                  f"(found {cached.get('_cache_version')}, need {CACHE_VERSION}) — refetching")
            return False
        if age > MAX_CACHE_AGE:
            print(f"  Cache is {age_hrs:.1f}h old (limit {MAX_CACHE_AGE / 3600:.0f}h) — refetching")
            return False
        if cached.get('_build', 'local') != BUILD_ID:
            print(f"  Cache is from build {str(cached.get('_build'))[:7]}, this is "
                  f"{BUILD_ID[:7]} — new deploy, refetching")
            return False

        print(f"  Cache found ({age_hrs:.1f}h old, season {cached.get('_season_label')}) — loading...")
        cached.pop('_cache_version', None)
        cached.pop('_season_label', None)
        cached.pop('_build', None)
        with DATA_LOCK:
            DATA.update(cached)
            DATA['refreshing'] = False
        # Rebuild SEASON from the cached bootstrap so photo URLs, thresholds
        # and the season label match what the cache was built against.
        if cached.get('bootstrap_data'):
            build_season_config(cached['bootstrap_data'])
        print(f"  Cache loaded — all tabs ready")
        return True
    except Exception as e:
        print(f"  Warning: failed to load cache: {e}")
        return False


def refresh_core_data():
    """
    Phase 1: Fetch bootstrap + fixtures only (2 API calls).
    Gives us team names, player stats, fixture difficulty, next-fixture info,
    transfers, differentials — everything EXCEPT individual player histories.
    Fast enough to run synchronously at startup (~2-3 seconds).
    """
    with DATA_LOCK:
        if DATA.get('refreshing'):
            return
        DATA['refreshing'] = True

    try:
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: Core data refresh ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'=' * 60}")

        print("Fetching FPL data...")
        bootstrap_data = fetch_bootstrap_data()
        build_season_config(bootstrap_data)
        df = process_player_data(bootstrap_data)
        current_gw = get_current_gameweek(bootstrap_data)
        next_gw = get_next_gameweek(bootstrap_data)
        started = season_has_started(bootstrap_data)

        # Pre-season FPL zeroes every player's minutes, so a `minutes > 0` filter
        # empties df_active and blanks the whole app. Fall back to the API's
        # `can_select` flag until the first gameweek has actually been played.
        if started:
            df_active = df[df['minutes'] > 0].copy()
        elif 'can_select' in df.columns:
            df_active = df[df['can_select'].fillna(True)].copy()
        else:
            df_active = df[df['status'] != 'u'].copy()
        print(f"  Season started: {started} | active players: {len(df_active)}")

        # Initialise consistency columns as NaN (Phase 2 will populate these)
        for col in ['p_10', 'p_15', 'sim_mean', 'captain_gain',
                    'qualifying_games', 'bonus_games', 'hit_rate', 'starts_60',
                    'bonus_games_60', 'defcon_per_90_games', 'defcon_var',
                    'avg_defcon_qualifying', 'max_defcon_game', 'min_defcon_game']:
            df_active[col] = np.nan

        # Fetch fixture difficulty data
        print("Fetching fixture data...")
        fixtures_data = fetch_fixtures()
        teams_df = pd.DataFrame(bootstrap_data['teams'])
        # The GW we're planning for (GW1 pre-season, not GW2)
        next_gw_num = get_target_gw_num(bootstrap_data)
        # Fixture look-ahead starts AT the target GW, so anchor one behind it
        fixture_anchor_gw = next_gw_num - 1
        # Next 5 GWs only. A season-long average FDR converges to ~3 for every
        # team (useless for ranking), and a 30-fixture string swamps every table
        # that shows it. 5 GWs is the planning horizon that differentiates teams.
        fixture_difficulty = calculate_fixture_difficulty(
            fixtures_data, teams_df, fixture_anchor_gw, num_gameweeks=5)
        print(f"  Calculated 5-GW fixture difficulty for {len(fixture_difficulty)} teams")

        df_active['avg_fdr_5'] = df_active['team'].map(lambda x: fixture_difficulty.get(x, {}).get('avg_fdr'))
        df_active['fixture_string'] = df_active['team'].map(
            lambda x: fixture_difficulty.get(x, {}).get('fixture_string'))
        df_active['fixture_count'] = df_active['team'].map(lambda x: fixture_difficulty.get(x, {}).get('fixture_count'))

        # Attack/defence-specific FDR from team strength ratings (next 5 GWs)
        custom_fdr = calculate_custom_fdr(fixtures_data, teams_df, fixture_anchor_gw, num_gameweeks=5)
        for col, key in [('att_fdr_5', 'att_fdr'), ('def_fdr_5', 'def_fdr'),
                         ('next_att_fdr', 'next_att_fdr'), ('next_def_fdr', 'next_def_fdr')]:
            df_active[col] = df_active['team'].map(lambda x, k=key: custom_fdr.get(x, {}).get(k))
        print(f"  Attack/defence FDR computed for {len(custom_fdr)} teams")

        # Market-implied goal expectations (no-op without ODDS_API_KEY).
        # Only actually calls the API when ODDS_REFRESH_INTERVAL has
        # elapsed since the last successful pull — see the constant's
        # definition for why this can't run on every 3-hour Phase 1 cycle.
        odds_age = time.time() - DATA.get('odds_last_refresh', 0)
        if ODDS_API_KEY and (odds_age > ODDS_REFRESH_INTERVAL or not DATA.get('odds_lambdas')):
            odds_lambdas = fetch_market_lambdas(teams_df)
            with DATA_LOCK:
                DATA['odds_lambdas'] = odds_lambdas
                DATA['odds_last_refresh'] = time.time()
        else:
            odds_lambdas = DATA.get('odds_lambdas', {}) or {}
            if ODDS_API_KEY:
                print(f"  Odds cache is {odds_age / 3600:.1f}h old "
                      f"(refreshes every {ODDS_REFRESH_INTERVAL / 3600:.0f}h) — reusing")

        # Expected-goals ledger: team xGF/xGC per match, from keeper xGC.
        # Feeds every team-level form estimate below. Failure here is
        # non-fatal — the models fall back to actual goals.
        try:
            xg_ledger = build_team_xg_ledger(df_active)
        except Exception as e:
            print(f"  xG ledger unavailable ({e}) — falling back to actual goals")
            xg_ledger = {}
        with DATA_LOCK:
            DATA['xg_ledger'] = xg_ledger

        # Team clean-sheet probabilities — ONE model, shared by the xCS page
        # and the projection engine. Previously the engine used its own linear
        # FDR ramp and the two disagreed.
        xcs_next = calculate_expected_clean_sheets(
            fixtures_data, teams_df, fixture_anchor_gw, num_gws=1,
            odds_lambdas=odds_lambdas, xg_ledger=xg_ledger)
        xcs_5 = calculate_expected_clean_sheets(
            fixtures_data, teams_df, fixture_anchor_gw, num_gws=5,
            odds_lambdas=odds_lambdas, xg_ledger=xg_ledger)
        df_active['cs_prob_next'] = df_active['team'].map(
            lambda t: (xcs_next.get(t, {}).get('avg_cs_prob') or 0) / 100.0 or np.nan)
        df_active['cs_prob_5'] = df_active['team'].map(
            lambda t: (xcs_5.get(t, {}).get('avg_cs_prob') or 0) / 100.0 or np.nan)
        with DATA_LOCK:
            DATA['xcs_next'] = xcs_next
        print(f"  Clean-sheet probabilities computed for {len(xcs_next)} teams")

        # Fixture-specific goal environments (form + strengths + market)
        goal_env_next = calculate_goal_environment(
            fixtures_data, teams_df, fixture_anchor_gw, num_gws=1,
            odds_lambdas=odds_lambdas, xg_ledger=xg_ledger)
        goal_env_5 = calculate_goal_environment(
            fixtures_data, teams_df, fixture_anchor_gw, num_gws=5,
            odds_lambdas=odds_lambdas, xg_ledger=xg_ledger)
        df_active['att_env_next'] = df_active['team'].map(
            lambda t: goal_env_next.get(t, {}).get('att_env_next'))
        df_active['att_env_5'] = df_active['team'].map(
            lambda t: goal_env_5.get(t, {}).get('att_env_avg'))
        print(f"  Goal environments computed for {len(goal_env_next)} teams")

        total_managers = bootstrap_data['total_players']

        # Next fixture venue & FDR
        print(f"Computing next-fixture data for GW{next_gw_num}...")
        next_gw_fixtures = sorted(
            [f for f in fixtures_data if f.get('event') == next_gw_num],
            key=lambda f: f.get('kickoff_time') or ''
        )

        team_next_fixture = {}
        for f in next_gw_fixtures:
            home_name = teams_df[teams_df['id'] == f['team_h']]['name'].values[0] if f['team_h'] in teams_df[
                'id'].values else 'Unknown'
            away_name = teams_df[teams_df['id'] == f['team_a']]['name'].values[0] if f['team_a'] in teams_df[
                'id'].values else 'Unknown'
            if f['team_h'] not in team_next_fixture:
                team_next_fixture[f['team_h']] = {'opponent': away_name, 'venue': 'H', 'fdr': f.get('team_h_difficulty', 3)}
            if f['team_a'] not in team_next_fixture:
                team_next_fixture[f['team_a']] = {'opponent': home_name, 'venue': 'A', 'fdr': f.get('team_a_difficulty', 3)}

        df_active['next_opponent'] = df_active['team'].map(lambda x: team_next_fixture.get(x, {}).get('opponent', ''))
        df_active['next_venue'] = df_active['team'].map(lambda x: team_next_fixture.get(x, {}).get('venue', ''))
        df_active['next_fdr'] = df_active['team'].map(lambda x: team_next_fixture.get(x, {}).get('fdr', 3))

        # Initialise home/away + Phase-2 columns as NaN (Phase 2 will populate)
        for col in ['home_ppg', 'away_ppg', 'home_games', 'away_games', 'venue_ppg', 'ha_diff',
                    'start_rate', 'recent_minutes_pct', 'top_eo']:
            df_active[col] = np.nan

        # Captain score (partial — venue_ppg/start_rate neutral until Phase 2)
        print("Computing initial captain scores...")
        df_active['captain_score'] = compute_captain_scores(df_active)

        # 8-GW fixture counts (for the wildcard-horizon projection)
        _gws8 = set(range(fixture_anchor_gw + 1, fixture_anchor_gw + 9))
        _c8 = Counter()
        for _f in fixtures_data:
            if _f.get('event') in _gws8:
                _c8[_f['team_h']] += 1
                _c8[_f['team_a']] += 1
        df_active['fixture_count_8'] = df_active['team'].map(_c8).fillna(0)

        # Expected points projection (Phase-1 pass: season minutes share;
        # Phase 2 re-runs it with recent start data, priors and hit rates)
        print("Computing expected points projections...")
        gw_elapsed = current_gw['id'] if current_gw else 38
        with DATA_LOCK:
            _priors = DATA.get('last_season_priors', {})
        _xp_parts = {}
        (df_active['exp_mins_next'],
         df_active['proj_pts_next'],
         df_active['proj_pts_5'],
         df_active['proj_pts_8'],
         df_active['haul_pct'],
         df_active['xgi_lam_neutral']) = compute_expected_points(
            df_active, gw_elapsed, priors=_priors, components_out=_xp_parts)

        # Attach the projection decomposition for the Model Lab breakdown.
        # Only the next-GW pass is collected; the 5/8-GW horizons reuse the
        # same component model with averaged fixtures.
        for _k, _v in (_xp_parts or {}).items():
            df_active[_k] = _v

        # Captain distribution + EO-adjusted gain, both built off those parts
        try:
            _cd = compute_captain_distribution(df_active)
            for _c in _cd.columns:
                df_active[_c] = _cd[_c]
            df_active['captain_gain'] = compute_captain_gain(df_active)
        except Exception as _e:
            print(f"  captain distribution unavailable ({_e})")

        # Neutral per-GW base (flat FDR-3, single fixture) — reused by the
        # chip planner and the squad builder's chip-target emphasis
        _neutral = df_active.copy()
        _neutral['next_att_fdr'] = 3.0
        _neutral['next_def_fdr'] = 3.0
        df_active['proj_neutral_gw'] = compute_expected_points(
            _neutral, gw_elapsed, priors=_priors)[1]
        del _neutral

        # Transfer trend / price prediction
        print("Computing price change likelihood scores...")
        df_active['price_change_likelihood'] = df_active.apply(
            lambda r: estimate_price_change_likelihood(r, total_managers), axis=1
        )
        # FPL's official price-change tracker replaces the estimate when present
        _tracker = read_fpl_price_tracker(bootstrap_data.get('elements', []))
        if _tracker is not None:
            df_active = apply_price_tracker(df_active, _tracker)
            DATA['price_tracker_at'] = time.time()
            DATA['price_tracker_next'] = time.time() + PRICE_TRACKER_INTERVAL
            print(f"  FPL price tracker: {int(_tracker['price_status'].str.contains('rise').sum())} "
                  f"likely risers, {int(_tracker['price_status'].str.contains('drop').sum())} likely fallers")
            # One raw sample in the logs, so the undocumented field layout can be checked
            _sample = next((e for e in bootstrap_data.get('elements', [])
                            if e.get('price_change_projections')), None)
            if _sample:
                print(f"  Price tracker sample ({_sample.get('web_name')}): "
                      f"percent={_sample.get('price_change_percent')!r} "
                      f"projections={str(_sample.get('price_change_projections'))[:200]}")
        else:
            print("  FPL price tracker fields not in bootstrap — using the transfer-based estimate")

        # Differential score
        df_active['differential_score'] = (
                (df_active['ppg'].fillna(0) * 0.5) +
                (df_active['form'].fillna(0) * 0.3) +
                ((100 - df_active['ownership'].fillna(50)) / 10 * 0.2)
        ).round(2)

        # --- Daily snapshot + 7-day trend deltas ---
        # Turns "what is true now" into "what is changing": ownership momentum
        # and realised price movement over the last week.
        try:
            save_daily_snapshot(df_active, next_gw_num)
            # Calibration bookkeeping: log what we predict, record what happened
            log_projections(df_active, next_gw_num)
            log_model_features(df_active, next_gw_num, _priors)
            if current_gw:
                log_actual_points(df_active, current_gw['id'])
            cal = compute_calibration()
            if cal:
                DATA_CAL_MSG = (f"  Calibration: model MAE {cal['mae_model']} vs "
                                f"FPL {cal['mae_fpl']} over {cal['gws']} GW(s)")
                print(DATA_CAL_MSG)
            with DATA_LOCK:
                DATA['calibration'] = cal
            baseline, baseline_date = load_snapshot_baseline(days=7)
            if baseline:
                df_active['own_delta_7d'] = (
                    df_active['ownership'] -
                    df_active['id'].map(lambda x: baseline.get(x, {}).get('ownership'))
                ).round(2)
                df_active['price_delta_7d'] = (
                    df_active['price'] -
                    df_active['id'].map(lambda x: baseline.get(x, {}).get('price'))
                ).round(1)
                DATA_DELTA_BASIS = f"7 days (snapshot {baseline_date})"
                print(f"  Trend deltas computed vs snapshot from {baseline_date}")
            else:
                df_active['own_delta_7d'] = np.nan
                df_active['price_delta_7d'] = np.nan
                DATA_DELTA_BASIS = None
                print("  No prior snapshot yet — falling back to gameweek deltas")
        except Exception as e:
            print(f"  Snapshot store unavailable: {e}")
            df_active['own_delta_7d'] = np.nan
            df_active['price_delta_7d'] = np.nan
            DATA_DELTA_BASIS = None

        # --- Storage-free gameweek deltas -------------------------------
        # The 7-day figures need a snapshot that survived to yesterday, which
        # an ephemeral disk never provides. These two need no history at all
        # because bootstrap-static already carries the movement:
        #
        #   price   cost_change_event is the realised change this gameweek.
        #   owners  net transfers / total managers x 100 IS the ownership
        #           change in percentage points — ownership is just a count
        #           of squads over the same denominator, so this is exact,
        #           not an approximation.
        try:
            _tm = max(int(total_managers or 0), 1)
            df_active['own_delta_gw'] = (
                df_active['net_transfers_gw'].fillna(0) / _tm * 100).round(3)
            df_active['price_delta_gw'] = pd.to_numeric(
                df_active.get('cost_change_event'), errors='coerce').round(1)
        except Exception as e:
            print(f"  GW deltas unavailable: {e}")
            df_active['own_delta_gw'] = np.nan
            df_active['price_delta_gw'] = np.nan

        # Fall back so the displayed columns are never empty
        if df_active['own_delta_7d'].isna().all():
            df_active['own_delta_7d'] = df_active['own_delta_gw']
        if df_active['price_delta_7d'].isna().all():
            df_active['price_delta_7d'] = df_active['price_delta_gw']
        if DATA_DELTA_BASIS is None:
            DATA_DELTA_BASIS = "this gameweek (no prior snapshot)"
        with DATA_LOCK:
            DATA['delta_basis'] = DATA_DELTA_BASIS

        sorted_teams = sorted(df['team_name'].unique())

        with DATA_LOCK:
            DATA['bootstrap_data'] = bootstrap_data
            DATA['df'] = df
            DATA['df_active'] = df_active
            DATA['current_gw'] = current_gw
            DATA['next_gw'] = next_gw
            DATA['total_managers'] = total_managers
            DATA['fixtures_data'] = fixtures_data
            DATA['teams_df'] = teams_df
            DATA['fixture_difficulty'] = fixture_difficulty
            DATA['player_histories'] = {}
            DATA['sorted_teams'] = sorted_teams
            DATA['next_gw_num'] = next_gw_num
            DATA['fixture_anchor_gw'] = fixture_anchor_gw
            DATA['season_started'] = started
            DATA['season_label'] = SEASON['label']
            DATA['last_refresh'] = time.time()
            DATA['refreshing'] = False
            DATA['heavy_loaded'] = False

        print(f"  Phase 1 complete at {datetime.now().strftime('%H:%M:%S')} — server ready")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"ERROR during core data refresh: {e}")
        with DATA_LOCK:
            DATA['refreshing'] = False


def refresh_heavy_data():
    """
    Phase 2: Fetch individual player histories (~300 API calls).
    Populates consistency data, home/away splits, and recalculates captain scores.
    Runs in a background thread so the server stays responsive.
    """
    with DATA_LOCK:
        if DATA.get('refreshing'):
            return
        DATA['refreshing'] = True

    try:
        print(f"\n{'=' * 60}")
        print(f"  Phase 2: Player history refresh ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'=' * 60}")

        # Work with a copy of df_active
        with DATA_LOCK:
            df_active = DATA['df_active'].copy()
            fixtures_data = DATA['fixtures_data']
            total_managers = DATA['total_managers']
            started = DATA.get('season_started', True)

        # Pre-season there is no match history to fetch — every element-summary
        # call returns an empty history. Skip ~300 requests (and the memory
        # spike that comes with them) rather than burning them for nothing.
        if not started:
            print("  Season has not started — skipping player history fetch.")
            with DATA_LOCK:
                DATA['last_refresh'] = time.time()
                DATA['refreshing'] = False
                DATA['heavy_loaded'] = True
            save_cache()
            print(f"{'=' * 60}\n")
            return

        # Bonus consistency data
        print("Fetching player match history for bonus consistency analysis...")
        with DATA_LOCK:
            _cur = DATA.get('current_gw')
        _gwn = _cur['id'] if _cur else 0
        _mins_lo = adaptive_min_minutes(200, _gwn)
        _mins_hi = adaptive_min_minutes(450, _gwn)
        defcon_positions = SEASON['defcon_positions']
        consistency_players = df_active[
            (df_active['minutes'] >= _mins_lo) &
            (df_active['position'].isin(defcon_positions))
            ]['id'].tolist()

        print(f"  Fetching data for {len(consistency_players)} players...")
        consistency_thresholds = dict(zip(
            df_active.loc[df_active['id'].isin(consistency_players), 'id'],
            df_active.loc[df_active['id'].isin(consistency_players), 'position'].map(SEASON['thresholds'])
        ))
        consistency_data, season_priors = calculate_bonus_consistency(
            consistency_players, consistency_thresholds)
        print(f"  Retrieved data for {len(consistency_data)} players "
              f"(+ last-season priors for {len(season_priors)})")
        with DATA_LOCK:
            DATA['last_season_priors'] = season_priors

        df_active['qualifying_games'] = df_active['id'].map(
            lambda x: consistency_data.get(x, {}).get('qualifying_games'))
        df_active['bonus_games'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('bonus_games'))
        df_active['hit_rate'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('hit_rate'))
        for _c in ('starts_60', 'bonus_games_60', 'defcon_per_90_games', 'defcon_var'):
            df_active[_c] = df_active['id'].map(
                lambda x, _c=_c: consistency_data.get(x, {}).get(_c))
        df_active['avg_defcon_qualifying'] = df_active['id'].map(
            lambda x: consistency_data.get(x, {}).get('avg_defcon'))
        df_active['max_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('max_defcon'))
        df_active['min_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('min_defcon'))

        # Commit consistency columns immediately — if any LATER Phase-2 step
        # fails (home/away, EO, projections), these results must survive
        # rather than being discarded with the local copy.
        with DATA_LOCK:
            DATA['df_active'] = df_active.copy()

        # Free consistency data to reclaim memory before next batch
        del consistency_data, consistency_thresholds, consistency_players
        gc.collect()

        # Home/Away splits
        print("Fetching player histories for captain & home/away analysis...")
        captain_candidates = df_active[
            (df_active['minutes'] >= _mins_hi) &
            (df_active['position'].isin(SEASON['outfield_positions']))
            ].nlargest(100, 'form')['id'].tolist()

        print(f"  Fetching match history for {len(captain_candidates)} captain candidates...")
        player_histories = fetch_player_history_batch(captain_candidates)

        # Schedule-adjusted 'true form' from the same histories
        with DATA_LOCK:
            _tdf = DATA.get('teams_df', pd.DataFrame())
        sched_form = calculate_schedule_adjusted_form(player_histories, _tdf)
        df_active['adj_xgi90'] = df_active['id'].map(
            lambda x: sched_form.get(x, {}).get('adj_xgi90'))
        df_active['raw_recent_xgi90'] = df_active['id'].map(
            lambda x: sched_form.get(x, {}).get('raw_xgi90'))
        df_active['sched_factor'] = df_active['id'].map(
            lambda x: sched_form.get(x, {}).get('sched_factor'))
        print(f"  Schedule-adjusted form computed for {len(sched_form)} players")
        print(f"  Retrieved history for {len(player_histories)} players")

        home_away_splits = calculate_home_away_splits(player_histories)

        df_active['home_ppg'] = df_active['id'].map(lambda x: home_away_splits.get(x, {}).get('home_ppg'))
        df_active['away_ppg'] = df_active['id'].map(lambda x: home_away_splits.get(x, {}).get('away_ppg'))
        df_active['home_games'] = df_active['id'].map(lambda x: home_away_splits.get(x, {}).get('home_games'))
        df_active['away_games'] = df_active['id'].map(lambda x: home_away_splits.get(x, {}).get('away_games'))
        df_active['venue_ppg'] = df_active.apply(
            lambda r: r['home_ppg'] if r['next_venue'] == 'H' else r['away_ppg'], axis=1
        )
        df_active['ha_diff'] = df_active['home_ppg'] - df_active['away_ppg']

        # Minutes security from the same histories — no extra API calls
        print("Computing minutes security (last 6 matches)...")
        minutes_sec = calculate_minutes_security(player_histories)
        df_active['start_rate'] = df_active['id'].map(
            lambda x: minutes_sec.get(x, {}).get('start_rate'))
        df_active['recent_minutes_pct'] = df_active['id'].map(
            lambda x: minutes_sec.get(x, {}).get('recent_minutes_pct'))

        # Incremental commit: home/away + minutes security now safe too
        with DATA_LOCK:
            DATA['df_active'] = df_active.copy()

        # Free intermediate data to reclaim memory
        del home_away_splits, captain_candidates, minutes_sec
        gc.collect()

        # Effective ownership among the overall-league leaders. Raw ownership
        # includes millions of dead teams; EO among top managers is what
        # actually separates a differential from a must-own.
        print("Sampling effective ownership from overall league leaders...")
        try:
            with DATA_LOCK:
                cur = DATA.get('current_gw')
            eo_gw = cur['id'] if cur else None
            if eo_gw:
                entry_ids = fetch_top_manager_entry_ids()
                eo_map, n_ok = calculate_effective_ownership(entry_ids, eo_gw)
                if n_ok >= 20:
                    df_active['top_eo'] = df_active['id'].map(eo_map).fillna(0).round(1)
                    print(f"  EO computed from {n_ok} top squads")
                else:
                    print(f"  EO sample too small ({n_ok} squads) — skipped")
                del eo_map, entry_ids
            else:
                print("  No current GW yet — EO skipped until GW1 starts")
        except Exception as e:
            print(f"  EO sampling failed (non-fatal): {e}")
        gc.collect()

        # Recalculate captain score with venue PPG + start security in play
        print("Recalculating captain scores with home/away + minutes data...")
        df_active['captain_score'] = compute_captain_scores(df_active)

        # Re-run projections now start_rate, recent minutes and measured
        # DEFCON hit rates are in the frame (Phase 1 used fallbacks)
        print("Recalculating expected points with Phase-2 data...")
        with DATA_LOCK:
            cur_for_proj = DATA.get('current_gw')
            _priors = DATA.get('last_season_priors', {})
        gw_elapsed = cur_for_proj['id'] if cur_for_proj else 38
        _xp_parts = {}
        (df_active['exp_mins_next'],
         df_active['proj_pts_next'],
         df_active['proj_pts_5'],
         df_active['proj_pts_8'],
         df_active['haul_pct'],
         df_active['xgi_lam_neutral']) = compute_expected_points(
            df_active, gw_elapsed, priors=_priors, components_out=_xp_parts)

        # Attach the projection decomposition for the Model Lab breakdown.
        # Only the next-GW pass is collected; the 5/8-GW horizons reuse the
        # same component model with averaged fixtures.
        for _k, _v in (_xp_parts or {}).items():
            df_active[_k] = _v

        # Captain distribution + EO-adjusted gain, both built off those parts
        try:
            _cd = compute_captain_distribution(df_active)
            for _c in _cd.columns:
                df_active[_c] = _cd[_c]
            df_active['captain_gain'] = compute_captain_gain(df_active)
        except Exception as _e:
            print(f"  captain distribution unavailable ({_e})")
        _neutral = df_active.copy()
        _neutral['next_att_fdr'] = 3.0
        _neutral['next_def_fdr'] = 3.0
        df_active['proj_neutral_gw'] = compute_expected_points(
            _neutral, gw_elapsed, priors=_priors)[1]
        del _neutral

        # Swap into global store
        with DATA_LOCK:
            DATA['df_active'] = df_active
            DATA['player_histories'] = player_histories
            DATA['last_refresh'] = time.time()
            DATA['refreshing'] = False
            DATA['heavy_loaded'] = True

        gc.collect()  # Free old df_active that was just replaced

        save_cache()

        print(f"  Phase 2 complete at {datetime.now().strftime('%H:%M:%S')}")
        print(
            f"  Next refresh after: {datetime.fromtimestamp(DATA['last_refresh'] + REFRESH_INTERVAL).strftime('%H:%M:%S')}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"ERROR during heavy data refresh: {e}")
        with DATA_LOCK:
            DATA['refreshing'] = False


def refresh_all_data():
    """Full refresh: Phase 1 then Phase 2 sequentially (used by background timer)."""
    refresh_core_data()
    gc.collect()  # Free Phase 1 temporaries before heavy Phase 2
    refresh_heavy_data()


def check_and_refresh():
    """Check if data is stale and refresh in background if needed."""
    check_shots_sync()   # independent cadence; no-op unless due
    if (DATA.get('df_active') is not None and not DATA.get('price_tracker_running')
            and not DATA.get('refreshing') and time.time() >= DATA.get('price_tracker_next', 0)):
        with DATA_LOCK:
            DATA['price_tracker_next'] = time.time() + PRICE_TRACKER_INTERVAL   # debounce
        threading.Thread(target=refresh_price_tracker, daemon=True).start()
    age = time.time() - DATA.get('last_refresh', 0)
    if age > REFRESH_INTERVAL and not DATA.get('refreshing', False):
        print(f"Data is {age / 3600:.1f}h old. Triggering background refresh...")
        thread = threading.Thread(target=refresh_all_data, daemon=True)
        thread.start()


def get_data():
    """Get current data, triggering refresh if stale. Returns DATA dict."""
    check_and_refresh()
    return DATA


# --- Initial data load ---
# Try cache first for instant startup. If no cache, load EVERYTHING
# synchronously so every tab has data before the server accepts requests.
# Background thread only handles the periodic 3-hour refresh cycle.
cache_hit = load_cache()
if not cache_hit:
    print("  No cache — running full synchronous data load...")
    refresh_all_data()  # Phase 1 + Phase 2, blocks until complete

# Convenience references for layout building (used once at startup)
df = DATA['df']
df_active = DATA['df_active']
current_gw = DATA['current_gw']
next_gw = DATA['next_gw']
total_managers = DATA['total_managers']
sorted_teams = DATA['sorted_teams']
next_gw_num = DATA['next_gw_num']
player_histories = DATA['player_histories']

# Chip name mapping (used by home tab callback)
_GW_BOOT = DATA['current_gw']['id'] if DATA.get('current_gw') else 0
DEF_MINS_HI = adaptive_min_minutes(450, _GW_BOOT)
DEF_MINS_LO = adaptive_min_minutes(200, _GW_BOOT)
DEF_MIN_GAMES = max(1, min(5, _GW_BOOT)) if _GW_BOOT else 5

chip_name_map = {
    'bboost': 'Bench Boost',
    '3xc': 'Triple Captain',
    'wildcard': 'Wildcard',
    'freehit': 'Free Hit',
}

# =============================================================================
# DASH APPLICATION
# =============================================================================

app = Dash(__name__, meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0"}
])
server = app.server

# Responsive CSS for mobile devices
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>FPL Room</title>
        {%favicon%}
        <!-- Home-screen app: icon, name and full-screen mode (files in /assets) -->
        <link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">
        <link rel="manifest" href="/assets/manifest.webmanifest">
        <meta name="apple-mobile-web-app-title" content="FPL Room">
        <meta name="application-name" content="FPL Room">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black">
        <meta name="theme-color" content="#37003c">
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Outfit', Arial, sans-serif; color: #37003c; }
            button, input, select, textarea { font-family: inherit; }
            /* ================================================================
               BASE — all screen sizes
            ================================================================ */

            * { box-sizing: border-box; }

            /* Horizontal scroll for tables */
            .dash-spreadsheet-container {
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch;
            }

            /* ================================================================
               SIDEBAR LAYOUT
            ================================================================ */

            #app-body {
                display: flex;
                height: calc(100vh - 68px);   /* header + 4px FPL stripe */
                position: relative;
                overflow: hidden;
            }

            /* --- Sidebar --- */
            #sidebar {
                width: 240px;
                min-width: 240px;
                background: #37003c;
                border-right: 0;
                height: 100%;
                overflow-y: auto;
                overflow-x: hidden;
                z-index: 500;
                transition: transform 0.22s ease-out;
                will-change: transform;
                flex-shrink: 0;
                padding-bottom: 24px;
            }

            /* Section label: This Week, My Team, Leagues, Planner, Research */
            .nav-group-label {
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #00ff87;
                padding: 20px 22px 6px 22px;
                margin: 0;
            }
            /* Sub-group inside Research */
            .nav-subgroup-label {
                font-size: 12px;
                font-weight: 600;
                color: rgba(255,255,255,0.55);
                padding: 10px 22px 2px 22px;
                margin: 0;
            }

            /* Nav item button */
            .nav-item {
                display: flex;
                align-items: center;
                gap: 10px;
                width: calc(100% - 20px);
                margin: 1px 10px;
                min-height: 38px;
                background: none;
                border: none;
                border-radius: 8px;
                padding: 8px 12px 8px 14px;
                font-size: 14px;
                font-family: inherit;
                font-weight: 500;
                color: rgba(255,255,255,0.82);
                cursor: pointer;
                text-align: left;
                transition: background 0.15s, color 0.15s;
                line-height: 1.3;
                position: relative;
            }

            .nav-item:hover {
                background: rgba(255,255,255,0.08);
                color: #ffffff;
            }

            .nav-item.active {
                background: rgba(255,255,255,0.12);
                color: #ffffff;
                font-weight: 700;
                box-shadow: inset 3px 0 0 #00ff87;
            }

            /* FPL's green-to-cyan stripe under the header */
            .fpl-stripe { height: 4px; background: linear-gradient(90deg, #00ff87, #04f5ff); }

            /* --- Content area --- */
            #content-area {
                flex: 1;
                min-width: 0;
                height: 100%;
                overflow-y: auto;
                padding: 24px 20px;
                background: #f5f5f5;
            }

            /* --- Mobile overlay: covers app-body, not the header --- */
            #sidebar-overlay {
                display: none;
                position: absolute;
                inset: 0;
                background: rgba(0,0,0,0.45);
                z-index: 499;
            }

            /* --- Hamburger button (hidden on desktop) --- */
            /* Refresh button: only when opened from the home screen, where
               iOS gives you no reload button and no pull-to-refresh. */
            #app-refresh-btn {
                display: none;
                background: none;
                border: none;
                color: #ffffff;
                font-size: 24px;
                line-height: 1;
                min-width: 44px;
                min-height: 44px;
                cursor: pointer;
                -webkit-tap-highlight-color: transparent;
            }
            @media (display-mode: standalone) {
                #app-refresh-btn { display: inline-flex; align-items: center; justify-content: center; }
            }
            html.ios-standalone #app-refresh-btn {
                display: inline-flex; align-items: center; justify-content: center;
            }

            #hamburger-btn {
                display: none;
                background: none;
                border: none;
                color: white;
                font-size: 22px;
                cursor: pointer;
                /* 44x44 is the minimum comfortable touch target; the old
                   ~26px button was easy to half-miss on a phone. */
                min-width: 44px;
                min-height: 44px;
                padding: 0 10px 0 0;
                line-height: 1;
                -webkit-tap-highlight-color: transparent;
            }

            /* Kill the legacy ~300ms double-tap-zoom wait on taps */
            button, .nav-item, #sidebar-overlay {
                touch-action: manipulation;
            }

            /* ================================================================
               TABLET & BELOW  (≤ 900px)
            ================================================================ */
            /* Horizontal scroll for fixture grids, all screen sizes */
            .chart-scroll {
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch;
            }

            /* ================================================================
               HEADER — compact single row on phones and narrow tablets
            ================================================================ */
            .hdr-short { display: none; }
            /* min-width lives here, not inline: an inline min-width on the header's
               left group matched the mobile filter-row rule, which stretched it to
               100% width and pushed the GW / Updated text off the right edge. */
            .hdr .hdr-row > .hdr-left { min-width: 0; flex: 1 1 auto; }
            .hdr .hdr-row > .hdr-status { flex: 0 0 auto; }
            @media (max-width: 700px) {
                .hdr { padding: 8px 0 !important; }
                .hdr-row { padding: 0 8px 0 4px !important; }
                .hdr-logo { height: 28px !important; margin-right: 8px !important; }
                .hdr-pill { font-size: 14px !important; padding: 4px 8px !important;
                            margin-right: 0 !important; }
                .hdr-long, .hdr-sub, .hdr-sep { display: none !important; }
                .hdr-short { display: inline; }
                /* GW and "Updated" stack as two small right-aligned lines */
                .hdr-texts { flex-direction: column !important; align-items: flex-end !important;
                             gap: 1px !important; text-align: right; line-height: 1.25; }
                .hdr-gw { font-size: 12px !important; white-space: nowrap; }
                .hdr-updated { font-size: 11px !important; max-width: 150px; }
                #hamburger-btn { min-width: 40px; padding: 0 !important; }
                #app-body { height: calc(100vh - 64px); }   /* 60px compact header + 4px stripe */
            }
            @media (max-width: 380px) {
                .hdr-pill { font-size: 13px !important; padding: 3px 6px !important; }
                .hdr-logo { height: 24px !important; margin-right: 6px !important; }
                .hdr-gw { font-size: 11px !important; }
                .hdr-updated { font-size: 10px !important; max-width: 128px; }
            }

            @media (max-width: 900px) {

                #hamburger-btn { display: block; }

                #sidebar {
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 100%;
                    overflow-y: auto;
                    transform: translateX(-100%);
                    z-index: 600;
                    box-shadow: 4px 0 16px rgba(0,0,0,0.18);
                }

                #sidebar.sidebar-open {
                    transform: translateX(0);
                }

                #sidebar-overlay.overlay-open {
                    display: block;
                }

                #content-area {
                    padding: 12px 8px;
                }

                .js-plotly-plot,
                .js-plotly-plot .plotly,
                .js-plotly-plot .plotly .main-svg {
                    width: 100% !important;
                }

                /* Grids are the exception: squeezing 6+ gameweek columns into
                   a 380px screen makes every cell unreadable. These keep a
                   fixed cell width and scroll sideways instead. */
                .chart-scroll .js-plotly-plot,
                .chart-scroll .js-plotly-plot .plotly,
                .chart-scroll .js-plotly-plot .plotly .main-svg {
                    width: auto !important;
                }

                .dash-dropdown { min-width: 100% !important; }

                /* React serialises style dicts to hyphenated CSS, so the DOM
                   attribute reads "min-width: 160px" — a [style*="minWidth"]
                   selector never matched and none of these filter rows have
                   been stacking on mobile. */
                [style*="display: flex"] > div[style*="min-width"] {
                    min-width: 100% !important;
                    flex: 1 1 100% !important;
                    padding-left: 0 !important;
                    padding-right: 0 !important;
                    margin-bottom: 10px;
                }
                /* The rule above is meant for filter columns, but it also caught the
                   Home spotlight cards and stripped their side padding, leaving the
                   text and photo pressed against the card edges on phones. */
                #content-area .spotlight-card {
                    padding-left: 14px !important;
                    padding-right: 14px !important;
                }

                [style*="margin: 0 -10px"] > div {
                    flex: 1 1 45% !important;
                    min-width: 45% !important;
                }

                [style*="gap: 20px"] > div {
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                }

                [style*="padding: 24px"] { padding: 16px !important; }

                h2 { font-size: 22px !important; }
                h3 { font-size: 18px !important; }
                h4 { font-size: 16px !important; }

                .dash-cell,
                .dash-spreadsheet-container td {
                    padding: 8px 6px !important;
                    font-size: 12px !important;
                }
                .dash-header,
                .dash-spreadsheet-container th {
                    padding: 8px 6px !important;
                    font-size: 11px !important;
                }

                .rc-slider {
                    margin-left: 8px !important;
                    margin-right: 8px !important;
                }
            }

            /* ================================================================
               PHONE  (≤ 480px)
            ================================================================ */
            @media (max-width: 480px) {
                h2 { font-size: 18px !important; }
                h3 { font-size: 16px !important; }
                h4 { font-size: 14px !important; }

                [style*="margin: 0 -10px"] > div {
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                }

                .dash-cell,
                .dash-spreadsheet-container td {
                    padding: 6px 4px !important;
                    font-size: 11px !important;
                }
                .dash-header,
                .dash-spreadsheet-container th {
                    padding: 6px 4px !important;
                    font-size: 10px !important;
                }

                .js-plotly-plot { max-height: 300px !important; }
                .chart-scroll .js-plotly-plot { max-height: none !important; }
            }
            /* Player headshots — keep the box stable while fallbacks resolve */
            .player-photo {
                object-fit: cover;
                background: #efeaf3;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
                // Older iOS versions only expose navigator.standalone, not the media query
                if (window.navigator.standalone) {
                    document.documentElement.classList.add('ios-standalone');
                }
            </script>
            <script>
                // Player headshots: advance through data-fallbacks on 404.
                // 'error' does not bubble, so we listen in the capture phase.
                document.addEventListener('error', function (e) {
                    var img = e.target;
                    if (!img || img.tagName !== 'IMG') { return; }
                    if (!img.classList || !img.classList.contains('player-photo')) { return; }
                    var raw = img.getAttribute('data-fallbacks') || '';
                    var list = raw.split('||').filter(function (u) { return u.length > 0; });
                    if (list.length === 0) { return; }
                    img.setAttribute('data-fallbacks', list.slice(1).join('||'));
                    img.src = list[0];
                }, true);
            </script>
        </footer>
    </body>
</html>
'''


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


def build_stat_card(title, value, subtitle=None, color=COLORS['primary'], image_code=None,
                    link_page=None, link_label='See all'):
    return html.Div([
        player_photo_img(image_code,
                         style={'width': '50px', 'height': '60px',
                                'borderRadius': '6px', 'marginBottom': '8px'})
        if image_code is not None else None,
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
            'fontSize': '14px',
            'margin': '0'
        }) if subtitle else None,
        html.Button(
            f"{link_label} \u2192",
            id={'type': 'home-jump', 'page': link_page},
            n_clicks=0,
            style={'marginTop': '10px', 'padding': '0', 'border': 'none',
                   'background': 'none', 'color': COLORS['accent'],
                   'fontWeight': '600', 'fontSize': '13px', 'cursor': 'pointer',
                   'fontFamily': 'inherit'}
        ) if link_page else None
    ], style=STAT_CARD_STYLE)


def build_player_spotlight(player, title, metric_label, metric_value,
                           link_page=None, link_label='See all'):
    """
    Spotlight card. `link_page` turns the footer into a jump to the page that
    shows the full ranking behind the single name — the card answers "who",
    the link answers "who else".
    """
    if player is None:
        return html.Div()

    text_section = html.Div([
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
        html.P(f"{player['team_name']} {player['position']}  £{player['price']:.1f}m", style={
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
        ]),
        html.Button(
            f"{link_label} \u2192",
            id={'type': 'home-jump', 'page': link_page},
            n_clicks=0,
            style={'marginTop': '12px', 'padding': '0', 'border': 'none',
                   'background': 'none', 'color': COLORS['accent'],
                   'fontWeight': '600', 'fontSize': '13px', 'cursor': 'pointer',
                   'fontFamily': 'inherit'}
        ) if link_page else None
    ], style={'flex': '1'})

    image_section = player_photo_img(player, style={
        'width': '70px', 'height': '90px',
        'borderRadius': '8px', 'alignSelf': 'center'
    })

    return html.Div([text_section, image_section], className='spotlight-card', style={
        **CARD_STYLE, 'flex': '1', 'minWidth': '220px',
        'display': 'flex', 'justifyContent': 'space-between'
    })


# =============================================================================
# SQUAD BUILDER HELPER
# =============================================================================

def build_optimal_squad(df, budget, objective='ppg', must_include=None,
                        must_exclude=None, min_minutes=0):
    """
    Solve a binary integer programme to find the highest-scoring 15-player
    FPL squad subject to:
      - 2 GKP, 5 DEF, 5 MID, 3 FWD
      - Total cost <= budget
      - Max 3 players per club
      - Must-include / must-exclude player lists
    Returns a DataFrame of the selected 15, or None if no solution found.
    """
    eligible = df.copy()
    eligible = eligible.dropna(subset=['price', 'position', 'team_name'])

    must_include = [int(x) for x in (must_include or [])]
    must_exclude = [int(x) for x in (must_exclude or [])]

    # Pinned players bypass the minutes filter
    pinned = eligible[eligible['id'].isin(must_include)].copy()
    eligible = eligible[eligible['minutes'] >= (min_minutes or 0)].copy()
    eligible = pd.concat([eligible, pinned]).drop_duplicates(subset=['id'])

    if must_exclude:
        eligible = eligible[~eligible['id'].isin(must_exclude)]

    # Blended score
    eligible['blended'] = (
            eligible['ppg'].fillna(0) * 0.40 +
            eligible['form'].fillna(0) * 0.35 +
            eligible['expected_goal_involvements'].fillna(0) * 2 * 0.25
    ).round(3)

    obj_col = objective if objective in eligible.columns else 'ppg'
    eligible[obj_col] = eligible[obj_col].fillna(0)
    eligible = eligible.reset_index(drop=True)

    prob = pulp.LpProblem("FPL_Squad_Builder", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat='Binary') for i in eligible.index}

    # Objective
    prob += pulp.lpSum(x[i] * eligible.loc[i, obj_col] for i in eligible.index)

    # Budget
    prob += pulp.lpSum(x[i] * eligible.loc[i, 'price'] for i in eligible.index) <= budget

    # Positional quotas
    for pos, quota in [('GKP', 2), ('DEF', 5), ('MID', 5), ('FWD', 3)]:
        idx = eligible[eligible['position'] == pos].index
        prob += pulp.lpSum(x[i] for i in idx) == quota

    # Max 3 per club
    for team in eligible['team_name'].unique():
        idx = eligible[eligible['team_name'] == team].index
        prob += pulp.lpSum(x[i] for i in idx) <= 3

    # Must-include
    for pid in must_include:
        idx = eligible[eligible['id'] == pid].index
        if len(idx) > 0:
            prob += x[idx[0]] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None

    selected_idx = [i for i in eligible.index if (x[i].value() or 0) > 0.5]
    return eligible.loc[selected_idx].copy()


# Home page table columns (used by callback)
home_value_cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'total_points', 'points_per_million',
                   'form', 'ownership']

# =============================================================================
# LAYOUT
# =============================================================================

# Threshold shorthands for layout copy (SEASON is populated by this point)
DEF_THR = SEASON['thresholds'].get('DEF', 10)
MID_THR = SEASON['thresholds'].get('MID', 12)

# Pages whose content is rendered on demand: only when first opened, and
# again only if the underlying data has refreshed since. Everything else on
# the page (dropdowns, filters) still re-renders live as you change it.
LAZY_PAGES = [
    'home', 'defcon-bonus', 'bonus-consistency', 'defcon', 'xg', 'underlying',
    'value', 'form', 'cs', 'fixtures', 'fixture-ticker', 'fixture-outlook',
    'differentials', 'captain', 'transfers', 'model-lab', 'transfer-planner',
    'squad-builder', 'xcs', 'shot-profiles', 'matchups', 'chance-quality',
]


def _need_visit(visit):
    """Page callbacks bail until their page has actually been opened."""
    if not visit:
        raise PreventUpdate


app.layout = html.Div([
    # Interval + stores
    dcc.Interval(id='refresh-interval', interval=2 * 60 * 1000, n_intervals=0),
    dcc.Store(id='active-page', data='home'),
    dcc.Store(id='active-page-local'),
    dcc.Store(id='sidebar-open', data=False),
    # Lazy page rendering — see LAZY_PAGES / gate_page_renders below.
    dcc.Store(id='data-version'),
    dcc.Store(id='page-rendered', data={}),
    *[dcc.Store(id=f'visit-{p}') for p in LAZY_PAGES],
    dcc.Store(id='my-squad-store', data=None),

    # Header
    html.Div([
        html.Div([
            html.Div([
                # Hamburger (visible on mobile only via CSS)
                html.Button('☰', id='hamburger-btn', n_clicks=0),
                html.Img(src="/assets/premier_league_logo.png", className='hdr-logo',
                         style={'height': '40px', 'marginRight': '12px'}),
                html.Span([
                    html.Span(f"Fantasy Premier League {SEASON['label']}", className='hdr-long'),
                    html.Span(f"FPL {SEASON['label']}", className='hdr-short'),
                ], className='hdr-pill',
                   style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                          'padding': '6px 12px', 'borderRadius': '6px', 'fontWeight': '800',
                          'fontSize': '18px', 'marginRight': '12px', 'whiteSpace': 'nowrap'}),
                html.Span("Analytics Hub", className='hdr-sub',
                          style={'color': 'white', 'fontSize': '20px', 'fontWeight': '600'})
            ], className='hdr-left', style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Div([
                    html.Span(id='gw-status-text', className='hdr-gw',
                              style={'color': 'rgba(255,255,255,0.8)', 'fontSize': '13px'}),
                    html.Span(" | ", className='hdr-sep',
                              style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '13px'}),
                    html.Span(id='last-updated-text', className='hdr-updated',
                              style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '12px'}),
                ], className='hdr-texts', style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}),
                # Home-screen app only (no Safari reload button there); CSS shows it
                html.Button('\u21bb', id='app-refresh-btn', n_clicks=0,
                            title='Refresh', **{'aria-label': 'Refresh'}),
            ], className='hdr-status', style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'})
        ], className='hdr-row', style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                  'maxWidth': '100%', 'margin': '0 auto', 'padding': '0 20px', 'gap': '8px'})
    ], className='hdr', style={'backgroundColor': COLORS['primary'], 'padding': '12px 0', 'position': 'sticky',
              'top': '0', 'zIndex': '1000', 'boxShadow': '0 2px 8px rgba(0,0,0,0.15)'}),
    html.Div(className='fpl-stripe'),

    # Body: sidebar + content
    html.Div([

        # Mobile overlay
        html.Div(id='sidebar-overlay', n_clicks=0),

        # Sidebar nav
        html.Div(
            id='sidebar',
            className='sidebar',
            children=[
                html.Div(style={'height': '12px'}),
                html.Button('Home', id='nav-home', className='nav-item active', n_clicks=0),
                html.P('This Week', className='nav-group-label'),
                html.Button('Deadline Dashboard', id='nav-deadline', className='nav-item', n_clicks=0),
                html.P('My Team', className='nav-group-label'),
                html.Button('My Squad', id='nav-my-squad', className='nav-item', n_clicks=0),
                html.P('Leagues', className='nav-group-label'),
                html.Button('Mini-League Rivals', id='nav-rivals', className='nav-item', n_clicks=0),
                html.P('Planner', className='nav-group-label'),
                html.Button('Transfer Planner', id='nav-transfer-planner', className='nav-item', n_clicks=0),
                html.Button('Chip Planner', id='nav-chip-planner', className='nav-item', n_clicks=0),
                html.Button('Squad Builder', id='nav-squad-builder', className='nav-item', n_clicks=0),
                html.Button('Fixture Ticker', id='nav-fixture-ticker', className='nav-item', n_clicks=0),
                html.Button('Fixture Outlook', id='nav-fixture-outlook', className='nav-item', n_clicks=0),
                html.P('Research', className='nav-group-label'),
                html.P('PLAYERS & FORM', className='nav-subgroup-label'),
                html.Button('Captain Optimiser', id='nav-captain', className='nav-item', n_clicks=0),
                html.Button('Differentials', id='nav-differentials', className='nav-item', n_clicks=0),
                html.Button('Value Analysis', id='nav-value', className='nav-item', n_clicks=0),
                html.Button('Form Tracker', id='nav-form', className='nav-item', n_clicks=0),
                html.Button('Transfer Trends', id='nav-transfers', className='nav-item', n_clicks=0),
                html.P('ATTACKING', className='nav-subgroup-label'),
                html.Button('Expected Goals & Assists', id='nav-xg', className='nav-item', n_clicks=0),
                html.Button('Underlying Numbers', id='nav-underlying', className='nav-item', n_clicks=0),
                html.P('DEFENSIVE', className='nav-subgroup-label'),
                html.Button('DEFCON Bonus', id='nav-defcon-bonus', className='nav-item', n_clicks=0),
                html.Button('DEFCON: Consistency', id='nav-bonus-consistency', className='nav-item', n_clicks=0),
                html.Button('DEFCONS', id='nav-defcon', className='nav-item', n_clicks=0),
                html.Button('Clean Sheets', id='nav-cs', className='nav-item', n_clicks=0),
                html.Button('Expected Clean Sheets', id='nav-xcs', className='nav-item', n_clicks=0),
                html.P('FIXTURES', className='nav-subgroup-label'),
                html.Button('Fixture Difficulty', id='nav-fixtures', className='nav-item', n_clicks=0),
                html.P('SHOT INTELLIGENCE', className='nav-subgroup-label'),
                html.Button('Team Shot Profiles', id='nav-shot-profiles', className='nav-item', n_clicks=0),
                html.Button('Matchup Finder', id='nav-matchups', className='nav-item', n_clicks=0),
                html.Button('Chance Quality', id='nav-chance-quality', className='nav-item', n_clicks=0),
                html.P('MODEL', className='nav-subgroup-label'),
                html.Button('Model Lab', id='nav-model-lab', className='nav-item', n_clicks=0),
            ]
        ),

        # Content area — all pages live here, show/hide via callback
        html.Div(id='content-area', children=[

            # Carried-over-stats warning; populated by callback, empty in-season.
            # Must live INSIDE the content column — as a direct child of the
            # #app-body flex row it renders as a third column and shoves every
            # page off the right-hand edge of the viewport.
            html.Div(id='stale-stats-banner'),

            # =================================================================
            # SHOT INTELLIGENCE PAGES
            # =================================================================
            html.Div(id='page-shot-profiles', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Team Shot Profiles", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Where each team creates its chances and where it gives them away, from every "
                            "non-penalty shot this season (Understat shot locations). The pitch maps show the "
                            "raw shots; the bars compare the team with the league average (",
                            html.Strong("100 = average"),
                            "), split into set pieces, crosses and open play by channel. Bars are shrunk "
                            "towards the league average until a team has enough matches, so early-season "
                            "extremes are damped rather than taken at face value."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '0'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),
                    html.Div(id='sp-status'),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='sp-team', options=[{'label': t, 'value': t} for t in sorted_teams],
                                             value=sorted_teams[0] if sorted_teams else None, clearable=False),
                            ], style={'flex': '1 1 200px', 'minWidth': '180px', 'padding': '0 10px 12px'}),
                            html.Div([
                                html.Label("Show", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='sp-show', clearable=False, value='all', options=[
                                    {'label': 'All shots', 'value': 'all'},
                                    {'label': 'On target', 'value': 'on_target'},
                                    {'label': 'Goals only', 'value': 'goals'},
                                    {'label': 'Big chances (xG 0.30+)', 'value': 'big'},
                                ]),
                            ], style={'flex': '1 1 180px', 'minWidth': '160px', 'padding': '0 10px 12px'}),
                            html.Div([
                                html.Label("Matches", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='sp-recent', clearable=False, value=0, options=[
                                    {'label': 'Whole season', 'value': 0},
                                    {'label': 'Last 10', 'value': 10},
                                    {'label': 'Last 5', 'value': 5},
                                    {'label': 'Last 3', 'value': 3},
                                ]),
                            ], style={'flex': '1 1 150px', 'minWidth': '140px', 'padding': '0 10px 12px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'}),
                        html.Div([
                            html.Div([
                                html.Label("Routes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Checklist(id='sp-routes', value=['open', 'cross', 'sp'], inline=True,
                                              options=[{'label': ' Open play', 'value': 'open'},
                                                       {'label': ' Crosses', 'value': 'cross'},
                                                       {'label': ' Set pieces', 'value': 'sp'}],
                                              labelStyle={'marginRight': '16px'}),
                            ], style={'flex': '1 1 260px', 'padding': '0 10px 8px'}),
                            html.Div([
                                html.Label("Display", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.RadioItems(id='sp-display', value='shots', inline=True,
                                               options=[{'label': ' Individual shots', 'value': 'shots'},
                                                        {'label': ' Heatmap (xG per match)', 'value': 'heat'}],
                                               labelStyle={'marginRight': '16px'}),
                            ], style={'flex': '1 1 260px', 'padding': '0 10px 8px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
                        html.P(id='sp-map-count', style={'color': COLORS['text_light'], 'fontSize': '13px',
                                                        'margin': '4px 10px 0'}),
                    ], style=CARD_STYLE),
                    html.Div([
                        html.Div([
                            html.H3("Where they create", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                            html.P("Their shots, attacking the goal at the top. Bigger = higher xG; stars are goals.",
                                   style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                            dcc.Graph(id='sp-map-created', config={'displayModeBar': False}),
                        ], style={**CARD_STYLE, 'flex': '1 1 340px', 'minWidth': '300px'}),
                        html.Div([
                            html.H3("Where they concede", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                            html.P("Opponents' shots against them, into their goal at the top. "
                                   "Left/right are from the defending team's point of view.",
                                   style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                            dcc.Graph(id='sp-map-conceded', config={'displayModeBar': False}),
                        ], style={**CARD_STYLE, 'flex': '1 1 340px', 'minWidth': '300px'}),
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}),
                    html.Div([
                        html.Div([
                            html.H3("Attack profile", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                            html.P("xG created per match by route, vs league average (100). "
                                   "Above 100 = a strength.",
                                   style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                            dcc.Graph(id='sp-bars-att', config={'displayModeBar': False}),
                        ], style={**CARD_STYLE, 'flex': '1 1 340px', 'minWidth': '300px'}),
                        html.Div([
                            html.H3("Defence profile", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                            html.P("xG conceded per match by route, vs league average (100). "
                                   "Above 100 = a weakness to target.",
                                   style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                            dcc.Graph(id='sp-bars-def', config={'displayModeBar': False}),
                        ], style={**CARD_STYLE, 'flex': '1 1 340px', 'minWidth': '300px'}),
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}),
                    html.Div([
                        html.H4("All Teams", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Raw per-match figures (not shrunk). Conceded-side splits are from the defending "
                               "team's point of view and cover open play only.",
                               style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='sp-team-table', data=[],
                            columns=[
                                {'name': 'Team', 'id': 'team'},
                                {'name': 'Matches', 'id': 'matches', 'type': 'numeric'},
                                {'name': 'npxG for/m', 'id': 'npxg_for_pm', 'type': 'numeric'},
                                {'name': 'npxG against/m', 'id': 'npxg_against_pm', 'type': 'numeric'},
                                {'name': 'Set-piece xG for/m', 'id': 'sp_for_pm', 'type': 'numeric'},
                                {'name': 'Set-piece xG against/m', 'id': 'sp_against_pm', 'type': 'numeric'},
                                {'name': 'Open-play xG from crosses %', 'id': 'cross_pct_for', 'type': 'numeric'},
                                {'name': 'xG from inside box %', 'id': 'box_pct_for', 'type': 'numeric'},
                                {'name': 'Conceded inside box %', 'id': 'box_pct_against', 'type': 'numeric'},
                                {'name': 'Conceded down their left %', 'id': 'conc_their_left_pct', 'type': 'numeric'},
                                {'name': 'Conceded central %', 'id': 'conc_centre_pct', 'type': 'numeric'},
                                {'name': 'Conceded down their right %', 'id': 'conc_their_right_pct', 'type': 'numeric'},
                            ],
                            sort_action='native', page_size=20,
                            sort_by=[{'column_id': 'npxg_against_pm', 'direction': 'desc'}],
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}],
                        ),
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            html.Div(id='page-matchups', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Matchup Finder", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "FDR and expected goals rate each fixture by overall strength. This asks a different "
                            "question: ", html.Strong("does this attack's style suit this defence's weaknesses?"),
                            " Each team's chances are split into set pieces, crosses and open play by channel, and "
                            "each defence's leaks the same way. ", html.Strong("Style fit"), " is how much more "
                            "(or less) xG the matchup should produce than strength alone suggests, e.g. a "
                            "cross-heavy attack meeting a defence that leaks crosses. The tag shows the biggest "
                            "single edge."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div(id='mu-validation'),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),
                    html.Div(id='mu-status'),
                    html.Div([
                        html.Div([
                            html.Label("Gameweeks ahead", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Slider(id='mu-gws', min=1, max=6, step=1, value=4,
                                       marks={i: str(i) for i in range(1, 7)}),
                        ], style={'flex': '0 1 320px', 'minWidth': '220px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("Position (target lists)", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(id='mu-position', options=[{'label': 'All', 'value': 'All'}] +
                                         [{'label': p, 'value': p} for p in ['DEF', 'MID', 'FWD']],
                                         value='All', clearable=False),
                        ], style={'flex': '0 1 200px', 'minWidth': '150px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("Min. minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Input(id='mu-minutes', type='number', value=DEF_MINS_LO, min=0, step=50,
                                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                             'border': '1px solid #ccc'}),
                        ], style={'flex': '0 1 140px', 'minWidth': '100px', 'padding': '0 10px'}),
                    ], style={**CARD_STYLE, 'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'}),
                    html.Div([
                        html.H3("Style Fit by Fixture", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                        html.P("Green = the attacking team's style suits this opponent; red = it plays into the "
                               "opponent's strengths. Tags: Set pcs, Crosses, Central, Left/Right = attacking "
                               "down their own left/right. Hover for the numbers.",
                               style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                        html.Div(dcc.Graph(id='mu-heatmap', config={'displayModeBar': False}),
                                 className='chart-scroll'),
                    ], style=CARD_STYLE),
                    html.Div([
                        html.H4("Set-Piece Targets", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players who get on the end of (or deliver) set-piece chances, facing defences that "
                               "concede them. Opponent index: 1.0 = league-average set-piece defence, 1.4 = concedes "
                               "40% more. Horizon SP xGI = their set-piece threat per 90 x the opponent indices.",
                               style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='mu-sp-table', data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'SP threat/90', 'id': 'sp_threat_90', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                {'name': 'Headers %', 'id': 'head_pct', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'SP chances made', 'id': 'sp_created', 'type': 'numeric'},
                                {'name': 'Fixtures (opp index)', 'id': 'fixtures'},
                                {'name': 'Horizon SP xGI', 'id': 'horizon', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native', page_size=15,
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}],
                        ),
                    ], style=CARD_STYLE),
                    html.Div([
                        html.H4("Flank Targets", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Wide players and attacking full-backs whose side of the pitch lines up with where "
                               "their opponents leak chances. Flank edge = how much their output should rise (or "
                               "fall) over the horizon, weighted by how much of their shooting comes down that side.",
                               style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='mu-flank-table', data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Side', 'id': 'side_label'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'npxGI/90', 'id': 'xgi_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Shots down that side %', 'id': 'channel_pct', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'Fixtures (opp index)', 'id': 'fixtures'},
                                {'name': 'Flank edge %', 'id': 'edge_pct', 'type': 'numeric', 'format': {'specifier': '+.0f'}},
                                {'name': 'Horizon npxGI', 'id': 'horizon', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native', page_size=15,
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{edge_pct} >= 10', 'column_id': 'edge_pct'},
                                 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
                                {'if': {'filter_query': '{edge_pct} <= -10', 'column_id': 'edge_pct'},
                                 'backgroundColor': '#fde8ef'},
                            ],
                        ),
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            html.Div(id='page-chance-quality', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Chance Quality", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Two players with the same xG can get there very differently: lots of speculative long "
                            "shots, or a few chances from six yards. ", html.Strong("xG per shot"), ", box share and ",
                            html.Strong("big chances"), " (xG 0.30+) show who gets into the positions that produce "
                            "hauls; the creation columns show who sets them up, and how (crosses, set pieces). "
                            "Non-penalty throughout, so penalty takers aren't flattered."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '0'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),
                    html.Div(id='cq-status'),
                    html.Div([
                        html.Div([
                            html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(id='cq-position', options=[{'label': 'All', 'value': 'All'}] +
                                         [{'label': p, 'value': p} for p in ['DEF', 'MID', 'FWD']],
                                         value='All', clearable=False),
                        ], style={'flex': '0 1 200px', 'minWidth': '150px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Dropdown(id='cq-team', options=[{'label': 'All', 'value': 'All'}] +
                                         [{'label': t, 'value': t} for t in sorted_teams],
                                         value='All', clearable=False),
                        ], style={'flex': '0 1 240px', 'minWidth': '150px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("Min. minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Input(id='cq-minutes', type='number', value=DEF_MINS_LO, min=0, step=50,
                                      style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                             'border': '1px solid #ccc'}),
                        ], style={'flex': '0 1 140px', 'minWidth': '100px', 'padding': '0 10px'}),
                    ], style={**CARD_STYLE, 'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'}),
                    html.Div([
                        html.H3("Quality vs Volume", style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                        html.P("Top right = shoots often AND from good positions. Bottom right = volume shooter "
                               "from positions that produce low quality chances. Top left = few but excellent chances. Bubble size = npxG per 90.",
                               style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                        dcc.Graph(id='cq-scatter', config={'displayModeBar': False}),
                    ], style=CARD_STYLE),
                    html.Div([
                        html.H4("Player Chance Profiles", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='cq-table', data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'Shots/90', 'id': 'shots_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'npxG/90', 'id': 'npxg_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xG/shot', 'id': 'xg_per_shot', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                {'name': 'Box %', 'id': 'box_pct', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'Big chances/90', 'id': 'big_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Headers %', 'id': 'head_pct', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'Chances made/90', 'id': 'chances_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xA/90', 'id': 'xa_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'From crosses %', 'id': 'cross_pct', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'SP threat/90', 'id': 'sp_threat_90', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                                {'name': 'Side', 'id': 'side_label'},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native', page_size=20,
                            sort_by=[{'column_id': 'npxg_90', 'direction': 'desc'}],
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}],
                        ),
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            # HOME PAGE
            html.Div(id='page-home', style={'display': 'block'}, children=[
                html.Div(id='home-content'),

                # RANK CONGESTION TOOL — static so interval never resets it
                html.Div([
                    html.H2("Rank Congestion Tool",
                            style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                    html.P("Enter two overall ranks to see the points gap between them and how congested that band is.",
                           style={'color': COLORS['text_light']})
                ], style={'marginBottom': '24px', 'padding': '20px 0 0 0'}),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Label("Your rank",
                                       style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Input(id='rank-your', type='number', value=None, min=1, step=1,
                                      placeholder='e.g. 444000',
                                      style={'width': '100%', 'padding': '10px', 'borderRadius': '6px',
                                             'border': '1px solid #ccc', 'fontSize': '15px'})
                        ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("Rival rank",
                                       style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            dcc.Input(id='rank-rival', type='number', value=None, min=1, step=1,
                                      placeholder='e.g. 22000',
                                      style={'width': '100%', 'padding': '10px', 'borderRadius': '6px',
                                             'border': '1px solid #ccc', 'fontSize': '15px'})
                        ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                        html.Div([
                            html.Label("\u00a0", style={'display': 'block', 'marginBottom': '6px'}),
                            html.Button("Analyse", id='rank-check-btn', n_clicks=0,
                                        style={
                                            'backgroundColor': COLORS['primary'], 'color': 'white',
                                            'border': 'none', 'padding': '10px 24px', 'borderRadius': '6px',
                                            'fontWeight': '600', 'fontSize': '15px', 'cursor': 'pointer',
                                            'width': '100%'
                                        })
                        ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end',
                              'marginBottom': '20px'}),
                    html.Div(id='rank-result')
                ], style=CARD_STYLE),
            ]),

            # DEFCON BONUS PAGE
            html.Div(id='page-defcon-bonus', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Understanding Defensive Contribution Bonuses",
                                style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P(["Players earn ", html.Strong("2 bonus points"), " when they hit the defcon threshold: ",
                                html.Strong(f"{DEF_THR}+ for DEF"), " or ", html.Strong(f"{MID_THR}+ for MID/FWD"),
                                " in a single match."],
                               style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([html.Span(f"Target: {DEF_THR} DEFCON/90 (DEF) | {MID_THR} DEFCON/90 (MID/FWD)",
                                            style={'backgroundColor': COLORS['secondary'],
                                                   'color': COLORS['primary'], 'padding': '8px 16px',
                                                   'borderRadius': '20px', 'fontWeight': '600'})])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='bonus-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                          [{'label': p, 'value': p} for p in
                                                                           ['DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='bonus-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                      [{'label': t, 'value': t} for t in
                                                                       sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='bonus-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='bonus-minutes', type='number', value=DEF_MINS_HI, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Defcon Per 90 vs Bonus Threshold",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            f"Purple line = DEF threshold ({DEF_THR}). Pink line = MID/FWD threshold ({MID_THR}). "
                            "Players above consistently earn defcon bonuses.",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='bonus-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Distance from Bonus Threshold",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            f"How far above or below their position threshold (DEF: {DEF_THR}, MID/FWD: {MID_THR}) each player averages.",
                            style={'color': COLORS['text_light']}),
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
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'Defcon', 'id': 'defcon', 'type': 'numeric'},
                                {'name': 'Defcon/90', 'id': 'defcon_per_90', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'vs Bonus', 'id': 'defcon_vs_bonus', 'type': 'numeric',
                                 'format': {'specifier': '+.2f'}},
                                {'name': '% Target', 'id': 'bonus_rate', 'type': 'numeric',
                                 'format': {'specifier': '.0f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{defcon_vs_bonus} >= 0', 'column_id': 'defcon_vs_bonus'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{defcon_vs_bonus} < 0', 'column_id': 'defcon_vs_bonus'},
                                 'backgroundColor': '#fde8ef'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # BONUS CONSISTENCY TAB
            # BONUS CONSISTENCY PAGE
            html.Div(id='page-bonus-consistency', style={'display': 'none'}, children=[
                html.Div([
                    # Explanation Card
                    html.Div([
                        html.H3("DEFCON Bonus Consistency Analysis",
                                style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "This shows how ", html.Strong("consistently"),
                            f" players hit their defcon bonus threshold (DEF: {DEF_THR}+, MID/FWD: {MID_THR}+) in individual matches. ",
                            "A player averaging the threshold per 90 minutes might be inconsistent (20 one week, 0 the next) vs someone who reliably hits it most games."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("Based on games with a minimum of 60 minutes played",
                                      style={'backgroundColor': COLORS['secondary'],
                                             'color': COLORS['primary'], 'padding': '8px 16px', 'borderRadius': '20px',
                                             'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Filters
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='consistency-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                                [{'label': p, 'value': p} for p in
                                                                                 ['DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='consistency-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                            [{'label': t, 'value': t} for t in
                                                                             sorted(df['team_name'].unique())],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='consistency-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. games",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='consistency-games', type='number', value=DEF_MIN_GAMES, min=1, step=1,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='consistency-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Chart - Hit Rate Distribution
                    html.Div([
                        html.H3("Bonus Hit Rate by Player", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            f"Percentage of APPEARANCES where the player hit their bonus threshold "
                            f"(DEF: {DEF_THR}+, MID/FWD: {MID_THR}+).",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='consistency-bar')
                    ], style=CARD_STYLE),

                    # Scatter - Hit Rate vs Avg Defcon
                    html.Div([
                        html.H3("Consistency vs Average Output",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            "Compare hit rate (consistency) against average defcon per appearance. Top right = "
                            "high output AND consistent. Defensive contribution points carry no 60-minute "
                            "requirement, so every appearance counts \u2014 the '60+ mins' column shows how "
                            "many were full games.",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='consistency-scatter')
                    ], style=CARD_STYLE),

                    # Table
                    html.Div([
                        html.H4("Bonus Consistency Rankings",
                                style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='consistency-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'Apps', 'id': 'qualifying_games', 'type': 'numeric'},
                                {'name': '60+ mins', 'id': 'starts_60', 'type': 'numeric'},
                                {'name': 'Bonus Games', 'id': 'bonus_games', 'type': 'numeric'},
                                {'name': 'Hit Rate %', 'id': 'hit_rate', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Avg Defcon', 'id': 'avg_defcon_qualifying', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
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
                                {'if': {'filter_query': '{hit_rate} >= 50', 'column_id': 'hit_rate'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{hit_rate} >= 25 && {hit_rate} < 50', 'column_id': 'hit_rate'},
                                 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{hit_rate} < 25', 'column_id': 'hit_rate'},
                                 'backgroundColor': '#fde8ef'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # DEFENSIVE CONTRIBUTIONS TAB
            # DEFCONS PAGE
            html.Div(id='page-defcon', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='defcon-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                           [{'label': p, 'value': p} for p in
                                                                            ['DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='defcon-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                       [{'label': t, 'value': t} for t in
                                                                        sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='defcon-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='defcon-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Actual vs Expected Defensive Contributions",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players above the diagonal are outperforming expectations.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='defcon-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Top Defensive Contributors",
                                style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='defcon-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'Defcon', 'id': 'defcon', 'type': 'numeric'},
                                {'name': 'Defcon/90', 'id': 'defcon_per_90', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'xDefcon', 'id': 'expected_defcon', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Diff', 'id': 'defcon_diff', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{defcon_diff} > 0', 'column_id': 'defcon_diff'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{defcon_diff} < 0', 'column_id': 'defcon_diff'},
                                 'backgroundColor': '#fde8ef'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # XG TAB
            # XG PAGE
            html.Div(id='page-xg', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xg-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                       [{'label': p, 'value': p} for p in
                                                                        ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xg-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                   [{'label': t, 'value': t} for t in
                                                                    sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='xg-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='xg-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Goals Scored vs Expected Goals",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("See who is outperforming and underperforming their expected goals.",
                               style={'color': COLORS['text_light']}),
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
                                {'name': 'xG', 'id': 'expected_goals', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'xG Diff', 'id': 'xg_diff', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Assists', 'id': 'assists', 'type': 'numeric'},
                                {'name': 'xA', 'id': 'expected_assists', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
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
                                {'if': {'filter_query': '{xg_diff} < -1', 'column_id': 'xg_diff'},
                                 'backgroundColor': "#FFEBEE"},
                                {'if': {'filter_query': '{xg_diff} >= -1 && {xg_diff} <= 1', 'column_id': 'xg_diff'},
                                 'backgroundColor': '#FFB938'},
                                {'if': {'filter_query': '{xg_diff} > 1', 'column_id': 'xg_diff'},
                                 'backgroundColor': '#e6fff2'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # UNDERLYING NUMBERS TAB
            # UNDERLYING NUMBERS PAGE
            html.Div(id='page-underlying', style={'display': 'none'}, children=[
                html.Div([
                    # Explanation
                    html.Div([
                        html.H3("Player Underlying Numbers", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Actual returns are noisy. A player can blank for weeks then bag a hat trick. ",
                            html.Strong("Underlying numbers"), " (xG, xA, threat, creativity) measure the ",
                            html.Strong("quality and volume of chances"), " a player is involved in, which is a far better predictor of future points. ",
                            "Look for players with strong per-90 rates who are ", html.Strong("underperforming their xGI"),
                            " they're due a correction upward."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("All stats normalised per 90 minutes for fair comparison",
                                      style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                             'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Filters
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='under-position', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': p, 'value': p} for p in ['GKP', 'DEF', 'MID', 'FWD']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='under-team', options=[{'label': 'All', 'value': 'All'}] +
                                             [{'label': t, 'value': t} for t in sorted(df['team_name'].unique())], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='under-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='under-minutes', type='number', value=DEF_MINS_HI, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Scatter — xGI/90 vs actual GI/90
                    html.Div([
                        html.H3("Actual vs Expected Goal Involvements (per 90)", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players below the diagonal are underperforming their underlying numbers",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='under-scatter')
                    ], style=CARD_STYLE),

                    # Table
                    html.Div([
                        html.H4("Underlying Numbers Scouting Report", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='under-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'xGI/90', 'id': 'xgi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'GI/90', 'id': 'gi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xGI Diff', 'id': 'xgi_diff_per_90', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                                {'name': 'xG/90', 'id': 'xg_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xA/90', 'id': 'xa_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Threat/90', 'id': 'threat_per_90', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'Create/90', 'id': 'creativity_per_90', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                                {'name': 'ICT/90', 'id': 'ict_per_90', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'BPS/90', 'id': 'bps_per_90', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{xgi_diff_per_90} < 0.00', 'column_id': 'xgi_diff_per_90'}, 'backgroundColor': '#fde8ef'},
                                {'if': {'filter_query': '{xgi_diff_per_90} > 0.00', 'column_id': 'xgi_diff_per_90'}, 'backgroundColor': '#e6fff2'},
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # VALUE TAB
            # VALUE PAGE
            html.Div(id='page-value', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='value-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                          [{'label': p, 'value': p} for p in
                                                                           ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='value-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                      [{'label': t, 'value': t} for t in
                                                                       sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='value-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='value-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Points vs Price", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Find the best value players by points returned per £1m invested.",
                               style={'color': COLORS['text_light']}),
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
                                {'name': 'Pts/£m', 'id': 'points_per_million', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
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
            # FORM PAGE
            html.Div(id='page-form', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='form-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                         [{'label': p, 'value': p} for p in
                                                                          ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='form-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                     [{'label': t, 'value': t} for t in
                                                                      sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='form-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='form-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Regression Watchlist", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("The discipline tool against chasing hauls. SELL: scoring well above underlying xGI "
                               "(the goals are borrowed — expect payback). BUY EARLY: elite underlying numbers the "
                               "goals haven't caught up with yet — get in before the price and ownership move.",
                               style={'color': COLORS['text_light']}),
                        html.Div([
                            html.Div([
                                html.H4("Sell-High Candidates", style={'color': COLORS['danger_text'], 'marginBottom': '10px'}),
                                dash_table.DataTable(
                                    id='regress-sell-table', data=[],
                                    columns=[
                                        {'name': 'Player', 'id': 'web_name'},
                                        {'name': 'Team', 'id': 'team_name'},
                                        {'name': 'GI/90', 'id': 'gi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'xGI/90', 'id': 'xgi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'Overperf', 'id': 'xgi_diff_per_90', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                                        {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    ],
                                    page_size=8, style_cell=TABLE_STYLE_CELL,
                                    style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
                                )
                            ], style={'flex': '1', 'minWidth': '320px', 'paddingRight': '10px'}),
                            html.Div([
                                html.H4("Buy-Early Candidates", style={'color': COLORS['success_text'], 'marginBottom': '10px'}),
                                dash_table.DataTable(
                                    id='regress-buy-table', data=[],
                                    columns=[
                                        {'name': 'Player', 'id': 'web_name'},
                                        {'name': 'Team', 'id': 'team_name'},
                                        {'name': 'GI/90', 'id': 'gi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'xGI/90', 'id': 'xgi_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'Underperf', 'id': 'xgi_diff_per_90', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
                                        {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                    ],
                                    page_size=8, style_cell=TABLE_STYLE_CELL,
                                    style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
                                )
                            ], style={'flex': '1', 'minWidth': '320px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Form vs Season Average", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players trending up or down from their season average.",
                               style={'color': COLORS['text_light']}),
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
                                {'name': 'Form Diff', 'id': 'form_vs_season', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'True xGI/90', 'id': 'adj_xgi90', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Sched', 'id': 'sched_factor', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=15,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{form_vs_season} > 1', 'column_id': 'form_vs_season'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{form_vs_season} < -1', 'column_id': 'form_vs_season'},
                                 'backgroundColor': '#fde8ef'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # CLEAN SHEETS TAB
            # CLEAN SHEETS PAGE
            html.Div(id='page-cs', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cs-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                       [{'label': p, 'value': p} for p in
                                                                        ['GKP', 'DEF']], value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cs-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                   [{'label': t, 'value': t} for t in
                                                                    sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='cs-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='cs-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Clean Sheets vs Goals Conceded",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Best assets appear in the top left quadrant (high CS, low GC).",
                               style={'color': COLORS['text_light']}),
                        html.P("cs = Clean sheet, gc = Goals conceded", style={'color': COLORS['text_light']}),
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
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
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

            # FIXTURE TICKER TAB
            # FIXTURE TICKER PAGE
            html.Div(id='page-fixture-outlook', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Fixture Outlook", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "FDR asks ", html.Strong("how difficult is this opponent"), ". This looks ",
                            html.Strong("at what this team could actually produce in this fixture"),
                            " \u2014 which for a weak side is a very different question. A "
                            "promoted team at home to a poor defence is a green cell on FDR and "
                            "still a low-scoring fixture here, because their own attack rating "
                            "holds the number down.",
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '10px'}),
                        html.P([
                            html.Strong("Attack"), " shows expected goals scored per fixture. ",
                            html.Strong("Defence"), " shows clean-sheet probability. They are "
                            "separate because they diverge \u2014 a run against three low-scoring "
                            "sides is good for your defenders and poor for your forwards, and one "
                            "number cannot say that. Totals sum across the window, so a blank "
                            "contributes nothing and a double contributes an extra fixture.",
                        ], style={'color': COLORS['text_light'], 'fontSize': '14px', 'marginBottom': '0'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Gameweeks to show",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='fo-gws', type='number', value=6, min=1, max=38, step=1,
                                          debounce=True,
                                          style={'width': '100%', 'padding': '9px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'}),
                                html.Div("Type any number from 1 to 38",
                                         style={'color': COLORS['text_light'], 'fontSize': '12px',
                                                'marginTop': '4px'}),
                            ], style={'flex': '1', 'minWidth': '160px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("View",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fo-view', clearable=False, value='attack',
                                             options=[{'label': ' Attack (expected goals)', 'value': 'attack'},
                                                      {'label': ' Defence (clean sheet %)', 'value': 'defence'}]),
                            ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Sort teams by",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fo-sort', clearable=False, value='total',
                                             options=[{'label': ' Best run first', 'value': 'total'},
                                                      {'label': ' Worst run first', 'value': 'total_asc'},
                                                      {'label': ' Team name', 'value': 'name'}]),
                            ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-start'})
                    ], style=CARD_STYLE),

                    html.Div([
                        dcc.Loading(
                            html.Div(
                                dcc.Graph(id='fo-heatmap', config={'displayModeBar': False}),
                                className='chart-scroll'),
                            type='circle', color=COLORS['primary']),
                        html.P("If needed, swipe sideways to see the rest of the window",
                               style={'color': COLORS['text_light'], 'fontSize': '12px',
                                      'margin': '6px 0 0 0'}),
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Run Summary", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Worst GW is the single weakest fixture in the window. A run of "
                               "2,2,2,5,2 and one of 3,3,3,3,3 average produce a similar average yet plan "
                               "differently.",
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Div(id='fo-table')
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            html.Div(id='page-fixture-ticker', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Team Fixture Ticker", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Every team's remaining fixtures at a glance. ",
                            html.Strong("Blanks"), " (grey) = no fixture that gameweek. ",
                            html.Strong("Doubles"), " (cyan) = two fixtures in one gameweek. ",
                            "Single fixtures are colour-coded by FDR: ",
                            html.Strong("Green = Easy"), ", ",
                            html.Strong("Amber = Medium"), ", ",
                            html.Strong("Red = Hard"), "."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '16px'}),
                        html.Div([
                            html.Span("■", style={'color': '#00ff87', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("FDR 1", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#7dde9e', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("FDR 2", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#ffc107', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("FDR 3", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#ff7043', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("FDR 4", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#dc3545', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("FDR 5", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#00bcd4', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("Double GW", style={'fontSize': '13px', 'marginRight': '14px', 'color': COLORS['text_dark']}),
                            html.Span("■", style={'color': '#d0d0d0', 'fontSize': '20px', 'marginRight': '4px'}),
                            html.Span("Blank GW", style={'fontSize': '13px', 'color': COLORS['text_dark']}),
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Gameweeks to show", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(
                                    id='ticker-gws',
                                    min=1, max=38, step=1, value=1,
                                    marks={1: '1', 5: '5', 10: '10', 15: '15', 20: '20',
                                           25: '25', 30: '30', 35: '35', 38: 'All'},
                                    included=True,
                                    tooltip={'placement': 'bottom',
                                             'always_visible': True}
                                )
                            ], style={'flex': '0 1 420px', 'minWidth': '260px',
                                      'maxWidth': '420px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Sort teams by", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='ticker-sort',
                                    options=[
                                        {'label': 'Alphabetical', 'value': 'name'},
                                        {'label': 'Easiest run first (avg FDR in window)', 'value': 'fdr'},
                                    ],
                                    value='name',
                                    clearable=False
                                )
                            ], style={'flex': '1', 'minWidth': '220px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Fixture Ticker",
                                id='ticker-title',
                                style={'color': COLORS['primary'], 'marginBottom': '4px'}),
                        html.P("Opponent shown in each cell (H = home, A = away). "
                               "Double GW cells show both fixtures. Beyond 15 gameweeks the "
                               "cell text is hidden to keep the grid readable. In this case hover over any "
                               "cell for the fixture.",
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Div(
                            dcc.Graph(id='ticker-heatmap', config={'displayModeBar': False}),
                            className='chart-scroll'),
                        html.P("Swipe sideways to see the rest of the window.",
                               style={'color': COLORS['text_light'], 'fontSize': '12px',
                                      'margin': '6px 0 0 0'}),
                    ], style=CARD_STYLE),

                ], style={'padding': '20px 0'})
            ]),

            # FIXTURE DIFFICULTY TAB
            # FIXTURE DIFFICULTY PAGE
            html.Div(id='page-fixtures', style={'display': 'none'}, children=[
                html.Div([
                    # Explanation Card
                    html.Div([
                        html.H3("Fixture Difficulty Rating (FDR) Analysis",
                                style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Rank players by their team's upcoming fixture difficulty. ",
                            html.Strong("Lower FDR = easier fixtures"), ". ",
                            "FDR ranges from 1 (very easy) to 5 (very hard). Use this tab to see which teams have easier fixtures and potentially target players from those teams."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("Next 5 Gameweeks", style={'backgroundColor': COLORS['secondary'],
                                                                 'color': COLORS['primary'], 'padding': '8px 16px',
                                                                 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Filters
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fdr-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                        [{'label': p, 'value': p} for p in
                                                                         ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='fdr-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                    [{'label': t, 'value': t} for t in
                                                                     sorted(df['team_name'].unique())], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='fdr-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'£{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='fdr-minutes', type='number', value=DEF_MINS_LO, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Fixture swing detector
                    html.Div([
                        html.H3("Fixture Swings — Buy the Run Early", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Teams whose attacking fixtures change sharply between the next 3 GWs and the 3 after. "
                               "A big improvement means buy their attackers BEFORE the run starts — getting there a "
                               "week early is the cheap edge. Negative swing = fixtures about to turn bad.",
                               style={'color': COLORS['text_light']}),
                        dash_table.DataTable(
                            id='fdr-swing-table', data=[],
                            columns=[
                                {'name': 'Team', 'id': 'team'},
                                {'name': 'aFDR GW+1..3', 'id': 'now_att', 'type': 'numeric'},
                                {'name': 'aFDR GW+4..6', 'id': 'later_att', 'type': 'numeric'},
                                {'name': 'Attack Swing', 'id': 'att_swing', 'type': 'numeric',
                                 'format': {'specifier': '+.2f'}},
                                {'name': 'dFDR GW+1..3', 'id': 'now_def', 'type': 'numeric'},
                                {'name': 'dFDR GW+4..6', 'id': 'later_def', 'type': 'numeric'},
                                {'name': 'Defence Swing', 'id': 'def_swing', 'type': 'numeric',
                                 'format': {'specifier': '+.2f'}},
                            ],
                            sort_action='native', page_size=20,
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{att_swing} <= -0.4', 'column_id': 'att_swing'},
                                 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
                                {'if': {'filter_query': '{att_swing} >= 0.4', 'column_id': 'att_swing'},
                                 'backgroundColor': '#fde8ef'},
                                {'if': {'filter_query': '{def_swing} <= -0.4', 'column_id': 'def_swing'},
                                 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
                                {'if': {'filter_query': '{def_swing} >= 0.4', 'column_id': 'def_swing'},
                                 'backgroundColor': '#fde8ef'},
                            ]
                        ),
                        html.P("Negative swing (green) = later fixtures are EASIER than the current window — buy window. "
                               "Swings under \u00b10.4 are noise.",
                               style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginTop': '10px'})
                    ], style=CARD_STYLE),

                    # Team FDR Chart
                    html.Div([
                        html.H3("Team Fixture Difficulty (Next 5 GWs)",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Teams sorted by average FDR. Green = easy run, Orange = average run and Red = tough run.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='fdr-team-bar')
                    ], style=CARD_STYLE),

                    # Player scatter
                    html.Div([
                        html.H3("Player Value vs Fixture Difficulty",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            "Find high value players with easy fixtures. Best picks are top left (high points, low FDR).",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='fdr-scatter')
                    ], style=CARD_STYLE),

                    # Table
                    html.Div([
                        html.H4("Players Ranked by Fixture Difficulty",
                                style={'color': COLORS['primary'], 'marginBottom': '16px'}),
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
                                {'name': 'Avg FDR', 'id': 'avg_fdr_5', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Next 5 Fixtures', 'id': 'fixture_string'},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{avg_fdr_5} <= 2.5', 'column_id': 'avg_fdr_5'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{avg_fdr_5} > 2.5 && {avg_fdr_5} <= 3.5',
                                        'column_id': 'avg_fdr_5'}, 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{avg_fdr_5} > 3.5', 'column_id': 'avg_fdr_5'},
                                 'backgroundColor': '#fde8ef'}
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # =================================================================
            # OWNERSHIP DIFFERENTIALS TAB
            # =================================================================
            # OWNERSHIP DIFFERENTIALS PAGE
            html.Div(id='page-differentials', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Ownership Differentials", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "FPL ranking is ", html.Strong("relative"), ". You gain rank by owning players ",
                            html.Strong("most managers don't"), " but only if those players score well. ",
                            "This tab cross references strong underlying stats (form, xGI, PPG) with low ownership ",
                            "to surface the highest upside differentials."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("Target: <10% ownership with above average output",
                                      style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                             'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='diff-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                         [{'label': p, 'value': p} for p in
                                                                          ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='diff-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                     [{'label': t, 'value': t} for t in sorted_teams],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='diff-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. ownership %",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                html.Div([
                                    html.Div(
                                        dcc.Slider(id='diff-max-own', min=5, max=100, step=1, value=15,
                                                   marks={i: f'{i}%' for i in [5, 25, 50, 75, 100]}),
                                        style={'flex': '1'}
                                    ),
                                    dcc.Input(id='diff-max-own-input', type='number', value=10, min=5, max=100, step=1,
                                              style={'width': '70px', 'marginLeft': '12px', 'padding': '8px',
                                                     'borderRadius': '4px', 'border': '1px solid #ccc'})
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='diff-minutes', type='number', value=DEF_MINS_HI, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("PPG vs Ownership", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Top left quadrant = high output, low ownership. These are your rank gainers.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='diff-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Top Differentials by Composite Score",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Weighted score combining PPG, form, and low ownership.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='diff-bar')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Differential Rankings", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='diff-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Points', 'id': 'total_points', 'type': 'numeric'},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'PPG', 'id': 'ppg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xGI', 'id': 'expected_goal_involvements', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'EO% (Top)', 'id': 'top_eo', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Own \u03947d', 'id': 'own_delta_7d', 'type': 'numeric',
                                 'format': {'specifier': '+.1f'}},
                                {'name': 'Diff Score', 'id': 'differential_score', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Proj Next 5', 'id': 'proj_pts_5', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Avg FDR', 'id': 'avg_fdr_5', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Next 5', 'id': 'fixture_string'},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{ownership} <= 5', 'column_id': 'ownership'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{ownership} > 5 && {ownership} <= 10',
                                        'column_id': 'ownership'}, 'backgroundColor': '#fff8e1'},
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # =================================================================
            # CAPTAIN OPTIMISER TAB
            # =================================================================
            # CAPTAIN OPTIMISER PAGE
            html.Div(id='page-captain', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Captain Pick Optimiser", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Captaincy is the ", html.Strong("single biggest rank differentiator"), " in FPL. ",
                            "Your captain's points are doubled, so getting it right every week compounds massively. ",
                            "This tool scores candidates 0\u2013100 with every input normalized across the pool, so the weights are true relative importances: ",
                            html.Strong(
                                "Form (25%), xGI/90 (20%), PPG (15%), Attack-fixture ease (15%), BPS/90 (10%), Venue PPG (10%), Differential (5%)"),
                            ". Scores are then discounted by availability flags and recent start rate \u2014 a great score means nothing on a 25% flag or a rotation risk."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span(f"Next fixture: GW{next_gw_num}",
                                      style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                             'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cap-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                        [{'label': p, 'value': p} for p in
                                                                         ['DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='cap-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                    [{'label': t, 'value': t} for t in sorted_teams],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='cap-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='cap-minutes', type='number', value=DEF_MINS_HI, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Mode", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.RadioItems(
                                    id='cap-mode',
                                    options=[
                                        {'label': ' Protect (expected pts)', 'value': 'ev'},
                                        {'label': ' Chase (P of 15+)', 'value': 'ceiling'},
                                        {'label': ' Differential (gain vs field)', 'value': 'gain'},
                                    ],
                                    value='ev', inline=True,
                                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'}
                                )
                            ], style={'flex': '2', 'minWidth': '300px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Top Captain Picks (Weighted Score)",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Composite score combining form, xGI, fixtures, BPS, and home/away splits.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='cap-bar')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Home vs Away PPG", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            "Players above the diagonal perform better at home. Use this alongside next fixture venue. Only the top 150 in form players will have Home and Away PPG.",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='cap-ha-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Captain Rankings", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='cap-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Proj Pts', 'id': 'proj_pts_next', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'P(10+)', 'id': 'p_10', 'type': 'numeric',
                                 'format': {'specifier': '.0f'}},
                                {'name': 'P(15+)', 'id': 'p_15', 'type': 'numeric',
                                 'format': {'specifier': '.0f'}},
                                {'name': 'Gain vs field', 'id': 'captain_gain', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Composite', 'id': 'captain_score', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Involv. haul %', 'id': 'haul_pct', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Env \u00d7', 'id': 'att_env_next', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'FPL xP', 'id': 'ep_next', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'PPG', 'id': 'ppg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'xGI/90', 'id': 'xgi_per_90', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Next', 'id': 'next_opponent'},
                                {'name': 'Venue', 'id': 'next_venue'},
                                {'name': 'aFDR', 'id': 'next_att_fdr', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Venue PPG', 'id': 'venue_ppg', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Start%', 'id': 'start_rate', 'type': 'numeric',
                                 'format': {'specifier': '.0f'}},
                                {'name': 'Avail%', 'id': 'avail_pct', 'type': 'numeric',
                                 'format': {'specifier': '.0f'}},
                                {'name': 'Set Pieces', 'id': 'set_pieces'},
                                {'name': 'BPS/90', 'id': 'bps_per_90', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'EO% (Top)', 'id': 'top_eo', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{next_att_fdr} <= 2.3', 'column_id': 'next_att_fdr'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{next_att_fdr} >= 3.7', 'column_id': 'next_att_fdr'},
                                 'backgroundColor': '#fde8ef'},
                                {'if': {'filter_query': '{next_venue} = H', 'column_id': 'next_venue'},
                                 'color': COLORS['success_text'], 'fontWeight': '600'},
                                {'if': {'filter_query': '{next_venue} = A', 'column_id': 'next_venue'},
                                 'color': COLORS['danger_text'], 'fontWeight': '600'},
                                {'if': {'filter_query': '{avail_pct} < 100', 'column_id': 'avail_pct'},
                                 'backgroundColor': '#fde8ef', 'fontWeight': '600'},
                                {'if': {'filter_query': '{start_rate} < 70', 'column_id': 'start_rate'},
                                 'backgroundColor': '#fff8e1'},
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # =================================================================
            # TRANSFER TRENDS TAB
            # =================================================================
            # TRANSFER TRENDS PAGE
            html.Div(id='page-transfers', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Transfer Trends & Price Prediction",
                                style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Getting ahead of price changes by even ", html.Strong("one day"),
                            " compounds over a season. ",
                            "Every 0.1m saved means better players later. This tab tracks net transfer velocity and estimates ",
                            "which players are closest to a price rise or fall based on transfer-to-ownership ratios."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("Price changes happen overnight based on transfer activity",
                                      style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                             'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Position",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xfer-position', options=[{'label': 'All', 'value': 'All'}] +
                                                                         [{'label': p, 'value': p} for p in
                                                                          ['GKP', 'DEF', 'MID', 'FWD']], value='All',
                                             clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='xfer-team', options=[{'label': 'All', 'value': 'All'}] +
                                                                     [{'label': t, 'value': t} for t in sorted_teams],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Max. price (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(id='xfer-price', min=4, max=16, step=0.5, value=16,
                                           marks={i: f'{i}' for i in [4, 6, 8, 10, 12, 14, 16]},
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           )
                            ], style={'flex': '2', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='xfer-minutes', type='number', value=0, min=0, step=10, debounce=True,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Your Price Alerts", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Protecting team value early season compounds into an extra player by January. "
                               "Load your squad: players at risk of dropping tonight (sell or accept), plus the "
                               "top-projected non-owned players about to rise (buy before the price does).",
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Div([
                            dcc.Input(id='pa-team-id', type='number', placeholder='FPL Team ID',
                                      style={'padding': '10px 14px', 'borderRadius': '6px',
                                             'border': f'2px solid {COLORS["primary"]}',
                                             'fontSize': '15px', 'width': '180px', 'marginRight': '12px'}),
                            html.Button("Check My Price Risk", id='pa-load-btn', n_clicks=0,
                                        style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                               'border': 'none', 'padding': '10px 24px', 'borderRadius': '6px',
                                               'fontSize': '14px', 'fontWeight': '700', 'cursor': 'pointer'}),
                        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '8px'}),
                        dcc.Loading(html.Div(id='pa-result'), type='circle', color=COLORS['primary'])
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Likely Price Risers", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            "Players with the highest net transfer-in velocity relative to ownership. Buy before the rise.",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='xfer-risers-bar')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Likely Price Fallers", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Players being sold fastest. Sell before the drop to preserve team value.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='xfer-fallers-bar')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3("Transfer Momentum vs Season Price Change",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("See which players still have room to rise or have further to fall.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='xfer-scatter')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Transfer Activity This Gameweek",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(id='xfer-delta-basis',
                               style={'color': COLORS['text_light'], 'fontSize': '13px',
                                      'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='xfer-table',
                            data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'GW In', 'id': 'transfers_in_gw', 'type': 'numeric',
                                 'format': {'specifier': ','}},
                                {'name': 'GW Out', 'id': 'transfers_out_gw', 'type': 'numeric',
                                 'format': {'specifier': ','}},
                                {'name': 'Net', 'id': 'net_transfers_gw', 'type': 'numeric',
                                 'format': {'specifier': ','}},
                                {'name': 'In/Out', 'id': 'transfer_ratio', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Price Status', 'id': 'price_status'},
                                {'name': 'Progress %', 'id': 'price_progress', 'type': 'numeric',
                                 'format': {'specifier': '+.1f'}},
                                {'name': 'Predicted %', 'id': 'price_predicted', 'type': 'numeric',
                                 'format': {'specifier': '+.1f'}},
                                {'name': 'Own \u0394', 'id': 'own_delta_7d', 'type': 'numeric',
                                 'format': {'specifier': '+.2f'}},
                                {'name': '\u00a3 \u0394', 'id': 'price_delta_7d', 'type': 'numeric',
                                 'format': {'specifier': '+.1f'}},
                                {'name': 'Season +/-', 'id': 'cost_change_start', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{net_transfers_gw} > 0', 'column_id': 'net_transfers_gw'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{net_transfers_gw} < 0', 'column_id': 'net_transfers_gw'},
                                 'backgroundColor': '#fde8ef'},
                                # FPL-style status chips: green rise, red drop
                                {'if': {'filter_query': '{price_status} contains "rise"',
                                        'column_id': 'price_status'},
                                 'backgroundColor': '#00ff87', 'color': '#37003c', 'fontWeight': '700'},
                                {'if': {'filter_query': '{price_status} contains "drop"',
                                        'column_id': 'price_status'},
                                 'backgroundColor': '#e90052', 'color': '#ffffff', 'fontWeight': '700'},
                                {'if': {'filter_query': '{price_status} = "Unlikely to change"',
                                        'column_id': 'price_status'},
                                 'color': '#6b5c70'},
                                {'if': {'filter_query': '{price_predicted} >= 100',
                                        'column_id': 'price_predicted'}, 'backgroundColor': '#e6fff2', 'fontWeight': '700'},
                                {'if': {'filter_query': '{price_predicted} <= -100',
                                        'column_id': 'price_predicted'}, 'backgroundColor': '#fde8ef', 'fontWeight': '700'},
                                {'if': {'filter_query': '{price_progress} >= 100',
                                        'column_id': 'price_progress'}, 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{price_progress} <= -100',
                                        'column_id': 'price_progress'}, 'backgroundColor': '#fde8ef'},
                            ]
                        )
                    ], style=CARD_STYLE)
                ], style={'padding': '20px 0'})
            ]),

            # =================================================================
            # SQUAD BUILDER TAB
            # =================================================================
            # SQUAD BUILDER PAGE
            # MY SQUAD PAGE
            # MODEL LAB PAGE
            html.Div(id='page-model-lab', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Model Lab", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P(["Every refresh logs the projections AND the exact inputs behind them; every "
                                "finished gameweek logs what actually happened. This page scores the model "
                                "against reality, and the parameter sweep ", html.Strong("replays history under "
                                "different constants"), " to find the settings that would have predicted best, "
                                "converting judgement calls into measured ones."],
                               style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.H4("Backtest \u2014 Projected vs Actual",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(["Walks forward through completed gameweeks. For each one it "
                                "rebuilds every player's rates from their match history using "
                                "ONLY rounds before that gameweek, projects, then scores against "
                                "what actually happened. No stored logs needed \u2014 which is why "
                                "this works today and the calibration table above does not.",
                                html.Br(), html.Br(),
                                html.Strong("Read the baselines first. "),
                                "Most players score 1-3 points, so guessing a flat 2.2 for "
                                "everyone already scores about 2.0 MAE. If the model does not "
                                "beat 'just use his points per game', the modelling layer is not "
                                "earning its place. And since every decision this app makes is a "
                                "ranking decision, Spearman and top-20 matter more than MAE.",
                                html.Br(), html.Br(),
                                html.Em("First run fetches match history for every player and can "
                                        "take a minute. Availability flags are not archived by FPL, "
                                        "so everyone is treated as fit \u2014 absolute error reads "
                                        "slightly worse than live, equally for all models.")],
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Button('Run Backtest', id='lab-bt-run', n_clicks=0,
                                    style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                           'border': 'none', 'padding': '12px 24px',
                                           'borderRadius': '8px', 'fontWeight': '600',
                                           'cursor': 'pointer', 'marginBottom': '16px'}),
                        dcc.Loading(html.Div(id='lab-bt-out'), type='circle',
                                    color=COLORS['primary']),
                        # Full unfiltered per-player-per-GW rows from the last backtest
                        # run — lets the Diff/Mins filter controls re-slice instantly
                        # without re-running the (expensive) backtest itself.
                        dcc.Store(id='lab-player-rows-store')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Projection Breakdown", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(["A projection is eight components summed. Collapsed into one number they are "
                                "impossible to interrogate: an 8.8 built from a modest rate \u00d7 a 1.6 fixture "
                                "multiplier looks identical to an 8.5 rate barely scaled, and those need "
                                "opposite fixes. ", html.Strong("Neutral"), " is the projection at a "
                                "league-average fixture; ", html.Strong("Fix \u00d7"), " is the attacking "
                                "environment multiplier actually applied. If Proj is far above Neutral the "
                                "fixture model is driving it; if Neutral is already high, the player's own "
                                "rates are."],
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Div([
                            html.Div([
                                html.Label("Position", style={'fontWeight': '600', 'marginBottom': '6px',
                                                              'display': 'block'}),
                                dcc.Dropdown(id='lab-bd-position',
                                             options=[{'label': 'All', 'value': 'All'}] +
                                                     [{'label': x, 'value': x} for x in ['GKP', 'DEF', 'MID', 'FWD']],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Team", style={'fontWeight': '600', 'marginBottom': '6px',
                                                          'display': 'block'}),
                                dcc.Dropdown(id='lab-bd-team',
                                             options=[{'label': 'All', 'value': 'All'}] +
                                                     [{'label': t, 'value': t} for t in sorted_teams],
                                             value='All', clearable=False)
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Search player", style={'fontWeight': '600', 'marginBottom': '6px',
                                                                   'display': 'block'}),
                                dcc.Input(id='lab-bd-search', type='text', value='', debounce=True,
                                          placeholder='e.g. Gross',
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '150px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end',
                                  'marginBottom': '16px'}),
                        dcc.Graph(id='lab-bd-chart', config={'displayModeBar': False}),
                        html.Div(id='lab-bd-table')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Calibration by Gameweek", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.Div(id='lab-calibration')
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Parameter Sweep", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Grid-searches the fixture-response ratio and shrinkage strength over every "
                               "scoreable gameweek. Needs at least one completed GW with logged features; "
                               "results sharpen as gameweeks accumulate — re-run every few weeks.",
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        html.Button("Run Parameter Sweep", id='lab-sweep-btn', n_clicks=0,
                                    style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                           'border': 'none', 'padding': '10px 28px', 'borderRadius': '6px',
                                           'fontSize': '15px', 'fontWeight': '700', 'cursor': 'pointer'}),
                        dcc.Loading(html.Div(id='lab-sweep-result'), type='circle', color=COLORS['primary'])
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            # EXPECTED CLEAN SHEETS PAGE
            html.Div(id='page-xcs', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Expected Clean Sheets", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Projected clean sheets per team over your chosen horizon, from a Poisson model: ",
                            "each fixture's expected goals conceded comes from the opponent's attacking strength ",
                            "and your team's defensive strength, both blended with ",
                            html.Strong("actual recent results (last 6 games)"),
                            " so the numbers move with form, not just reputation. ",
                            "P(clean sheet) = e",
                            html.Sup("\u2212\u03bb"),
                            " per fixture; the horizon total simply sums them, so doubles count twice and blanks count zero."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div(id='xcs-form-note')
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Horizon (gameweeks)", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(
                                    id='xcs-horizon', min=1, max=10, step=1, value=5,
                                    marks={i: str(i) for i in range(1, 11)},
                                )
                            ], style={'flex': '0 1 420px', 'minWidth': '260px',
                                      'maxWidth': '420px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H3(id='xcs-chart-title', style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("Hover a bar for the fixture-by-fixture clean sheet probabilities.",
                               style={'color': COLORS['text_light']}),
                        dcc.Graph(id='xcs-bar', config={'displayModeBar': False})
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Team Clean Sheet Projections", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='xcs-team-table', data=[],
                            columns=[
                                {'name': 'Team', 'id': 'team'},
                                {'name': 'xCS', 'id': 'xcs', 'type': 'numeric'},
                                {'name': 'Avg CS% / match', 'id': 'avg_cs_prob', 'type': 'numeric'},
                                {'name': 'Fixtures', 'id': 'count', 'type': 'numeric'},
                                {'name': 'xGC/game (last 6)', 'id': 'recent_conceded', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Goals conc/game', 'id': 'recent_goals_conceded', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'CS (last 6)', 'id': 'recent_cs'},
                                {'name': 'Fixture-by-fixture CS%', 'id': 'fixture_string'},
                            ],
                            sort_action='native', page_size=20,
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{avg_cs_prob} >= 40', 'column_id': 'avg_cs_prob'},
                                 'backgroundColor': '#e6fff2'},
                                {'if': {'filter_query': '{avg_cs_prob} < 25', 'column_id': 'avg_cs_prob'},
                                 'backgroundColor': '#fde8ef'},
                            ]
                        )
                    ], style=CARD_STYLE),

                    html.Div([
                        html.H4("Defender & Keeper Picks by Expected Clean Sheets",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P("GKP and DEF ranked by their team's xCS over the horizon — cross-referenced with "
                               "price, projections and ownership so you can pick the cheapest reliable route "
                               "into a good defence.",
                               style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
                        dash_table.DataTable(
                            id='xcs-player-table', data=[],
                            columns=[
                                {'name': 'Player', 'id': 'web_name'},
                                {'name': 'Team', 'id': 'team_name'},
                                {'name': 'Pos', 'id': 'position'},
                                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                {'name': 'Team xCS', 'id': 'team_xcs', 'type': 'numeric'},
                                {'name': 'Proj Next 5', 'id': 'proj_pts_5', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'CS/90', 'id': 'cs_per_90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                {'name': 'Mins', 'id': 'minutes', 'type': 'numeric', 'format': {'specifier': ','}},
                                {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            ],
                            sort_action='native', page_size=20,
                            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}]
                        )
                    ], style=CARD_STYLE),
                ], style={'padding': '20px 0'})
            ]),

            # TRANSFER PLANNER PAGE
            html.Div(id='page-transfer-planner', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Transfer Gain Calculator", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "The most expensive habit in FPL is paying 4 points for transfers that don't return 4 points. ",
                            "This compares any two players on ", html.Strong("projected points over your chosen horizon"),
                            " and tells you the net gain of the move — including the hit, if you're taking one."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Player OUT", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='tp-player-out', options=[], placeholder='Search player to sell...',
                                             optionHeight=50)
                            ], style={'flex': '2', 'minWidth': '260px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Player IN", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(id='tp-player-in', options=[], placeholder='Search player to buy...',
                                             optionHeight=50)
                            ], style={'flex': '2', 'minWidth': '260px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Horizon", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.RadioItems(
                                    id='tp-horizon',
                                    options=[{'label': ' Next GW', 'value': 'next'},
                                             {'label': ' Next 5 GWs', 'value': 'five'}],
                                    value='five', inline=True,
                                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'}
                                )
                            ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Transfer cost", style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.RadioItems(
                                    id='tp-hit',
                                    options=[{'label': ' Free transfer', 'value': 0},
                                             {'label': ' -4 hit', 'value': 4}],
                                    value=0, inline=True,
                                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'}
                                )
                            ], style={'flex': '1', 'minWidth': '220px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    html.Div(id='tp-result')
                ], style={'padding': '20px 0'})
            ]),

            # CHIP PLANNER PAGE
            html.Div(id='page-chip-planner', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Chip Planner", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Chips are 30\u201360 points a season decided in a handful of choices \u2014 and the right week ",
                            "depends on ", html.Strong("your specific fifteen"), ", not the community consensus. ",
                            "This projects your actual squad gameweek by gameweek (DGW/BGW aware) and scores the best ",
                            "windows for ", html.Strong("Bench Boost"), " (bench projection), ",
                            html.Strong("Triple Captain"), " (best single-player week), and flags ",
                            html.Strong("Free Hit"), " candidates (blank-hit weeks)."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '16px'}),
                        html.Div([
                            html.Label("FPL Team ID",
                                       style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            html.Div([
                                dcc.Input(
                                    id='cp-team-id', type='number', placeholder='e.g. 1234567',
                                    style={'padding': '10px 14px', 'borderRadius': '6px',
                                           'border': f'2px solid {COLORS["primary"]}',
                                           'fontSize': '16px', 'width': '200px', 'marginRight': '12px'}
                                ),
                                dcc.Input(
                                    id='cp-league-id', type='number',
                                    placeholder='League ID (optional)',
                                    style={'padding': '10px 14px', 'borderRadius': '6px',
                                           'border': '1px solid #999',
                                           'fontSize': '16px', 'width': '200px', 'marginRight': '12px'}
                                ),
                                dcc.RadioItems(
                                    id='cp-horizon',
                                    options=[{'label': ' Next 10 GWs', 'value': 10},
                                             {'label': ' Full season roadmap', 'value': 38}],
                                    value=8, inline=True,
                                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'},
                                    style={'marginRight': '12px'}
                                ),
                                html.Button(
                                    "Analyse Chip Windows", id='cp-load-btn', n_clicks=0,
                                    style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                           'border': 'none', 'padding': '10px 28px', 'borderRadius': '6px',
                                           'fontSize': '15px', 'fontWeight': '700', 'cursor': 'pointer'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '8px'}),
                        ]),
                        html.P("Your team ID is in the URL on the FPL Points page: fantasy.premierleague.com/entry/XXXXXXX/…",
                               style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginTop': '10px'})
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    dcc.Loading(html.Div(id='cp-content'), type='circle', color=COLORS['primary'])
                ], style={'padding': '20px 0'})
            ]),

            # DEADLINE DASHBOARD PAGE
            html.Div(id='page-deadline', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Deadline Dashboard", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P(["One screen, one hour before the deadline. Most rank improvement is ",
                                html.Strong("consistency of process"), " — this enforces it: lineup vs optimal, "
                                "captain EV and ceiling, availability flags, and price risk, all in one pass."],
                               style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '16px'}),
                        html.Div([
                            dcc.Input(id='dd-team-id', type='number', placeholder='FPL Team ID',
                                      style={'padding': '10px 14px', 'borderRadius': '6px',
                                             'border': f'2px solid {COLORS["primary"]}',
                                             'fontSize': '16px', 'width': '180px', 'marginRight': '12px'}),
                            html.Button("Run Pre-Deadline Check", id='dd-load-btn', n_clicks=0,
                                        style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                               'border': 'none', 'padding': '10px 28px', 'borderRadius': '6px',
                                               'fontSize': '15px', 'fontWeight': '700', 'cursor': 'pointer'}),
                        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '8px'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),
                    dcc.Loading(html.Div(id='dd-content'), type='circle', color=COLORS['primary'])
                ], style={'padding': '20px 0'})
            ]),

            # MINI-LEAGUE RIVALS PAGE
            html.Div(id='page-rivals', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Mini-League Rivals", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "A mini-league isn't scored in points \u2014 it's scored in ", html.Strong("gaps"),
                            ". Players you share with a rival cancel out; only the differences move the table. ",
                            "This loads every squad in your league and shows the ", html.Strong("threats"),
                            " (they own, you don't), your ", html.Strong("leverage"), " (you own, they don't), ",
                            "everyone's captain, and the chips each rival has already burned."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '16px'}),
                        html.Div([
                            html.Div([
                                html.Label("League ID",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(
                                    id='rv-league-id', type='number', placeholder='e.g. 123456',
                                    style={'padding': '10px 14px', 'borderRadius': '6px',
                                           'border': f'2px solid {COLORS["primary"]}',
                                           'fontSize': '16px', 'width': '180px'}
                                ),
                            ], style={'marginRight': '16px'}),
                            html.Div([
                                html.Label("Your Team ID (optional)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(
                                    id='rv-my-id', type='number', placeholder='e.g. 1234567',
                                    style={'padding': '10px 14px', 'borderRadius': '6px',
                                           'border': f'2px solid {COLORS["primary"]}',
                                           'fontSize': '16px', 'width': '180px'}
                                ),
                            ], style={'marginRight': '16px'}),
                            html.Button(
                                "Load League", id='rv-load-btn', n_clicks=0,
                                style={'backgroundColor': COLORS['primary'], 'color': 'white',
                                       'border': 'none', 'padding': '10px 28px', 'borderRadius': '6px',
                                       'fontSize': '15px', 'fontWeight': '700', 'cursor': 'pointer',
                                       'alignSelf': 'flex-end', 'marginBottom': '2px'}
                            ),
                        ], style={'display': 'flex', 'alignItems': 'flex-end', 'flexWrap': 'wrap', 'gap': '8px'}),
                        html.P(["Your league ID is in the URL of the league standings page: ",
                                "fantasy.premierleague.com/leagues/", html.Strong("XXXXXX"), "/standings/c. ",
                                "Works for any classic league. Leagues bigger than 20 are capped at the top 20 by rank. ",
                                "Add your team ID to unlock the you-vs-them views."],
                               style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginTop': '10px'})
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    dcc.Loading(html.Div(id='rv-content'), type='circle', color=COLORS['primary'])
                ], style={'padding': '20px 0'})
            ]),

            html.Div(id='page-my-squad', style={'display': 'none'}, children=[
                html.Div([
                    html.Div([
                        html.H3("My Squad Analyser", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P(
                            "Enter your FPL team ID to see your current squad with form, fixture difficulty, "
                            "and injury flags.",
                            style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '16px'}
                        ),
                        html.Div([
                            html.Label("FPL Team ID",
                                       style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                            html.Div([
                                dcc.Input(
                                    id='squad-team-id-input',
                                    type='number',
                                    placeholder='e.g. 1234567',
                                    debounce=False,
                                    style={
                                        'padding': '10px 14px',
                                        'borderRadius': '6px',
                                        'border': f'2px solid {COLORS["primary"]}',
                                        'fontSize': '16px',
                                        'width': '200px',
                                        'marginRight': '12px',
                                    }
                                ),
                                html.Button(
                                    "Load My Squad",
                                    id='squad-load-btn',
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': COLORS['primary'],
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '10px 28px',
                                        'borderRadius': '6px',
                                        'fontSize': '15px',
                                        'fontWeight': '700',
                                        'cursor': 'pointer',
                                    }
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center',
                                      'flexWrap': 'wrap', 'gap': '8px'}),
                        ]),
                        html.P([
                            "Find your team ID in the FPL website URL: ",
                            html.Code(
                                "fantasy.premierleague.com/entry/{YOUR_ID}/event/...",
                                style={'backgroundColor': '#f0f0f0', 'padding': '2px 6px',
                                       'borderRadius': '4px', 'fontSize': '13px'}
                            )
                        ], style={'color': COLORS['text_light'], 'fontSize': '13px',
                                  'marginTop': '12px', 'marginBottom': '0'}),
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    dcc.Loading(
                        id='my-squad-loading',
                        type='circle',
                        color=COLORS['primary'],
                        children=[html.Div(
                            id='my-squad-content',
                            children=[html.P(
                                "Enter your team ID above to get started.",
                                style={'color': COLORS['text_light'], 'textAlign': 'center',
                                       'padding': '40px 0', 'fontSize': '15px'}
                            )]
                        )]
                    ),
                ], style={'padding': '20px 0'})
            ]),

            html.Div(id='page-squad-builder', style={'display': 'none'}, children=[
                html.Div([

                    # Explanation
                    html.Div([
                        html.H3("Budget Squad Optimiser", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                        html.P([
                            "Set your budget and objective, and the optimiser will find the ",
                            html.Strong("highest scoring 15 player squad"),
                            " that satisfies FPL's rules: ",
                            html.Strong("2 GKP · 5 DEF · 5 MID · 3 FWD · max 3 per club"), "."
                        ], style={'color': COLORS['text_dark'], 'fontSize': '15px', 'marginBottom': '12px'}),
                        html.Div([
                            html.Span("Click on 'Build Optimal Squad' once your parameters have been set",
                                      style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                             'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
                        ])
                    ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

                    # Controls
                    html.Div([
                        # Row 1: Budget, Objective, Min minutes
                        html.Div([
                            html.Div([
                                html.Label("Budget (£m)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Slider(
                                    id='sq-budget', min=75, max=105, step=0.1, value=100,
                                    marks={i: f'£{i}m' for i in range(75, 106, 5)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'flex': '3', 'minWidth': '280px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Optimise For",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='sq-objective',
                                    options=[
                                        {'label': 'Points Per Game (season average)', 'value': 'ppg'},
                                        {'label': 'Current Form (last 5 GWs)', 'value': 'form'},
                                        {'label': 'Expected Goal Involvements', 'value': 'expected_goal_involvements'},
                                        {'label': 'Total Points (season)', 'value': 'total_points'},
                                        {'label': 'Blended (PPG + Form + xGI)', 'value': 'blended'},
                                        {'label': 'Projected Points (next GW)', 'value': 'proj_pts_next'},
                                        {'label': 'Projected Points (next 5 GWs)', 'value': 'proj_pts_5'},
                                        {'label': 'Wildcard: Projected Points (next 8 GWs)', 'value': 'proj_pts_8'},
                                    ],
                                    value='ppg', clearable=False
                                )
                            ], style={'flex': '2', 'minWidth': '220px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Build toward chip GW (optional)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='sq-chip-gw', options=[], placeholder='None',
                                    clearable=True
                                )
                            ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),

                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end',
                                  'marginBottom': '20px'}),

                        # Row 2: Must include / Must exclude
                        html.Div([
                            html.Div([
                                html.Label("Must Include (pin players)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='sq-must-include',
                                    options=[
                                        {
                                            'label': f"{r['web_name']} ({r['team_name']}, {r['position']}, £{r['price']:.1f}m)",
                                            'value': int(r['id'])}
                                        for _, r in df_active.sort_values('web_name').iterrows()
                                    ],
                                    multi=True,
                                    placeholder='Search and select players to pin...',
                                )
                            ], style={'flex': '1', 'minWidth': '280px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Must Exclude (blacklist players)",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='sq-must-exclude',
                                    options=[
                                        {
                                            'label': f"{r['web_name']} ({r['team_name']}, {r['position']}, £{r['price']:.1f}m)",
                                            'value': int(r['id'])}
                                        for _, r in df_active.sort_values('web_name').iterrows()
                                    ],
                                    multi=True,
                                    placeholder='Search and select players to exclude...',
                                )
                            ], style={'flex': '1', 'minWidth': '280px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end',
                                  'marginBottom': '20px'}),

                        # Build button
                        html.Div([
                            html.Button(
                                "Build Optimal Squad",
                                id='sq-build-btn',
                                n_clicks=0,
                                style={
                                    'backgroundColor': COLORS['primary'],
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '12px 36px',
                                    'borderRadius': '8px',
                                    'fontSize': '16px',
                                    'fontWeight': '700',
                                    'cursor': 'pointer',
                                }
                            )
                        ], style={'padding': '0 10px'}),

                    ], style=CARD_STYLE),

                    # Results populated by callback
                    dcc.Loading(
                        id='sq-loading',
                        type='circle',
                        color=COLORS['primary'],
                        children=[html.Div(id='sq-results')]
                    ),

                ], style={'padding': '20px 0'})
            ]),

        ]),  # end content-area

    ], id='app-body'),  # end app-body (sidebar + content)

    # Footer
    html.Div([
        html.P(["Built for analytical Fantasy Premier League decision making  Data from ",
                html.A("Official FPL API", href="https://fantasy.premierleague.com/api/bootstrap-static/", target="_blank",
                       style={'color': COLORS['secondary']})],
               style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '13px', 'margin': '0'}),
        # Which commit is actually live — the quickest check that a deploy took
        html.P(f"Version {BUILD_ID[:7]}",
               style={'color': 'rgba(255,255,255,0.45)', 'fontSize': '11px', 'margin': '6px 0 0'}),
    ], style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'textAlign': 'center'})

], style={'fontFamily': FONT_FAMILY, 'backgroundColor': COLORS['background'], 'margin': '0', 'padding': '0'})


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output('stale-stats-banner', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def render_stale_stats_banner(_n):
    """
    Until the season starts, every cumulative counter in bootstrap-static
    (minutes, points, clean sheets, goals conceded, BPS, DEFCON...) still holds
    its FINAL 2025/26 value. FPL zeroes them around the GW1 deadline. Nothing in
    the payload flags this, so any tab built on season totals silently plots last
    season's numbers under this season's clubs. Say so rather than let the charts
    imply otherwise.
    """
    data = get_data()
    if data.get('season_started', True):
        return None

    next_gw = data.get('next_gw') or {}
    deadline = ''
    if next_gw.get('deadline_time'):
        try:
            deadline = datetime.fromisoformat(
                next_gw['deadline_time'].replace('Z', '+00:00')
            ).strftime('%d %b, %H:%M')
        except (TypeError, ValueError):
            deadline = ''

    return html.Div([
        html.Span("Pre-season", style={
            'backgroundColor': COLORS['warning'], 'color': '#3a2c00',
            'padding': '4px 12px', 'borderRadius': '20px',
            'fontSize': '12px', 'fontWeight': '700',
            'textTransform': 'uppercase', 'letterSpacing': '0.5px',
            'marginRight': '12px', 'whiteSpace': 'nowrap', 'flexShrink': '0'
        }),
        html.Span([
            "Player stats shown reflect the end of the ",
            html.Strong("2025/26 season"),
            ". FPL carries them over until the "
            f"{next_gw.get('name', 'Gameweek 1')} deadline"
            f"{f' ({deadline})' if deadline else ''}, when they reset to zero. "
            "Prices, clubs and fixtures are current whereas minutes, points, clean "
            "sheets and DEFCON are not. Transferred players still carry their "
            "old club's numbers, and promoted clubs are absent because they have "
            "no Premier League minutes yet."
        ], style={'color': COLORS['text_dark'], 'fontSize': '14px', 'lineHeight': '1.5'})
    ], style={
        'display': 'flex', 'alignItems': 'flex-start', 'flexWrap': 'wrap', 'gap': '4px',
        'backgroundColor': '#fff8e1', 'border': '1px solid #ffe082',
        'borderLeft': f"4px solid {COLORS['warning']}",
        'borderRadius': '8px', 'padding': '14px 18px', 'marginBottom': '20px'
    })


# All page values in order
ALL_PAGES = [
    'home', 'model-lab', 'defcon-bonus', 'bonus-consistency', 'defcon',
    'xg', 'underlying', 'value', 'form', 'cs',
    'fixture-ticker', 'fixture-outlook', 'fixtures', 'xcs', 'differentials',
    'captain', 'transfers', 'transfer-planner', 'chip-planner',
    'rivals', 'deadline', 'my-squad', 'squad-builder',
    'shot-profiles', 'matchups', 'chance-quality',
]

# -----------------------------------------------------------------------------
# NAVIGATION & SIDEBAR — all clientside.
# These used to be server callbacks, so every hamburger tap and nav click made
# a network round trip to Render (two, for the hamburger: toggle -> store ->
# classes) and had to queue behind whatever heavy callback a gunicorn worker
# was already running. None of this needs Python — it's pure show/hide — so
# it now runs in the browser and responds instantly. Page CONTENT callbacks
# that listen to active-page are unchanged and still run server-side.
# -----------------------------------------------------------------------------

def _nav_trigger_js():
    """Shared JS snippet: id of whatever triggered this clientside callback."""
    return """
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        var propId = trig ? trig.prop_id : '';
        var trigId = propId.substring(0, propId.lastIndexOf('.'));
        var trigVal = trig ? trig.value : null;
    """


# --- NAV: clicks → active-page store ---
clientside_callback(
    """
    function() {
        var args = Array.prototype.slice.call(arguments);
        var localVal = args[args.length - 1];
        var pages = %s;
        %s
        if (trigId === 'active-page-local') {
            return pages.indexOf(localVal) !== -1 ? localVal : 'home';
        }
        if (trigId.indexOf('nav-') === 0) {
            return trigId.substring(4);
        }
        return window.dash_clientside.no_update;
    }
    """ % (json.dumps(ALL_PAGES), _nav_trigger_js()),
    Output('active-page', 'data'),
    [Input(f'nav-{p}', 'n_clicks') for p in ALL_PAGES],
    Input('active-page-local', 'data'),
    prevent_initial_call=True
)


# --- Home spotlight / stat cards → jump to the full page ---
# Separate from the sidebar router because those Inputs are keyed on
# `nav-<page>` and reusing those ids would duplicate them. Pattern-matching
# ids plus allow_duplicate lets both write to the same store.
clientside_callback(
    """
    function(clicks) {
        var pages = %s;
        %s
        // Re-rendered Home cards fire with n_clicks 0/null — ignore those.
        if (!trigVal) { return window.dash_clientside.no_update; }
        try {
            var page = JSON.parse(trigId).page;
            return pages.indexOf(page) !== -1 ? page : window.dash_clientside.no_update;
        } catch (e) {
            return window.dash_clientside.no_update;
        }
    }
    """ % (json.dumps(ALL_PAGES), _nav_trigger_js()),
    Output('active-page', 'data', allow_duplicate=True),
    Input({'type': 'home-jump', 'page': ALL}, 'n_clicks'),
    prevent_initial_call=True
)


# --- Read saved page from localStorage on load ---
clientside_callback(
    """
    function() {
        var page = localStorage.getItem('fpl_active_page');
        return page || 'home';
    }
    """,
    Output('active-page-local', 'data'),
    Input('active-page-local', 'id'),
)

# --- Write active page to localStorage whenever it changes ---
clientside_callback(
    """
    function(page) {
        if (page) {
            localStorage.setItem('fpl_active_page', page);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('active-page', 'id'),
    Input('active-page', 'data'),
)


# --- PAGES: show active page, hide the rest; highlight its nav button ---
clientside_callback(
    """
    function(active) {
        var pages = %s;
        var styles = pages.map(function(p) {
            return p === active ? {'display': 'block'} : {'display': 'none'};
        });
        var classes = pages.map(function(p) {
            return p === active ? 'nav-item active' : 'nav-item';
        });
        return styles.concat(classes);
    }
    """ % json.dumps(ALL_PAGES),
    [Output(f'page-{p}', 'style') for p in ALL_PAGES]
    + [Output(f'nav-{p}', 'className') for p in ALL_PAGES],
    Input('active-page', 'data')
)


# --- LAZY PAGES: only render a page when it's opened (or its data changed) ---
# Previously every page's callbacks fired on load and every 2 minutes whether
# visible or not, so the page you were actually looking at queued behind ~20
# hidden ones on Render's single worker. Now each page renders the first time
# it's opened, and again only if the data version has moved since — revisiting
# an up-to-date page costs nothing, because its content is still in the DOM.
clientside_callback(
    """
    function(active, version, rendered) {
        var lazy = %s;
        var noUp = window.dash_clientside.no_update;
        var out = lazy.map(function() { return noUp; });
        var idx = lazy.indexOf(active);
        rendered = rendered || {};
        if (!version || idx === -1 || rendered[active] === version) {
            out.push(noUp);
            return out;
        }
        var next = Object.assign({}, rendered);
        next[active] = version;
        out[idx] = version;
        out.push(next);
        return out;
    }
    """ % json.dumps(LAZY_PAGES),
    [Output(f'visit-{p}', 'data') for p in LAZY_PAGES] + [Output('page-rendered', 'data')],
    [Input('active-page', 'data'), Input('data-version', 'data')],
    State('page-rendered', 'data'),
)


# --- Home-screen app: full reload (fresh data and any newly deployed code) ---
clientside_callback(
    """
    function(n) {
        if (n) { window.location.reload(); }
        return window.dash_clientside.no_update;
    }
    """,
    Output('app-refresh-btn', 'title'),
    Input('app-refresh-btn', 'n_clicks'),
    prevent_initial_call=True
)


# --- SIDEBAR: toggle open/closed on mobile, in one step ---
clientside_callback(
    """
    function(hamburgerClicks, overlayClicks, activePage, isOpen) {
        %s
        var open = (trigId === 'hamburger-btn') ? !isOpen : false;
        return [open,
                open ? 'sidebar sidebar-open' : 'sidebar',
                open ? 'overlay-open' : ''];
    }
    """ % _nav_trigger_js(),
    [Output('sidebar-open', 'data'),
     Output('sidebar', 'className'),
     Output('sidebar-overlay', 'className')],
    [Input('hamburger-btn', 'n_clicks'),
     Input('sidebar-overlay', 'n_clicks'),
     Input('active-page', 'data')],
    State('sidebar-open', 'data'),
    prevent_initial_call=True
)



@callback(
    [Output('gw-status-text', 'children'), Output('last-updated-text', 'children'),
     Output('data-version', 'data')],
    Input('refresh-interval', 'n_intervals')
)
def update_refresh_status(n):
    gw_text, status_text = _refresh_status_text(n)
    last = DATA.get('last_refresh', 0)
    # Changes only when the data does: a full refresh, or the heavy stats
    # phase finishing after startup. Drives lazy page re-renders.
    version = (f"{int(last)}-{int(bool(DATA.get('heavy_loaded', False)))}"
               f"-{DATA.get('shots_version', 0)}-{int(DATA.get('price_tracker_at') or 0)}") if last else None
    return gw_text, status_text, version


def _refresh_status_text(n):
    """Check data freshness every 2 minutes. Trigger background refresh if stale."""
    check_and_refresh()
    current_gw_now = DATA.get('current_gw')
    if current_gw_now:
        gw_text = [html.Span(f"Data as of {current_gw_now['name']}", className='hdr-long'),
                   html.Span(f"Data as of GW{current_gw_now['id']}", className='hdr-short')]
    else:
        gw_text = "N/A"
    last = DATA.get('last_refresh', 0)
    if last > 0:
        refresh_time = datetime.fromtimestamp(last, tz=ZoneInfo('Europe/London')).strftime('%H:%M')
        age_mins = int((time.time() - last) / 60)
        if DATA.get('refreshing', False):
            return gw_text, "Refreshing data..."
        if not DATA.get('heavy_loaded', False):
            return gw_text, f"Updated {refresh_time} ({age_mins}m ago) · Loading detailed stats..."
        return gw_text, f"Updated {refresh_time} ({age_mins}m ago)"
    return gw_text, "Loading..."


def filter_data(position, team, max_price, min_minutes, positions_allowed=None):
    # Dash number inputs deliver None whenever the typed value is empty or
    # violates min/max/step — comparing a column against None raises and
    # silently kills the whole page. Coerce every input to a safe value.
    min_minutes = 0 if min_minutes is None else min_minutes
    max_price = 20 if max_price is None else max_price
    position = position or 'All'
    team = team or 'All'

    data = get_data()
    filtered = data['df_active'].copy()
    if positions_allowed:
        filtered = filtered[filtered['position'].isin(positions_allowed)]
    if position != 'All':
        filtered = filtered[filtered['position'] == position]
    if team != 'All':
        filtered = filtered[filtered['team_name'] == team]
    filtered = filtered[filtered['price'] <= max_price]
    filtered = filtered[filtered['minutes'] >= min_minutes]
    return filtered


# HOME TAB (dynamic - rebuilds from fresh DATA on each interval tick)
@callback(
    Output('home-content', 'children'),
    Input('visit-home', 'data'),
    prevent_initial_call=True
)
def update_home_tab(n):
    """Rebuild the entire Home tab from current DATA so it reflects refreshed data."""
    _need_visit(n)
    data = get_data()
    df_now = data.get('df_active', pd.DataFrame())
    current_gw_now = data.get('current_gw')
    next_gw_now = data.get('next_gw')
    total_mgrs = data.get('total_managers', 0)

    # Both are null until a gameweek has actually been scored — `or 0` is unsafe
    # for NaN but these are plain ints/None from JSON, so an explicit check is fine.
    avg_gw = (current_gw_now or {}).get('average_entry_score')
    highest_gw = (current_gw_now or {}).get('highest_score')
    avg_gw = 0 if avg_gw is None else avg_gw
    highest_gw = 0 if highest_gw is None else highest_gw

    # Top players
    top_scorer_now = df_now.nlargest(1, 'total_points').iloc[0] if len(df_now) > 0 else None
    most_selected_now = df_now.nlargest(1, 'ownership').iloc[0] if len(df_now) > 0 else None
    _home_thr = adaptive_min_minutes(450, current_gw_now['id'] if current_gw_now else 0)
    best_value_now = df_now[df_now['minutes'] >= _home_thr].nlargest(1, 'points_per_million').iloc[0] if len(
        df_now[df_now['minutes'] >= _home_thr]) > 0 else None
    top_form_now = df_now.nlargest(1, 'form').iloc[0] if len(df_now) > 0 else None

    # Most captained
    most_cap_id = current_gw_now.get('most_captained') if current_gw_now else None
    most_cap = None
    if most_cap_id:
        match = df_now[df_now['id'] == most_cap_id]
        if len(match) > 0:
            most_cap = match.iloc[0]

    # Most vice-captained
    most_vice_id = current_gw_now.get('most_vice_captained') if current_gw_now else None
    most_vice = None
    if most_vice_id:
        match = df_now[df_now['id'] == most_vice_id]
        if len(match) > 0:
            most_vice = match.iloc[0]

    # Chip usage — THIS gameweek (for the weekly card and bar chart)
    chips = current_gw_now.get('chip_plays', []) if current_gw_now else []
    chip_sum = ', '.join(
        [f"{chip_name_map.get(c['chip_name'], c['chip_name'])}: {c['num_played']:,}" for c in chips]
    ) if chips else "No data yet"
    total_chips = sum(c['num_played'] for c in chips) if chips else 0

    # Chip usage — SEASON TO DATE, summed across every gameweek. The card
    # above is a weekly snapshot; this is how much of the game has spent each
    # chip so far, which is the number people actually mean.
    season_chips = summarise_chip_usage(data.get('bootstrap_data'))

    # Chip bar chart
    chip_colors = {
        'Bench Boost': COLORS['info'],
        'Triple Captain': COLORS['accent'],
        'Wildcard': COLORS['success'],
        'Free Hit': COLORS['warning'],
    }
    if chips:
        c_names = [chip_name_map.get(c['chip_name'], c['chip_name']) for c in chips]
        c_counts = [c['num_played'] for c in chips]
        sorted_pairs = sorted(zip(c_names, c_counts), key=lambda x: x[1], reverse=True)
        c_names = [p[0] for p in sorted_pairs]
        c_counts = [p[1] for p in sorted_pairs]
        c_colors = [chip_colors.get(n, COLORS['primary']) for n in c_names]
        chip_fig = go.Figure()
        chip_fig.add_trace(go.Bar(x=c_names, y=c_counts, marker_color=c_colors,
                                  text=[f"{c:,}" for c in c_counts], textposition='outside'))
        chip_fig.update_layout(template='plotly_white', height=300,
                               margin=dict(t=40, b=40, l=40, r=40),
                               yaxis_title='Managers', showlegend=False,
                               font=dict(family=FONT_FAMILY),
                               yaxis=dict(range=[0, max(c_counts) * 1.15]))
    else:
        chip_fig = go.Figure()
        chip_fig.add_annotation(text="No chip data available yet", xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=COLORS['text_light']))
        chip_fig.update_layout(template='plotly_white', height=300)

    # Position breakdown chart
    pos_data = df_now[df_now['minutes'] >= _home_thr]
    if len(pos_data) > 0:
        position_stats = pos_data.groupby('position').agg({
            'total_points': 'mean', 'points_per_million': 'mean', 'price': 'mean'
        }).round(2).reset_index()
        position_order = ['GKP', 'DEF', 'MID', 'FWD']
        position_stats['position'] = pd.Categorical(position_stats['position'], categories=position_order, ordered=True)
        position_stats = position_stats.sort_values('position')
        pos_fig = go.Figure()
        pos_fig.add_trace(go.Bar(name='Avg Points', x=position_stats['position'], y=position_stats['total_points'],
                                 marker_color=COLORS['primary'], text=position_stats['total_points'].round(1),
                                 textposition='outside'))
        pos_fig.add_trace(
            go.Bar(name='Avg Pts/m (x10)', x=position_stats['position'], y=position_stats['points_per_million'] * 10,
                   marker_color=COLORS['secondary'], text=position_stats['points_per_million'].round(2),
                   textposition='outside'))
        pos_fig.update_layout(barmode='group', template='plotly_white', height=350,
                              margin=dict(t=60, b=40, l=40, r=40),
                              legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                              yaxis_title='Value', xaxis_title='Position', font=dict(family=FONT_FAMILY),
                              yaxis=dict(range=[0, position_stats['points_per_million'].max() * 10 * 1.2]))
    else:
        pos_fig = go.Figure()

    # Model calibration card (populated once a logged GW has completed)
    cal = data.get('calibration')
    if cal and cal.get('mae_model') is not None:
        beats = cal.get('mae_fpl') is not None and cal['mae_model'] < cal['mae_fpl']
        cal_card = html.Div([
            html.H3("Projection Model Calibration", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P([
                f"Over {cal['gws']} scored gameweek(s) ({cal['n']:,} player-predictions): this model's mean "
                f"absolute error is ", html.Strong(f"{cal['mae_model']:.2f} pts"),
                f" vs FPL's own xP at ", html.Strong(f"{cal['mae_fpl']:.2f} pts" if cal['mae_fpl'] else 'n/a'),
                ". " + ("The model is currently beating FPL's projections — trust the Proj Pts columns."
                        if beats else
                        "FPL's xP is currently ahead — treat Proj Pts as a second opinion, not gospel."),
            ], style={'color': COLORS['text_dark'], 'margin': 0}),
        ], style={**CARD_STYLE, 'backgroundColor': '#f0faf4' if beats else '#fdf6ec'})
    else:
        cal_card = html.Div()

    # Build and return layout
    return html.Div([
        cal_card,
        html.Div([
            html.H2("Season Overview", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P(f"Key statistics from the {data.get('season_label', SEASON['label'])} FPL season",
                   style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([
            html.Div([build_stat_card("Total Managers", f"{total_mgrs:,}", "Competing worldwide")],
                     style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card("Current Gameweek",
                                      current_gw_now['name'].replace('Gameweek ', 'GW') if current_gw_now else "N/A",
                                      f"Average: {avg_gw} pts")],
                     style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card("Highest GW Score", f"{highest_gw}", "This gameweek", color=COLORS['success_text'])],
                     style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([
                build_stat_card(
                    "Next Deadline",
                    next_gw_now['name'].replace('Gameweek ', 'GW') if next_gw_now else "N/A",
                    datetime.fromisoformat(
                        next_gw_now['deadline_time'].replace('Z', '+00:00')
                    ).astimezone(
                        ZoneInfo('Europe/London')
                    ).strftime('%a %d %b, %H:%M') if next_gw_now else "",
                    link_page='deadline',
                    link_label='Deadline dashboard'
                )
            ],
                style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}),

        html.Div([
            html.Div([build_stat_card(
                "Most Captained",
                most_cap['web_name'] if most_cap is not None else "N/A",
                f"{most_cap['team_name']} - £{most_cap['price']:.1f}m" if most_cap is not None else "Data available after deadline",
                color=COLORS['primary'],
                image_code=most_cap if most_cap is not None else None,
                link_page='captain', link_label='Captain optimiser'
            )], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card(
                "Most Vice-Captained",
                most_vice['web_name'] if most_vice is not None else "N/A",
                f"{most_vice['team_name']} - £{most_vice['price']:.1f}m" if most_vice is not None else "Data available after deadline",
                color=COLORS['accent'],
                image_code=most_vice if most_vice is not None else None,
                link_page='captain', link_label='Captain optimiser'
            )], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card(
                "Chips Used This GW",
                f"{total_chips:,}" if total_chips > 0 else "N/A",
                chip_sum,
                color=COLORS['info_text']
            ,
                link_page='chip-planner', link_label='Chip planner')], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}),

        html.Div([
            html.H2("Chip Usage This Gameweek", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Number of managers activating each chip", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([dcc.Graph(figure=chip_fig, config={'displayModeBar': False})], style=CARD_STYLE),

        html.Div([
            html.H2("How Much of the Game Has Used Each Chip",
                    style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P(f"Season to date across all {season_chips['total_players']:,} teams, "
                   f"summed over {season_chips['gws']} gameweeks. Two chip sets per season "
                   f"means a share can exceed the share of managers \u2014 read it as plays "
                   f"per team.", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}) if season_chips['rows'] else html.Div(),

        html.Div([
            html.Div([
                build_stat_card(r['chip'], f"{r['pct']:.1f}%", f"{r['played']:,} played",
                                color=COLORS['accent'])
            ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'})
            for r in season_chips['rows']
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}
        ) if season_chips['rows'] else html.Div(),

        html.Div([
            html.H2("Player Spotlights", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Top performers across key metrics", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([
            build_player_spotlight(top_scorer_now, "Top Scorer", "Total Points",
                                   f"{int(top_scorer_now['total_points'])}" if top_scorer_now is not None else "N/A",
                                   link_page='value', link_label='Points table'),
            build_player_spotlight(most_selected_now, "Most Selected", "Ownership",
                                   f"{most_selected_now['ownership']:.1f}%" if most_selected_now is not None else "N/A",
                                   link_page='differentials', link_label='Ownership table'),
            build_player_spotlight(best_value_now, "Best Value", "Points/£m",
                                   f"{best_value_now['points_per_million']:.2f}" if best_value_now is not None else "N/A",
                                   link_page='value', link_label='Value analysis'),
            build_player_spotlight(top_form_now, "In Form", "Form Rating",
                                   f"{top_form_now['form']:.1f}" if top_form_now is not None else "N/A",
                                   link_page='form', link_label='Form tracker'),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '40px'}),

        html.Div([
            html.H2("Position Breakdown", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Average points and value by position", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([dcc.Graph(figure=pos_fig, config={'displayModeBar': False})], style=CARD_STYLE),

    ], style={'padding': '20px 0'})


# RANK CONGESTION TOOL
def get_congestion_divisor(rank_gap: int) -> int:
    """Pick a scale so 'pts per N ranks' stays in a readable range."""
    if rank_gap < 1_000:
        return 100
    elif rank_gap < 10_000:
        return 1_000
    elif rank_gap < 100_000:
        return 10_000
    else:
        return 100_000


@callback(
    Output('rank-result', 'children'),
    Input('rank-check-btn', 'n_clicks'),
    State('rank-your', 'value'),
    State('rank-rival', 'value'),
    prevent_initial_call=True
)
def check_rank_gap(n_clicks, your_rank, rival_rank):
    if not your_rank or not rival_rank:
        return html.P("Please enter both ranks.", style={'color': COLORS['danger_text']})

    def fetch_rank_points(rank):
        try:
            page = ((rank - 1) // 50) + 1
            url = f"https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings={page}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            standings = data.get('standings', {}).get('results', [])
            for entry in standings:
                if entry.get('rank') == rank:
                    return entry.get('total', None), entry.get('entry_name', f'Rank {rank}')
            if standings:
                closest = min(standings, key=lambda x: abs(x.get('rank', 0) - rank))
                return closest.get('total', None), closest.get('entry_name', f'Rank {rank}')
            return None, None
        except Exception as e:
            print(f"Error fetching rank {rank}: {e}")
            return None, None

    your_points, your_name = fetch_rank_points(your_rank)
    rival_points, rival_name = fetch_rank_points(rival_rank)

    if your_points is None or rival_points is None:
        return html.P("Could not fetch rank data. Please try again.",
                      style={'color': COLORS['danger_text'], 'marginTop': '12px'})

    gap = abs(rival_points - your_points)
    rank_gap = abs(your_rank - rival_rank)
    if rank_gap > 0:
        congestion_divisor = get_congestion_divisor(rank_gap)
        pts_per_x_ranks = round((gap / rank_gap) * congestion_divisor, 1)
    else:
        congestion_divisor = 0
        pts_per_x_ranks = 0
    higher_rank = rival_rank if rival_rank < your_rank else your_rank
    lower_rank = your_rank if rival_rank < your_rank else rival_rank
    higher_pts = rival_points if rival_rank < your_rank else your_points
    lower_pts = your_points if rival_rank < your_rank else rival_points

    return html.Div([
        html.Div([
            html.Div([
                html.P("Your Points", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                             'marginBottom': '4px', 'textTransform': 'uppercase',
                                             'letterSpacing': '0.5px', 'fontWeight': '600'}),
                html.H3(f"{your_points:,}", style={'color': COLORS['primary'], 'margin': '0',
                                                    'fontSize': '28px', 'fontWeight': '700'}),
                html.P(f"Rank {your_rank:,}", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                                      'margin': '4px 0 0 0'}),
            ], style={**STAT_CARD_STYLE, 'flex': '1', 'minWidth': '160px', 'minHeight': 'auto', 'padding': '16px'}),

            html.Div([
                html.P("Points Gap", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                            'marginBottom': '4px', 'textTransform': 'uppercase',
                                            'letterSpacing': '0.5px', 'fontWeight': '600'}),
                html.H3(f"{gap:,} pts", style={'color': COLORS['accent'], 'margin': '0',
                                                'fontSize': '28px', 'fontWeight': '700'}),
                html.P(f"across {rank_gap:,} rank places", style={'color': COLORS['text_light'],
                                                                    'fontSize': '13px', 'margin': '4px 0 0 0'}),
            ], style={**STAT_CARD_STYLE, 'flex': '1', 'minWidth': '160px', 'minHeight': 'auto', 'padding': '16px'}),

            html.Div([
                html.P("Congestion", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                            'marginBottom': '4px', 'textTransform': 'uppercase',
                                            'letterSpacing': '0.5px', 'fontWeight': '600'}),
                html.H3(f"{pts_per_x_ranks} pts", style={'color': COLORS['success_text'], 'margin': '0',
                                                          'fontSize': '28px', 'fontWeight': '700'}),
                html.P(f"per {congestion_divisor:,} rank places", style={'color': COLORS['text_light'],
                                                                          'fontSize': '13px', 'margin': '4px 0 0 0'}),
            ], style={**STAT_CARD_STYLE, 'flex': '1', 'minWidth': '160px', 'minHeight': 'auto', 'padding': '16px'}),

            html.Div([
                html.P("Rival Points", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                              'marginBottom': '4px', 'textTransform': 'uppercase',
                                              'letterSpacing': '0.5px', 'fontWeight': '600'}),
                html.H3(f"{rival_points:,}", style={'color': COLORS['primary'], 'margin': '0',
                                                     'fontSize': '28px', 'fontWeight': '700'}),
                html.P(f"Rank {rival_rank:,}", style={'color': COLORS['text_light'], 'fontSize': '13px',
                                                       'margin': '4px 0 0 0'}),
            ], style={**STAT_CARD_STYLE, 'flex': '1', 'minWidth': '160px', 'minHeight': 'auto', 'padding': '16px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px', 'marginBottom': '20px'}),
    ])


# DEFCON BONUS
@callback(
    [Output('bonus-scatter', 'figure'), Output('bonus-bar', 'figure'), Output('bonus-table', 'data')],
    [Input('bonus-position', 'value'), Input('bonus-team', 'value'), Input('bonus-price', 'value'),
     Input('bonus-minutes', 'value'), Input('visit-defcon-bonus', 'data')],
    prevent_initial_call=True
)
def update_bonus(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=SEASON['defcon_positions'])
    filtered = filtered.dropna(subset=['defcon_per_90'])

    scatter_fig = px.scatter(filtered, x='price', y='defcon_per_90', color='position', size='minutes',
                             hover_name='web_name', hover_data=['team_name', 'defcon', 'defcon_vs_bonus'],
                             color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                                 'FWD': COLORS['info']})
    scatter_fig.add_hline(y=SEASON['thresholds'].get('DEF', 10), line_dash="dash", line_color=COLORS['primary'],
                          annotation_text="DEF Threshold (10)", annotation_position="top right")
    scatter_fig.add_hline(y=SEASON['thresholds'].get('MID', 12), line_dash="dash", line_color=COLORS['accent'],
                          annotation_text="MID/FWD Threshold (12)", annotation_position="bottom right")
    scatter_fig.update_layout(template='plotly_white', height=400, xaxis_title='Price (£m)',
                              yaxis_title='Defcon per 90',
                              font=dict(family=FONT_FAMILY))

    top_25 = filtered.nlargest(25, 'defcon_vs_bonus')
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_25['web_name'], y=top_25['defcon_vs_bonus'],
        marker_color=[COLORS['success'] if x >= 0 else COLORS['danger'] for x in top_25['defcon_vs_bonus']],
        text=top_25['defcon_vs_bonus'].round(2), textposition='outside'
    ))
    bar_fig.add_hline(y=0, line_color='#333', line_width=2)
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Distance from Threshold', showlegend=False,
                          yaxis=dict(range=[top_25['defcon_vs_bonus'].min() * 1.2, top_25['defcon_vs_bonus'].max() * 1.2]),
                          font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'defcon_vs_bonus',
            'bonus_rate', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'defcon_vs_bonus'), cols)

    return scatter_fig, bar_fig, table_data


# BONUS CONSISTENCY
@callback(
    [Output('consistency-bar', 'figure'), Output('consistency-scatter', 'figure'), Output('consistency-table', 'data')],
    [Input('consistency-position', 'value'), Input('consistency-team', 'value'), Input('consistency-price', 'value'),
     Input('consistency-games', 'value'), Input('consistency-minutes', 'value'),
     Input('visit-bonus-consistency', 'data')],
    prevent_initial_call=True
)
def update_consistency(position, team, max_price, min_games, min_minutes, _n):
    _need_visit(_n)
    # Guard invalid/blank inputs (Dash sends None for out-of-step values)
    min_games = 1 if min_games is None else min_games
    min_minutes = 0 if min_minutes is None else min_minutes
    max_price = 20 if max_price is None else max_price
    # Filter to players with consistency data
    data = get_data()
    dfa = data['df_active']
    filtered = dfa[dfa['qualifying_games'].notna()].copy()

    # GW1 fallback: with exactly one gameweek played, season totals ARE the
    # single match — reconstruct per-match consistency exactly from
    # aggregates while the history fetch is still loading in Phase 2.
    if len(filtered) == 0:
        cur = data.get('current_gw')
        if cur and cur.get('id') == 1 and len(dfa) > 0:
            synth = dfa[(dfa['minutes'] >= 60) & (dfa['minutes'] <= 90)].copy()
            if len(synth) > 0:
                synth['qualifying_games'] = 1
                synth['bonus_games'] = (synth['defcon'] >= synth['bonus_threshold']).astype(int)
                synth['hit_rate'] = synth['bonus_games'] * 100.0
                synth['avg_defcon_qualifying'] = synth['defcon'].astype(float)
                synth['max_defcon_game'] = synth['defcon']
                synth['min_defcon_game'] = synth['defcon']
                filtered = synth

    # If Phase 2 hasn't loaded yet, show loading message
    if len(filtered) == 0:
        loading_fig = go.Figure()
        loading_fig.add_annotation(text="Player history data is still loading. Please wait a few minutes and refresh.",
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=14, color=COLORS['text_light']))
        loading_fig.update_layout(template='plotly_white', height=400)
        return loading_fig, loading_fig, []

    # Apply position filter
    if position != 'All':
        filtered = filtered[filtered['position'] == position]
    else:
        filtered = filtered[filtered['position'].isin(SEASON['defcon_positions'])]

    # Apply team filter
    if team != 'All':
        filtered = filtered[filtered['team_name'] == team]

    # Apply price filter
    filtered = filtered[filtered['price'] <= max_price]

    # Apply Min. minutes filter
    filtered = filtered[filtered['minutes'] >= min_minutes]

    # Apply Min. games filter
    filtered = filtered[filtered['qualifying_games'] >= min_games]

    # Bar chart - Top 25 by hit rate
    top_25 = filtered.nlargest(25, 'hit_rate')

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_25['web_name'],
        y=top_25['hit_rate'],
        marker_color=[COLORS['success'] if x >= 50 else (COLORS['warning'] if x >= 25 else COLORS['danger']) for x in
                      top_25['hit_rate']],
        text=[f"{x:.0f}%" for x in top_25['hit_rate']],
        textposition='outside',
        hovertemplate='%{x}<br>%{customdata[2]}  %{customdata[3]}<br>Hit Rate: %{y:.1f}%<br>Bonus Games: %{customdata[0]}/%{customdata[1]}<extra></extra>',
        customdata=top_25[['bonus_games', 'qualifying_games', 'position', 'team_name']].values
    ))
    bar_fig.add_hline(y=50, line_dash="dash", line_color=COLORS['warning'], annotation_text="50% threshold",
                      annotation_position="right")
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Bonus Hit Rate (%)', showlegend=False,
                          yaxis=dict(range=[0, max(top_25['hit_rate'].max() * 1.15, 55) if len(top_25) > 0 else 100]),
                          font=dict(family=FONT_FAMILY))

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
    _def_thr = SEASON['thresholds'].get('DEF', 10)
    _mid_thr = SEASON['thresholds'].get('MID', 12)
    scatter_fig.add_vline(x=_def_thr, line_dash="dash", line_color=COLORS['primary'],
                          annotation_text=f"DEF threshold ({_def_thr})")
    scatter_fig.add_vline(x=_mid_thr, line_dash="dash", line_color=COLORS['accent'],
                          annotation_text=f"MID/FWD threshold ({_mid_thr})")
    scatter_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title='Avg Defcon (in 60+ Min. games)',
        yaxis_title='Bonus Hit Rate (%)',
        font=dict(family=FONT_FAMILY)
    )

    # Table data
    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'qualifying_games', 'bonus_games', 'hit_rate',
            'starts_60', 'avg_defcon_qualifying', 'max_defcon_game', 'min_defcon_game', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'hit_rate'), cols)

    return bar_fig, scatter_fig, table_data


# DEFCON
@callback(
    [Output('defcon-scatter', 'figure'), Output('defcon-table', 'data')],
    [Input('defcon-position', 'value'), Input('defcon-team', 'value'), Input('defcon-price', 'value'),
     Input('defcon-minutes', 'value'), Input('visit-defcon', 'data')],
    prevent_initial_call=True
)
def update_defcon(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=SEASON['defcon_positions'])
    filtered = filtered.dropna(subset=['defcon_per_90', 'expected_defcon'])

    fig = px.scatter(filtered, x='expected_defcon', y='defcon', color='position', size='minutes',
                     hover_name='web_name', hover_data=['team_name', 'price', 'defcon_per_90'],
                     labels={'expected_defcon': 'Expected Def Con', 'defcon': 'Def Cons'},
                     color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['defcon'].max(), filtered['expected_defcon'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='#999'),
                                 name='Expected'))
    fig.update_layout(template='plotly_white', height=400, font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'expected_defcon',
            'defcon_diff', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'defcon_per_90'), cols)

    return fig, table_data


# XG
@callback(
    [Output('xg-scatter', 'figure'), Output('xg-table', 'data')],
    [Input('xg-position', 'value'), Input('xg-team', 'value'), Input('xg-price', 'value'), Input('xg-minutes', 'value'), Input('visit-xg', 'data')],
    prevent_initial_call=True
)
def update_xg(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['expected_goals'])

    fig = px.scatter(filtered, x='expected_goals', y='goals_scored', color='position', size='minutes',
                     hover_name='web_name', hover_data=['team_name', 'price', 'xg_diff'],
                     labels={'expected_goals': 'Expected Goals', 'goals_scored': 'Goals Scored'},
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                         'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['goals_scored'].max(), filtered['expected_goals'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='#999'),
                                 name='Expected'))
    fig.update_layout(template='plotly_white', height=400, font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'goals_scored', 'expected_goals', 'xg_diff', 'assists',
            'expected_assists', 'xa_diff', 'ownership']
    table_data = prepare_table_data(filtered.sort_values('xg_diff').head(50), cols)

    return fig, table_data


# UNDERLYING NUMBERS
@callback(
    [Output('under-scatter', 'figure'), Output('under-table', 'data')],
    [Input('under-position', 'value'), Input('under-team', 'value'), Input('under-price', 'value'),
     Input('under-minutes', 'value'), Input('visit-underlying', 'data')],
    prevent_initial_call=True
)
def update_underlying(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['xgi_per_90', 'gi_per_90'])

    # Scatter — actual GI/90 vs xGI/90
    fig = px.scatter(filtered, x='xgi_per_90', y='gi_per_90', color='position', size='minutes',
                     hover_name='web_name',
                     hover_data=['team_name', 'price', 'xgi_diff_per_90', 'threat_per_90', 'creativity_per_90'],
                     labels={'xgi_per_90': 'Expected GI per 90', 'gi_per_90': 'Actual GI per 90'},
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['xgi_per_90'].max(), filtered['gi_per_90'].max(), 0.5)
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                 line=dict(dash='dash', color='#999'), name='Expected'))
    fig.update_layout(template='plotly_white', height=400, font=dict(family=FONT_FAMILY))

    # Table — sorted by xGI/90 descending
    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'xgi_per_90', 'gi_per_90',
            'xgi_diff_per_90', 'xg_per_90', 'xa_per_90', 'threat_per_90', 'creativity_per_90',
            'ict_per_90', 'bps_per_90', 'form', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'xgi_per_90'), cols)

    return fig, table_data


# VALUE
@callback(
    [Output('value-scatter', 'figure'), Output('value-table', 'data')],
    [Input('value-position', 'value'), Input('value-team', 'value'), Input('value-price', 'value'),
     Input('value-minutes', 'value'), Input('visit-value', 'data')],
    prevent_initial_call=True
)
def update_value(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['points_per_million'])

    fig = px.scatter(filtered, x='price', y='total_points', color='position', size='ownership',
                     hover_name='web_name', hover_data=['team_name', 'points_per_million', 'form'],
                     labels={'price': 'Price', 'total_points': 'Total Points'},
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                         'FWD': COLORS['info']})
    fig.update_layout(template='plotly_white', height=400, font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'points_per_million', 'form', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'points_per_million'), cols)

    return fig, table_data


# FORM
@callback(
    [Output('form-chart', 'figure'), Output('form-table', 'data')],
    [Input('form-position', 'value'), Input('form-team', 'value'), Input('form-price', 'value'),
     Input('form-minutes', 'value'), Input('visit-form', 'data')],
    prevent_initial_call=True
)
def update_form(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['form_vs_season'])
    top_form = filtered.nlargest(20, 'form_vs_season')

    fig = px.bar(top_form, x='web_name', y='form_vs_season',
                 hover_data=['team_name', 'form', 'ppg'],
                 text=top_form['form_vs_season'].round(1),
                 labels={'form_vs_season': 'Form vs Season', 'web_name': 'Player'})
    fig.update_traces(marker_color=[COLORS['success'] if x > 0 else COLORS['warning'] for x in top_form['form_vs_season']],
                      textposition='outside')
    fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                      font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'form', 'ppg', 'form_vs_season',
            'adj_xgi90', 'sched_factor', 'ownership']
    table_data = prepare_table_data(filtered.sort_values('form_vs_season', ascending=False).head(50), cols)

    return fig, table_data


# CLEAN SHEETS
@callback(
    [Output('cs-chart', 'figure'), Output('cs-table', 'data')],
    [Input('cs-position', 'value'), Input('cs-team', 'value'), Input('cs-price', 'value'), Input('cs-minutes', 'value'), Input('visit-cs', 'data')],
    prevent_initial_call=True
)
def update_cs(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=['GKP', 'DEF'])
    filtered = filtered.dropna(subset=['cs_per_90', 'gc_per_90'])

    fig = px.scatter(filtered, x='gc_per_90', y='cs_per_90', color='team_name', size='minutes',
                     hover_name='web_name', hover_data=['price', 'clean_sheets', 'goals_conceded'],
                     labels={'team_name': 'Club', 'gc_per_90': 'Goals Conceded per 90',
                             'cs_per_90': 'Clean Sheet per 90'})
    fig.update_layout(template='plotly_white', height=400, font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'clean_sheets', 'cs_per_90', 'goals_conceded',
            'gc_per_90', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'cs_per_90'), cols)

    return fig, table_data


# FIXTURE DIFFICULTY
@callback(
    [Output('fdr-team-bar', 'figure'), Output('fdr-scatter', 'figure'), Output('fdr-table', 'data')],
    [Input('fdr-position', 'value'), Input('fdr-team', 'value'), Input('fdr-price', 'value'),
     Input('fdr-minutes', 'value'), Input('visit-fixtures', 'data')],
    prevent_initial_call=True
)
def update_fdr(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    # Team bar chart uses ALL teams (unfiltered) since FDR is team-level
    data = get_data()
    all_active = data['df_active'].dropna(subset=['avg_fdr_5'])

    # Apply only team filter to the bar chart
    if team != 'All':
        all_for_bar = all_active[all_active['team_name'] == team]
    else:
        all_for_bar = all_active

    team_fdr = all_for_bar.groupby('team_name').agg({
        'avg_fdr_5': 'first',
        'fixture_string': 'first'
    }).reset_index().sort_values('avg_fdr_5')

    bar_fig = go.Figure()
    if len(team_fdr) > 0:
        bar_fig.add_trace(go.Bar(
            x=team_fdr['team_name'],
            y=team_fdr['avg_fdr_5'],
            marker_color=[COLORS['success'] if x <= 2.6 else (COLORS['warning'] if x <= 3.5 else COLORS['danger']) for x
                          in team_fdr['avg_fdr_5']],
            text=[f"{x:.2f}" for x in team_fdr['avg_fdr_5']],
            textposition='outside',
            hovertemplate='%{x}<br>Avg FDR: %{y:.2f}<br>Fixtures: %{customdata}<extra></extra>',
            customdata=team_fdr['fixture_string']
        ))
    bar_fig.add_hline(y=3.0, line_dash="dash", line_color='#999', annotation_text="Avg (3.0)",
                      annotation_position="right")
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Average FDR (Next 5 GWs)', showlegend=False,
                          yaxis=dict(range=[0, 5.5]),
                          font=dict(family=FONT_FAMILY))

    # Player scatter and table use full filters
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['avg_fdr_5'])

    if len(filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No players match current filters", xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=COLORS['text_light']))
        empty_fig.update_layout(template='plotly_white', height=400)
        return bar_fig, empty_fig, []

    # Scatter - Total Points vs FDR
    plot_df = filtered.copy()
    plot_df['bubble_size'] = plot_df['form'].fillna(0.1).clip(lower=0.1)
    scatter_fig = px.scatter(
        plot_df,
        x='avg_fdr_5',
        y='total_points',
        color='position',
        size='bubble_size',
        hover_name='web_name',
        hover_data=['team_name', 'price', 'fixture_string'],
        color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    scatter_fig.add_vline(x=3.0, line_dash="dash", line_color='#999')
    scatter_fig.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title='Avg Fixture Difficulty (lower = easier)',
        yaxis_title='Total Points',
        font=dict(family=FONT_FAMILY)
    )

    # Table - sorted by FDR (ascending = easiest first)
    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'form', 'ownership', 'avg_fdr_5',
            'fixture_string']
    table_data = prepare_table_data(filtered.nsmallest(50, 'avg_fdr_5'), cols)

    return bar_fig, scatter_fig, table_data


# --- FIXTURE TICKER ---
@callback(
    [Output('ticker-heatmap', 'figure'), Output('ticker-title', 'children')],
    [Input('ticker-sort', 'value'), Input('ticker-gws', 'value'),
     Input('visit-fixture-ticker', 'data')],
    prevent_initial_call=True
)
def update_fixture_ticker(sort_by, num_gws, n):
    _need_visit(n)
    data = get_data()
    fixtures_data = data.get('fixtures_data', [])
    teams_df      = data.get('teams_df', pd.DataFrame())
    # Anchor one GW behind the next deadline so the target GW is included.
    # Pre-season this is 0 (target GW1), not 1 — which used to hide GW1.
    current_gw_num = data.get('fixture_anchor_gw')
    if current_gw_num is None:
        current_gw_info = data.get('current_gw')
        current_gw_num = current_gw_info['id'] if current_gw_info else 0

    def _empty(msg):
        fig = go.Figure()
        fig.add_annotation(text=msg, xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=COLORS['text_light']))
        fig.update_layout(template='plotly_white', height=500)
        return [fig, "Fixture Ticker"]

    if teams_df.empty or not fixtures_data:
        return _empty('Data loading — please wait...')

    team_id_to_short = dict(zip(teams_df['id'], teams_df['short_name']))
    team_id_to_name  = dict(zip(teams_df['id'], teams_df['name']))
    all_team_ids     = sorted(teams_df['id'].tolist())

    # Remaining fixtures only (event assigned and still to come)
    remaining = [
        f for f in fixtures_data
        if f.get('event') is not None and f['event'] > current_gw_num
    ]

    if not remaining:
        return _empty('No remaining fixtures found.')

    # Window: the next N gameweeks from the upcoming deadline. Everything
    # downstream (matrices, avg-FDR sort, sizing) works off this slice, so the
    # FDR sort ranks teams by their run over the WINDOW you're looking at.
    all_remaining_gws = sorted(set(f['event'] for f in remaining))
    num_gws = int(num_gws or 5)
    remaining_gws = all_remaining_gws[:num_gws]
    showing_all = len(remaining_gws) == len(all_remaining_gws)
    remaining = [f for f in remaining if f['event'] in set(remaining_gws)]

    # Build team × gw → list of {opponent, venue, fdr}
    team_gw = {tid: {gw: [] for gw in remaining_gws} for tid in all_team_ids}

    for f in remaining:
        gw      = f['event']
        home_id = f['team_h']
        away_id = f['team_a']
        h_fdr   = f.get('team_h_difficulty', 3)
        a_fdr   = f.get('team_a_difficulty', 3)
        h_short = team_id_to_short.get(home_id, '???')
        a_short = team_id_to_short.get(away_id, '???')
        if home_id in team_gw and gw in team_gw[home_id]:
            team_gw[home_id][gw].append({'opponent': a_short, 'venue': 'H', 'fdr': h_fdr})
        if away_id in team_gw and gw in team_gw[away_id]:
            team_gw[away_id][gw].append({'opponent': h_short, 'venue': 'A', 'fdr': a_fdr})

    def _avg_fdr(tid):
        vals = [fx['fdr'] for gw in remaining_gws for fx in team_gw[tid][gw]]
        return sum(vals) / len(vals) if vals else 5.0

    sorted_ids   = sorted(all_team_ids, key=(_avg_fdr if sort_by == 'fdr' else lambda t: team_id_to_name[t]))
    sorted_names = [team_id_to_name[tid] for tid in sorted_ids]

    # Build matrices
    # z encoding: 0=BGW  1=FDR1  2=FDR2  3=FDR3  4=FDR4  5=FDR5  6=DGW
    z_matrix     = []
    text_matrix  = []
    hover_matrix = []

    for tid in sorted_ids:
        z_row = []; text_row = []; hover_row = []
        for gw in remaining_gws:
            fixes = team_gw[tid][gw]
            if len(fixes) == 0:
                z_row.append(0)
                text_row.append('BGW')
                hover_row.append('Blank Gameweek')
            elif len(fixes) == 1:
                fx = fixes[0]
                z_row.append(fx['fdr'])
                text_row.append(f"{fx['opponent']} ({fx['venue']})")
                hover_row.append(f"{fx['opponent']} ({fx['venue']})  FDR {fx['fdr']}")
            else:
                z_row.append(6)
                text_row.append('<br>'.join(f"{fx['opponent']} ({fx['venue']})" for fx in fixes))
                hover_row.append('DGW: ' + '  +  '.join(
                    f"{fx['opponent']} ({fx['venue']}) FDR {fx['fdr']}" for fx in fixes))
        z_matrix.append(z_row)
        text_matrix.append(text_row)
        hover_matrix.append(hover_row)

    # Piecewise-constant colorscale — midpoint boundaries between integer z values.
    # zmin=0, zmax=6.  Normalised position of integer n is n/6.
    # Each band is centred on its integer; colour switches at (n + 0.5)/6.
    # All values guaranteed within [0, 1].
    colorscale = [
        [0/6,    '#d0d0d0'],  # 0 BGW  (grey)
        [0.5/6,  '#d0d0d0'],
        [0.5/6,  '#00ff87'],  # 1 FDR1 (bright green)
        [1.5/6,  '#00ff87'],
        [1.5/6,  '#7dde9e'],  # 2 FDR2 (soft green)
        [2.5/6,  '#7dde9e'],
        [2.5/6,  '#ffc107'],  # 3 FDR3 (amber)
        [3.5/6,  '#ffc107'],
        [3.5/6,  '#ff7043'],  # 4 FDR4 (orange-red)
        [4.5/6,  '#ff7043'],
        [4.5/6,  '#dc3545'],  # 5 FDR5 (red)
        [5.5/6,  '#dc3545'],
        [5.5/6,  '#00bcd4'],  # 6 DGW  (cyan — distinct from FDR scale, readable with dark text)
        [1.0,    '#00bcd4'],
    ]

    # Adaptive sizing: fewer columns = bigger, clearer cells. Past 15 columns
    # no font is small enough to avoid overlap, so drop cell text entirely and
    # let colour + hover carry the information.
    n_cols = len(remaining_gws)
    if n_cols <= 8:
        cell_font, tick_font, show_text = 13, 16, True
    elif n_cols <= 12:
        cell_font, tick_font, show_text = 11, 14, True
    elif n_cols <= 15:
        cell_font, tick_font, show_text = 9, 12, True
    else:
        cell_font, tick_font, show_text = 9, 10, False

    height = max(520, len(sorted_ids) * 34 + 120)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[f"GW{gw}" for gw in remaining_gws],
        y=sorted_names,
        text=text_matrix,
        customdata=hover_matrix,
        texttemplate='%{text}' if show_text else '',
        colorscale=colorscale,
        zmin=0,
        zmax=6,
        showscale=False,
        hovertemplate='<b>%{y}</b>  %{x}<br>%{customdata}<extra></extra>',
        xgap=2,
        ygap=2,
        textfont=dict(size=cell_font, color='#333333'),
    ))

    # Fixed cell width rather than fitting the viewport. A 380px phone
    # divided by six columns gives 50px cells in which nothing is legible;
    # the .chart-scroll container scrolls sideways instead.
    _cell_w = 74 if n_cols > 4 else 96
    fig.update_layout(
        template='plotly_white',
        height=height,
        width=118 + _cell_w * max(n_cols, 1) + 28,
        font=dict(family=FONT_FAMILY, size=12),
        xaxis=dict(side='top', tickangle=0 if n_cols <= 15 else -45,
                   fixedrange=True, tickfont=dict(size=tick_font)),
        yaxis=dict(autorange='reversed', fixedrange=True, tickfont=dict(size=12)),
        margin=dict(l=110, r=18, t=56, b=10),
    )

    if showing_all:
        title = "Fixture Ticker: Remainder of the Season"
    else:
        title = (f"Fixture Ticker: Next {len(remaining_gws)} Gameweeks "
                 f"(GW{remaining_gws[0]} to GW{remaining_gws[-1]})")

    return [fig, title]


# --- OWNERSHIP DIFFERENTIALS ---
@callback(
    [Output('diff-scatter', 'figure'), Output('diff-bar', 'figure'), Output('diff-table', 'data')],
    [Input('diff-position', 'value'), Input('diff-team', 'value'), Input('diff-price', 'value'),
     Input('diff-max-own', 'value'), Input('diff-minutes', 'value'), Input('visit-differentials', 'data')],
    prevent_initial_call=True
)
def update_differentials(position, team, max_price, max_own, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered[filtered['ownership'] <= max_own]
    filtered = filtered.dropna(subset=['ppg', 'ownership'])

    # Handle empty data
    if len(filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No players match current filters. Try decreasing the value of the min. minutes",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=COLORS['text_light']))
        empty_fig.update_layout(template='plotly_white', height=400)
        return empty_fig, empty_fig, []

    plot_df = filtered.copy()
    plot_df['bubble_size'] = plot_df['form'].fillna(0.1).clip(lower=0.1)
    scatter_fig = px.scatter(
        plot_df, x='ownership', y='ppg', color='position', size='bubble_size',
        hover_name='web_name',
        hover_data=['team_name', 'price', 'form', 'expected_goal_involvements', 'total_points'],
        color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    if len(filtered) > 0:
        _d = get_data()
        _cgd = _d.get('current_gw')
        _thr = adaptive_min_minutes(450, _cgd['id'] if _cgd else 0)
        median_ppg = _d['df_active'][_d['df_active']['minutes'] >= _thr]['ppg'].median()
        scatter_fig.add_hline(y=median_ppg, line_dash='dash', line_color='#999',
                              annotation_text=f'Median PPG ({median_ppg:.1f})', annotation_position='top right')
    scatter_fig.update_layout(template='plotly_white', height=400, xaxis_title='Ownership %',
                              yaxis_title='Points per Game',
                              font=dict(family=FONT_FAMILY))

    top_25 = filtered.nlargest(25, 'differential_score')
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_25['web_name'], y=top_25['differential_score'],
        marker_color=COLORS['accent'],
        text=[f"{x:.1f}" for x in top_25['differential_score']],
        textposition='outside',
        hovertemplate='%{x}<br>Score: %{y:.2f}<br>Own: %{customdata[0]:.1f}%<br>PPG: %{customdata[1]:.2f}<extra></extra>',
        customdata=top_25[['ownership', 'ppg']].values
    ))
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Differential Score', showlegend=False,
                          yaxis=dict(range=[0, top_25['differential_score'].max() * 1.1]),
                          font=dict(family=FONT_FAMILY))

    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'form', 'ppg',
            'expected_goal_involvements', 'ownership', 'top_eo', 'own_delta_7d',
            'differential_score', 'proj_pts_5', 'avg_fdr_5', 'fixture_string']
    table_data = prepare_table_data(filtered.nlargest(50, 'differential_score'), cols)

    return scatter_fig, bar_fig, table_data


@callback(
    Output('diff-max-own-input', 'value'),
    Input('diff-max-own', 'value')
)
def sync_own_input(slider_val):
    return slider_val


@callback(
    Output('diff-max-own', 'value'),
    Input('diff-max-own-input', 'value'),
    prevent_initial_call=True
)
def sync_own_slider(input_val):
    if input_val is None:
        return 15
    return max(5, min(100, input_val))


# --- REGRESSION WATCHLIST ---
@callback(
    [Output('regress-sell-table', 'data'), Output('regress-buy-table', 'data')],
    Input('visit-form', 'data'),
    prevent_initial_call=True
)
def update_regression_watchlist(_n):
    _need_visit(_n)
    data = get_data()
    dfa = data.get('df_active', pd.DataFrame())
    if dfa.empty or 'xgi_diff_per_90' not in dfa.columns:
        return [], []
    cur = data.get('current_gw')
    thr = adaptive_min_minutes(450, cur['id'] if cur else 0)
    pool = dfa[(dfa['minutes'] >= thr) &
               (dfa['position'].isin(['DEF', 'MID', 'FWD']))].copy()
    pool = pool.dropna(subset=['xgi_diff_per_90', 'xgi_per_90'])
    cols = ['web_name', 'team_name', 'gi_per_90', 'xgi_per_90', 'xgi_diff_per_90', 'ownership']
    # SELL: producing well above xGI, meaningfully owned (someone to sell)
    sell = pool[(pool['xgi_diff_per_90'] >= 0.30) & (pool['ownership'] >= 5)]
    sell = sell.nlargest(8, 'xgi_diff_per_90')
    # BUY: elite underlying, output lagging
    buy = pool[(pool['xgi_diff_per_90'] <= -0.20) & (pool['xgi_per_90'] >= 0.35)]
    buy = buy.nsmallest(8, 'xgi_diff_per_90')
    return prepare_table_data(sell, cols), prepare_table_data(buy, cols)


# --- FIXTURE SWING DETECTOR ---
@callback(
    Output('fdr-swing-table', 'data'),
    Input('visit-fixtures', 'data'),
    prevent_initial_call=True
)
def update_fixture_swings(_n):
    _need_visit(_n)
    data = get_data()
    fixtures_data = data.get('fixtures_data', [])
    teams_df = data.get('teams_df', pd.DataFrame())
    anchor = data.get('fixture_anchor_gw')
    if anchor is None:
        cur = data.get('current_gw')
        anchor = cur['id'] if cur else 0
    if teams_df.empty or not fixtures_data:
        return []
    near = calculate_custom_fdr(fixtures_data, teams_df, anchor, num_gameweeks=3)
    far = calculate_custom_fdr(fixtures_data, teams_df, anchor + 3, num_gameweeks=3)
    name_map = dict(zip(teams_df['id'], teams_df['name']))
    rows = []
    for tid in name_map:
        n_ = near.get(tid, {})
        f_ = far.get(tid, {})
        if n_.get('att_fdr') is None or f_.get('att_fdr') is None:
            continue
        rows.append({
            'team': name_map[tid],
            'now_att': n_['att_fdr'], 'later_att': f_['att_fdr'],
            'att_swing': round(f_['att_fdr'] - n_['att_fdr'], 2),
            'now_def': n_.get('def_fdr'), 'later_def': f_.get('def_fdr'),
            'def_swing': round((f_.get('def_fdr') or 3) - (n_.get('def_fdr') or 3), 2),
        })
    rows.sort(key=lambda r: r['att_swing'])
    return rows


# --- CAPTAIN Optimiser ---
@callback(
    [Output('cap-bar', 'figure'), Output('cap-ha-scatter', 'figure'), Output('cap-table', 'data')],
    [Input('cap-position', 'value'), Input('cap-team', 'value'), Input('cap-price', 'value'),
     Input('cap-minutes', 'value'), Input('cap-mode', 'value'),
     Input('visit-captain', 'data')],
    prevent_initial_call=True
)
def update_captain(position, team, max_price, min_minutes, mode, _n):
    """
    Captain ranking, env-aware. Protect mode ranks by projected points
    (which now scale attacking output by the fixture's goal environment, so
    'elite striker vs promoted side at home' beats 'good week last week').
    Chase mode ranks by haul probability — when you're behind, the doubled
    captain is your variance lever and P(2+ involvements) is the metric.
    Defensive: any failure renders an error message, never a dead page.
    """
    _need_visit(_n)
    try:
        filtered = filter_data(position, team, max_price, min_minutes,
                               positions_allowed=SEASON['outfield_positions'])
        filtered = filtered.dropna(subset=['captain_score'])
        filtered = filtered[filtered['captain_score'] > 0]

        # Chase ranks on P(15+ TOTAL points) — every scoring route, not just
        # goal involvements. Differential ranks on points gained against the
        # field after effective ownership. Protect stays on expected points.
        if mode == 'ceiling' and 'p_15' in filtered.columns and filtered['p_15'].notna().any():
            rank_col, rank_label = 'p_15', 'P(15+ points) %'
        elif mode == 'ceiling' and 'haul_pct' in filtered.columns and filtered['haul_pct'].notna().any():
            rank_col, rank_label = 'haul_pct', 'Haul probability (%)'
        elif mode == 'gain' and 'captain_gain' in filtered.columns and filtered['captain_gain'].notna().any():
            rank_col, rank_label = 'captain_gain', 'Expected gain vs field (pts)'
        elif 'proj_pts_next' in filtered.columns and filtered['proj_pts_next'].notna().any():
            rank_col, rank_label = 'proj_pts_next', 'Projected points (next GW)'
        else:
            rank_col, rank_label = 'captain_score', 'Captain score'

        if len(filtered) == 0:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No captain candidates found. Try reducing the min. minutes",
                                     xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                     font=dict(size=14, color=COLORS['text_light']))
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, empty_fig, []

        top_20 = filtered.nlargest(20, rank_col)
        env_series = pd.to_numeric(top_20.get('att_env_next'), errors='coerce').fillna(1.0) \
            if 'att_env_next' in top_20.columns else pd.Series(1.0, index=top_20.index)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=top_20['web_name'], y=top_20[rank_col],
            marker_color=[COLORS['success'] if v == 'H' else COLORS['info']
                          for v in top_20['next_venue']],
            text=[f"{v:.1f}" for v in top_20[rank_col]],
            textposition='outside',
            hovertemplate=('%{x}<br>' + rank_label + ': %{y:.2f}<br>'
                           'vs %{customdata[0]} (%{customdata[1]})<br>'
                           'Attack env: \u00d7%{customdata[2]:.2f}<br>'
                           'Proj: %{customdata[3]:.2f}  |  Haul: %{customdata[4]:.0f}%'
                           '<extra></extra>'),
            customdata=np.column_stack([
                top_20['next_opponent'].fillna(''),
                top_20['next_venue'].fillna(''),
                env_series,
                pd.to_numeric(top_20.get('proj_pts_next'), errors='coerce').fillna(0),
                pd.to_numeric(top_20.get('haul_pct'), errors='coerce').fillna(0),
            ]),
        ))
        bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                              yaxis_title=rank_label, showlegend=False,
                              yaxis=dict(range=[0, float(top_20[rank_col].max()) * 1.15]),
                              font=dict(family=FONT_FAMILY))

        ha_filtered = filtered.dropna(subset=['home_ppg', 'away_ppg'])
        ha_scatter = px.scatter(
            ha_filtered, x='away_ppg', y='home_ppg', color='position',
            hover_name='web_name',
            hover_data=['team_name', 'next_opponent', 'next_venue', 'price'],
            color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                'FWD': COLORS['info']}
        )
        if len(ha_filtered) > 0:
            max_val = max(ha_filtered['home_ppg'].max(), ha_filtered['away_ppg'].max(), 1)
            ha_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                            line=dict(dash='dash', color='#999'), name='Equal'))
        else:
            ha_scatter.add_annotation(text="Home/away splits appear once enough matches are played",
                                      xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                      font=dict(size=13, color=COLORS['text_light']))
        ha_scatter.update_layout(template='plotly_white', height=400,
                                 xaxis_title='Away PPG', yaxis_title='Home PPG')

        cols = ['web_name', 'team_name', 'position', 'price', 'proj_pts_next',
                'p_10', 'p_15', 'captain_gain', 'captain_score',
                'haul_pct', 'att_env_next', 'ep_next', 'form', 'ppg',
                'xgi_per_90', 'next_att_fdr', 'venue_ppg', 'start_rate', 'avail_pct',
                'set_pieces', 'top_eo',
                'next_opponent', 'next_venue', 'next_fdr',
                'home_ppg', 'away_ppg', 'bps_per_90', 'ownership']
        cols = [c for c in cols if c in filtered.columns]
        table_data = prepare_table_data(filtered.nlargest(50, rank_col), cols)

        return bar_fig, ha_scatter, table_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"Captain page error: {e}", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=13, color=COLORS['danger_text']))
        err_fig.update_layout(template='plotly_white', height=400)
        return err_fig, err_fig, []


# --- TRANSFER TRENDS ---
@callback(
    [Output('xfer-risers-bar', 'figure'), Output('xfer-fallers-bar', 'figure'),
     Output('xfer-scatter', 'figure'), Output('xfer-table', 'data'),
     Output('xfer-delta-basis', 'children')],
    [Input('xfer-position', 'value'), Input('xfer-team', 'value'),
     Input('xfer-price', 'value'), Input('xfer-minutes', 'value'), Input('visit-transfers', 'data')],
    prevent_initial_call=True
)
def update_transfers(position, team, max_price, min_minutes, _visit=None):
    _need_visit(_visit)
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['net_transfers_gw'])

    risers = filtered.nlargest(20, 'net_transfers_gw')
    risers_fig = go.Figure()
    risers_fig.add_trace(go.Bar(
        x=risers['web_name'], y=risers['net_transfers_gw'],
        marker_color=COLORS['success'],
        text=[f"+{int(x):,}" for x in risers['net_transfers_gw']],
        textposition='outside',
        hovertemplate='%{x}<br>Net In: +%{y:,.0f}<br>Price: %{customdata[0]:.1f}<br>Own: %{customdata[1]:.1f}%<extra></extra>',
        customdata=risers[['price', 'ownership']].values
    ))
    risers_fig.update_layout(template='plotly_white', height=380, xaxis_tickangle=-45,
                             yaxis_title='Net Transfers In', showlegend=False,
                             yaxis=dict(range=[0, risers['net_transfers_gw'].max() * 1.1]),
                             font=dict(family=FONT_FAMILY))

    fallers = filtered.nsmallest(20, 'net_transfers_gw')
    fallers_fig = go.Figure()
    fallers_fig.add_trace(go.Bar(
        x=fallers['web_name'], y=fallers['net_transfers_gw'],
        marker_color=COLORS['danger'],
        text=[f"{int(x):,}" for x in fallers['net_transfers_gw']],
        textposition='outside',
        hovertemplate='%{x}<br>Net Out: %{y:,.0f}<br>Price: %{customdata[0]:.1f}<br>Own: %{customdata[1]:.1f}%<extra></extra>',
        customdata=fallers[['price', 'ownership']].values
    ))
    fallers_fig.update_layout(template='plotly_white', height=380, xaxis_tickangle=-45,
                              yaxis_title='Net Transfers Out', showlegend=False,
                              yaxis=dict(range=[fallers['net_transfers_gw'].min() * 1.1, 0]),
                              font=dict(family=FONT_FAMILY))

    scatter_fig = px.scatter(
        filtered[filtered['ownership'] >= 1],
        x='net_transfers_gw', y='cost_change_start',
        color='position', size='ownership',
        hover_name='web_name',
        hover_data=['team_name', 'price', 'form', 'transfers_in_gw', 'transfers_out_gw'],
        color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    scatter_fig.add_hline(y=0, line_dash='dash', line_color='#999')
    scatter_fig.add_vline(x=0, line_dash='dash', line_color='#999')
    scatter_fig.update_layout(template='plotly_white', height=400,
                              xaxis_title='Net Transfers This GW', yaxis_title='Season Price Change (m)',
                              font=dict(family=FONT_FAMILY))

    sorted_by_activity = filtered.copy()
    sorted_by_activity['abs_net'] = sorted_by_activity['net_transfers_gw'].abs()
    cols = ['web_name', 'team_name', 'position', 'price', 'transfers_in_gw', 'transfers_out_gw',
            'net_transfers_gw', 'transfer_ratio', 'price_status', 'price_progress', 'price_predicted',
            'own_delta_7d',
            'price_delta_7d', 'cost_change_start', 'form', 'ownership']
    for c in ('price_status', 'price_progress', 'price_predicted'):
        if c not in sorted_by_activity.columns:
            sorted_by_activity[c] = None
    table_data = prepare_table_data(sorted_by_activity.nlargest(50, 'abs_net'), cols)

    basis = get_data().get('delta_basis') or 'this gameweek'
    _pt = get_data().get('price_tracker_at')
    tracker_txt = (f"Price Status, Progress and Predicted come straight from FPL's official price "
                   f"change tracker (read at {datetime.fromtimestamp(_pt, tz=ZoneInfo('Europe/London')):%H:%M}; "
                   f"a price changes when progress reaches \u00b1100% at the overnight update). "
                   if _pt else "FPL's price tracker isn't available right now, so those columns are empty. ")
    note = (tracker_txt + f"Own \u0394 and \u00a3 \u0394 are measured over {basis}. "
            f"Ownership change is derived from net transfers \u00f7 total managers, "
            f"and price change from FPL's own gameweek movement \u2014 neither needs "
            f"stored history, so both populate on the first run.")
    return risers_fig, fallers_fig, scatter_fig, table_data, note


# --- FIXTURE OUTLOOK: modelled grid, not FDR integers ---
@callback(
    [Output('fo-heatmap', 'figure'), Output('fo-table', 'children')],
    [Input('visit-fixture-outlook', 'data'), Input('fo-gws', 'value'),
     Input('fo-view', 'value'), Input('fo-sort', 'value')],
    prevent_initial_call=True
)
def update_fixture_outlook(page, n_gws, view, sort_by):
    _need_visit(page)
    blank = go.Figure(); blank.update_layout(template='plotly_white', height=320)

    data = get_data()
    fixtures, teams_df = data.get('fixtures_data'), data.get('teams_df')
    if not fixtures or teams_df is None or teams_df.empty:
        return blank, html.P("Data not loaded yet.", style={'color': COLORS['text_light']})

    try:
        n_gws = max(1, min(38, int(n_gws or 6)))
    except (TypeError, ValueError):
        n_gws = 6

    anchor_gw = data.get('fixture_anchor_gw') or (data.get('next_gw_num', 1) - 1)
    odds = data.get('odds_lambdas')
    ledger = data.get('xg_ledger')

    if view == 'defence':
        model = calculate_expected_clean_sheets(fixtures, teams_df, anchor_gw,
                                                num_gws=n_gws, odds_lambdas=odds,
                                                xg_ledger=ledger)
        # fixtures: (gw, opp, venue, p_cs) — p_cs already a probability
        cell_fn = lambda v: v * 100
        fmt, unit = '{:.0f}%', 'Clean sheet %'
        total_label, total_fmt = 'Expected CS', '{:.2f}'
        good_high = True
    else:
        genv = calculate_goal_environment(fixtures, teams_df, anchor_gw,
                                          num_gws=n_gws, odds_lambdas=odds,
                                          xg_ledger=ledger)
        model = genv
        # fixtures: (gw, opp, venue, env_ratio) — convert ratio to goals
        cell_fn = None
        fmt, unit = '{:.2f}', 'Expected goals'
        total_label, total_fmt = 'Expected goals', '{:.2f}'
        good_high = True

    lg_avg = 1.40
    for v in model.values():
        if isinstance(v, dict) and v.get('league_avg_goals'):
            lg_avg = v['league_avg_goals']
            break

    short = dict(zip(teams_df['id'], teams_df['short_name']))
    names = dict(zip(teams_df['id'], teams_df['name']))
    gws = sorted({gw for v in model.values()
                  for (gw, *_rest) in (v.get('fixtures') or []) if gw})[:n_gws]
    if not gws:
        return blank, html.P("No upcoming fixtures in range.",
                             style={'color': COLORS['text_light']})

    rows = []
    for tid, v in model.items():
        cells = {g: [] for g in gws}
        for (gw, opp, ven, val) in (v.get('fixtures') or []):
            if gw in cells:
                cells[gw].append((opp, ven, val * 100 if view == 'defence' else val * lg_avg))
        vals = [x[2] for g in gws for x in cells[g]]
        if not vals:
            continue
        rows.append({
            'tid': tid, 'name': names.get(tid, '?'), 'cells': cells,
            'total': sum(vals), 'worst': min(vals), 'n': len(vals),
        })
    if not rows:
        return blank, html.P("No fixtures to show.", style={'color': COLORS['text_light']})

    if sort_by == 'name':
        rows.sort(key=lambda r: r['name'])
    else:
        rows.sort(key=lambda r: r['total'], reverse=(sort_by == 'total'))

    # Cell geometry: a fixed width per gameweek beats fitting the screen,
    # because a 380px phone divided by 6 columns gives 45px cells in which
    # nothing is readable. The container scrolls instead.
    compact = len(gws) > 4
    cell_w = 74 if compact else 96
    fig_w = 132 + cell_w * len(gws) + 96   # labels + cells + colourbar

    z, text, hover = [], [], []
    for r in rows:
        zr, tr, hr = [], [], []
        for g in gws:
            fx = r['cells'][g]
            if not fx:
                zr.append(None); tr.append('BGW'); hr.append('Blank gameweek')
            else:
                tot = sum(x[2] for x in fx)
                zr.append(tot)
                label = ' + '.join(f"{o} ({vn})" for o, vn, _ in fx)
                if len(fx) > 1:
                    tr.append(f"DGW<br>{fmt.format(tot)}")
                elif compact:
                    # Narrow cells: opponent only on the top line, no venue
                    # brackets, so the number underneath stays legible.
                    tr.append(f"{fx[0][0]}<br>{fmt.format(tot)}")
                else:
                    tr.append(f"{label}<br>{fmt.format(tot)}")
                hr.append(f"{r['name']} GW{g}<br>{label}<br>{unit}: {fmt.format(tot)}")
        z.append(zr); text.append(tr); hover.append(hr)

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"GW{g}" for g in gws], y=[r['name'] for r in rows],
        text=text, texttemplate='%{text}',
        textfont={'size': 9 if compact else 11, 'family': FONT_FAMILY},
        hovertext=hover, hoverinfo='text',
        colorscale=[[0, '#dc3545'], [0.35, '#ff7043'], [0.55, '#ffc107'],
                    [0.75, '#7dde9e'], [1, '#00ff87']],
        showscale=True,
        colorbar=dict(title=dict(text=unit, side='right'), thickness=12,
                      len=0.75, tickfont={'size': 10}),
        xgap=2, ygap=2))
    fig.update_layout(template='plotly_white',
                      width=fig_w,
                      height=max(360, 34 * len(rows) + 110),
                      margin=dict(t=34, b=30, l=124, r=8),
                      xaxis=dict(side='top', fixedrange=True,
                                 tickfont={'size': 11}),
                      yaxis=dict(autorange='reversed', fixedrange=True,
                                 tickfont={'size': 11}),
                      font=dict(family=FONT_FAMILY))

    table = dash_table.DataTable(
        data=[{'team': r['name'], 'total': round(r['total'], 2),
               'per_fix': round(r['total'] / r['n'], 2),
               'worst': round(r['worst'], 2), 'n': r['n']} for r in rows],
        columns=[
            {'name': 'Team', 'id': 'team'},
            {'name': f'{total_label} (window)', 'id': 'total', 'type': 'numeric',
             'format': {'specifier': '.2f'}},
            {'name': 'Per fixture', 'id': 'per_fix', 'type': 'numeric',
             'format': {'specifier': '.2f'}},
            {'name': 'Worst GW', 'id': 'worst', 'type': 'numeric',
             'format': {'specifier': '.2f'}},
            {'name': 'Fixtures', 'id': 'n', 'type': 'numeric'},
        ],
        sort_action='native',
        style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'}])
    return fig, table


# --- MODEL LAB: WALK-FORWARD BACKTEST ---

# Shared by the Diff and Mins filter dropdowns on the backtest player table —
# a plain-English operator picker instead of requiring Dash's native
# `{diff} >= 4`-style filter-box syntax.
NUMERIC_FILTER_OPTIONS = [
    {'label': 'Any', 'value': 'any'},
    {'label': 'Greater than (>)', 'value': 'gt'},
    {'label': 'Greater than or equal to (\u2265)', 'value': 'gte'},
    {'label': 'Less than (<)', 'value': 'lt'},
    {'label': 'Less than or equal to (\u2264)', 'value': 'lte'},
    {'label': 'Equal to (=)', 'value': 'eq'},
    {'label': 'Not equal to (\u2260)', 'value': 'neq'},
]
_NUMERIC_FILTER_FNS = {
    'gt': lambda v, t: v > t, 'gte': lambda v, t: v >= t,
    'lt': lambda v, t: v < t, 'lte': lambda v, t: v <= t,
    'eq': lambda v, t: v == t, 'neq': lambda v, t: v != t,
}


def _apply_numeric_filter(rows, field, op, value):
    """Filter a list of row dicts on one numeric field by operator + value.
    No-ops (returns rows unchanged) if op is 'any'/unset or value is blank —
    0 is a legitimate threshold, so this checks for None, not falsiness."""
    fn = _NUMERIC_FILTER_FNS.get(op)
    if fn is None or value is None or value == '':
        return rows
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return rows
    return [r for r in rows if r.get(field) is not None and fn(r[field], threshold)]

@callback(
    [Output('lab-bt-out', 'children'), Output('lab-player-rows-store', 'data')],
    Input('lab-bt-run', 'n_clicks'),
    prevent_initial_call=True
)
def render_backtest(n_clicks):
    try:
        return _render_backtest_inner()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.P("Backtest failed.", style={'color': COLORS['danger_text'],
                                              'fontWeight': '600', 'marginBottom': '6px'}),
            html.Pre(f"{type(e).__name__}: {e}",
                     style={'color': COLORS['text_light'], 'fontSize': '13px',
                            'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa',
                            'padding': '10px', 'borderRadius': '6px'}),
            html.P("Full traceback is in the server logs.",
                   style={'color': COLORS['text_light'], 'fontSize': '13px'}),
        ]), []


def _render_backtest_inner():
    """Returns (children, player_rows) — player_rows is the raw, unfiltered
    per-player-per-GW list that feeds lab-player-rows-store, so the Diff/Mins
    filter controls can re-slice it without re-running the backtest."""
    data = get_data()
    dfa, boot = data.get('df_active'), data.get('bootstrap_data')
    fixtures, teams_df = data.get('fixtures_data'), data.get('teams_df')
    if dfa is None or dfa.empty or not fixtures or teams_df is None:
        return html.P("Data not loaded yet.", style={'color': COLORS['text_light']}), []

    finished_gws = sorted({f['event'] for f in fixtures
                           if f.get('event') and (f.get('finished') or
                                                  f.get('finished_provisional'))})
    # GW1 has no prior history to reconstruct from, so scoring starts at GW2
    gws = [g for g in finished_gws if g >= 2]
    if not gws:
        return html.P("No completed gameweeks with a prior gameweek to learn from yet. "
                      "The backtest becomes available from GW2.",
                      style={'color': COLORS['text_light']}), []


    # Fetch budget. Render kills long-running HTTP requests, and fetching
    # ~600 element-summaries inside a callback blows straight through that —
    # the spinner then hangs forever with nothing in the logs. So: seed from
    # whatever the refresh already pulled, restrict to players with a real
    # sample, and time-box the rest. A partial fetch still produces a valid
    # backtest over fewer players; a killed request produces nothing.
    t_start = time.time()
    cached = dict(data.get('backtest_histories') or {})
    seed = data.get('player_histories') or {}
    for _k, _v in seed.items():
        cached.setdefault(int(_k), _v)

    wanted = dfa[dfa['minutes'] >= BACKTEST_MIN_MINUTES]['id'].astype(int).tolist()
    missing = [p for p in wanted if p not in cached]
    print(f"[backtest] {len(wanted)} players in scope, {len(missing)} to fetch, "
          f"budget {BACKTEST_FETCH_BUDGET}s")

    if missing:
        for i in range(0, len(missing), BACKTEST_CHUNK):
            if time.time() - t_start > BACKTEST_FETCH_BUDGET:
                print(f"[backtest] fetch budget spent, continuing with "
                      f"{len(cached)} players")
                break
            chunk = missing[i:i + BACKTEST_CHUNK]
            cached.update(fetch_player_history_batch(chunk, max_workers=10))
            print(f"[backtest] fetched {min(i + BACKTEST_CHUNK, len(missing))}"
                  f"/{len(missing)} ({time.time() - t_start:.0f}s)")
        with DATA_LOCK:
            DATA['backtest_histories'] = cached

    usable = {k: v for k, v in cached.items() if k in set(wanted)}
    if len(usable) < 30:
        return html.P(f"Only {len(usable)} player histories available. Press Run "
                      f"Backtest again — each press fetches another batch and keeps "
                      f"what it already has.", style={'color': COLORS['text_light']}), []
    print(f"[backtest] scoring {len(usable)} players over GWs {gws}")

    meta = {int(r.id): {'position': r.position, 'team': int(r.team),
                        'web_name': r.web_name, 'pen_rank': getattr(r, 'pen_rank', None)}
            for r in dfa.itertuples()}
    priors = data.get('last_season_priors', {})

    rows, player_rows = run_projection_backtest(usable, meta, fixtures, teams_df,
                                                gws, priors=priors)
    print(f"[backtest] done in {time.time() - t_start:.0f}s, "
          f"{len(rows)} gameweeks scored")
    if not rows:
        return html.P("Not enough reconstructable history yet.",
                      style={'color': COLORS['text_light']}), []

    n = sum(r['players'] for r in rows)
    w = lambda k: sum(r[k] * r['players'] for r in rows) / n
    mae, mae_ppg, mae_pos = w('mae'), w('mae_ppg'), w('mae_pos')
    spear = sum(r['spearman'] * r['players'] for r in rows) / n
    bias = w('bias')
    top20 = sum(r['top20'] for r in rows) / len(rows)

    beats_ppg = mae < mae_ppg
    verdict = (f"Model MAE {mae:.3f} vs {mae_ppg:.3f} for 'use his points per game' and "
               f"{mae_pos:.3f} for the positional average. ")
    verdict += ("The model is beating both naive baselines."
                if beats_ppg and mae < mae_pos else
                "The model is NOT beating the naive baselines \u2014 the modelling layer "
                "is not earning its place yet.")
    verdict += (f" Rank correlation {spear:.3f}; {top20:.1f} of the top 20 projections "
                f"landed in the actual top 20 per gameweek. Bias {bias:+.3f} "
                f"({'over' if bias > 0 else 'under'}-projecting on average).")

    return html.Div([
        html.P(verdict, style={'color': COLORS['text_dark'], 'fontWeight': '600',
                               'marginBottom': '12px'}),
        dash_table.DataTable(
            data=[{**r, 'gw': f"GW{r['gw']}"} for r in rows],
            columns=[
                {'name': 'GW', 'id': 'gw'},
                {'name': 'Players', 'id': 'players', 'type': 'numeric'},
                {'name': 'Model MAE', 'id': 'mae', 'type': 'numeric',
                 'format': {'specifier': '.3f'}},
                {'name': 'PPG baseline', 'id': 'mae_ppg', 'type': 'numeric',
                 'format': {'specifier': '.3f'}},
                {'name': 'Pos-avg baseline', 'id': 'mae_pos', 'type': 'numeric',
                 'format': {'specifier': '.3f'}},
                {'name': 'Spearman', 'id': 'spearman', 'type': 'numeric',
                 'format': {'specifier': '.3f'}},
                {'name': 'Top-20 hits', 'id': 'top20', 'type': 'numeric'},
                {'name': 'Bias', 'id': 'bias', 'type': 'numeric',
                 'format': {'specifier': '+.3f'}},
            ],
            sort_action='native',
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
            style_data=TABLE_STYLE_DATA,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
            ]),

        html.H4("Every Player, Every Gameweek",
                style={'color': COLORS['primary'], 'margin': '24px 0 8px 0'}),
        html.P("What the model projected and what he actually scored. Sort any column, "
               "or type in the GW/Player/Pos boxes to search by text. "
               "Diff is actual minus projected, so positive means the model was too low. "
               "Use the Diff/Mins filters below for numeric thresholds \u2014 no query "
               "syntax needed.",
               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),

        html.Div([
            html.Div([
                html.Label("Diff filter", style={'fontWeight': '600', 'marginBottom': '6px',
                                                  'display': 'block'}),
                html.Div([
                    dcc.Dropdown(id='lab-diff-op', options=NUMERIC_FILTER_OPTIONS,
                                 value='any', clearable=False,
                                 style={'minWidth': '190px', 'flex': '1'}),
                    dcc.Input(id='lab-diff-value', type='number', placeholder='e.g. 0',
                              style={'width': '90px', 'padding': '8px', 'marginLeft': '8px',
                                     'borderRadius': '4px', 'border': '1px solid #ccc'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'flex': '1', 'minWidth': '280px', 'padding': '0 10px'}),
            html.Div([
                html.Label("Mins filter", style={'fontWeight': '600', 'marginBottom': '6px',
                                                  'display': 'block'}),
                html.Div([
                    dcc.Dropdown(id='lab-mins-op', options=NUMERIC_FILTER_OPTIONS,
                                 value='any', clearable=False,
                                 style={'minWidth': '190px', 'flex': '1'}),
                    dcc.Input(id='lab-mins-value', type='number', placeholder='e.g. 90',
                              style={'width': '90px', 'padding': '8px', 'marginLeft': '8px',
                                     'borderRadius': '4px', 'border': '1px solid #ccc'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'flex': '1', 'minWidth': '280px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '16px'}),

        dash_table.DataTable(
            id='lab-player-table',
            data=player_rows,
            columns=[
                {'name': 'GW', 'id': 'gw'},
                {'name': 'Player', 'id': 'web_name'},
                {'name': 'Pos', 'id': 'position'},
                {'name': 'Projected', 'id': 'proj', 'type': 'numeric',
                 'format': {'specifier': '.2f'}},
                {'name': 'Actual', 'id': 'actual', 'type': 'numeric'},
                {'name': 'Diff', 'id': 'diff', 'type': 'numeric',
                 'format': {'specifier': '+.2f'}},
                {'name': 'Mins', 'id': 'minutes_played', 'type': 'numeric'},
            ],
            sort_action='native', filter_action='native', page_size=25,
            sort_by=[{'column_id': 'gw', 'direction': 'asc'},
                     {'column_id': 'actual', 'direction': 'desc'}],
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
            style_data=TABLE_STYLE_DATA,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                {'if': {'filter_query': '{diff} >= 4', 'column_id': 'diff'},
                 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
                {'if': {'filter_query': '{diff} <= -4', 'column_id': 'diff'},
                 'backgroundColor': '#fde8ef', 'fontWeight': '600'},
                {'if': {'filter_query': '{minutes_played} = 0'},
                 'color': COLORS['text_light'], 'fontStyle': 'italic'},
            ])
    ]), player_rows


@callback(
    Output('lab-player-table', 'data'),
    [Input('lab-diff-op', 'value'), Input('lab-diff-value', 'value'),
     Input('lab-mins-op', 'value'), Input('lab-mins-value', 'value')],
    State('lab-player-rows-store', 'data'),
    prevent_initial_call=True
)
def filter_backtest_player_table(diff_op, diff_val, mins_op, mins_val, stored_rows):
    """Re-slices the already-computed backtest rows on the Diff/Mins operator
    + value controls. Deliberately separate from render_backtest — that one
    re-runs the (slow, rate-limited) backtest itself, and a filter tweak
    shouldn't trigger that."""
    rows = stored_rows or []
    rows = _apply_numeric_filter(rows, 'diff', diff_op, diff_val)
    rows = _apply_numeric_filter(rows, 'minutes_played', mins_op, mins_val)
    return rows


# --- MODEL LAB: PROJECTION BREAKDOWN ---
XP_PARTS = [('xp_appear', 'Appear'), ('xp_goals', 'Goals'), ('xp_assists', 'Assists'),
            ('xp_cs', 'CS'), ('xp_defcon', 'DEFCON'), ('xp_saves', 'Saves'),
            ('xp_bonus', 'Bonus'), ('xp_gc', 'GC pen')]
XP_PART_COLOURS = {'Appear': '#9e9e9e', 'Goals': COLORS['accent'], 'Assists': '#ff8a65',
                   'CS': COLORS['primary'], 'DEFCON': COLORS['info'], 'Saves': '#7e57c2',
                   'Bonus': COLORS['secondary'], 'GC pen': COLORS['danger']}


@callback(
    [Output('lab-bd-chart', 'figure'), Output('lab-bd-table', 'children')],
    [Input('visit-model-lab', 'data'), Input('lab-bd-position', 'value'),
     Input('lab-bd-team', 'value'), Input('lab-bd-search', 'value')],
    prevent_initial_call=True
)
def render_projection_breakdown(page, position, team, search):
    """Split each next-GW projection into its eight scoring components."""
    _need_visit(page)
    blank = go.Figure()
    blank.update_layout(template='plotly_white', height=320)

    dfa = get_data().get('df_active')
    if dfa is None or dfa.empty or 'xp_goals' not in dfa.columns:
        blank.add_annotation(text="Breakdown appears after the next data refresh",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                             font=dict(size=15, color=COLORS['text_light']))
        return blank, html.P("The component columns are written during a refresh. If this "
                             "persists, the projection pass did not run.",
                             style={'color': COLORS['text_light']})

    d = dfa.copy()
    if position != 'All':
        d = d[d['position'] == position]
    if team != 'All':
        d = d[d['team_name'] == team]
    if search:
        d = d[d['web_name'].str.contains(str(search).strip(), case=False, na=False)]
    d = d[pd.to_numeric(d['proj_pts_next'], errors='coerce').notna()]
    if d.empty:
        blank.add_annotation(text="No players match these filters", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(size=15, color=COLORS['text_light']))
        return blank, html.Div()

    top = d.nlargest(20, 'proj_pts_next').iloc[::-1]
    fig = go.Figure()
    for col, label in XP_PARTS:
        if col not in top.columns:
            continue
        fig.add_trace(go.Bar(
            y=top['web_name'], x=pd.to_numeric(top[col], errors='coerce').fillna(0),
            name=label, orientation='h',
            marker_color=XP_PART_COLOURS.get(label, COLORS['text_light']),
            hovertemplate='%{y}<br>' + label + ': %{x:.2f} pts<extra></extra>'))
    fig.update_layout(barmode='relative', template='plotly_white',
                      height=max(360, 22 * len(top) + 120),
                      xaxis_title='Projected points (next GW), by component',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='center', x=0.5),
                      margin=dict(t=60, b=40, l=110, r=20),
                      font=dict(family=FONT_FAMILY))

    d = d.copy()
    d['fix_effect'] = (pd.to_numeric(d['proj_pts_next'], errors='coerce')
                       - pd.to_numeric(d.get('proj_neutral_gw'), errors='coerce'))
    cols = ['web_name', 'team_name', 'position', 'price', 'proj_pts_next',
            'proj_neutral_gw', 'fix_effect', 'xp_att_mult', 'exp_mins_next',
            'xp_appear', 'xp_goals', 'xp_assists', 'xp_cs', 'xp_defcon',
            'xp_saves', 'xp_bonus', 'xp_gc']
    cols = [c for c in cols if c in d.columns]
    table = dash_table.DataTable(
        data=prepare_table_data(d.nlargest(60, 'proj_pts_next'), cols),
        columns=[
            {'name': 'Player', 'id': 'web_name'},
            {'name': 'Team', 'id': 'team_name'},
            {'name': 'Pos', 'id': 'position'},
            {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Proj', 'id': 'proj_pts_next', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Neutral', 'id': 'proj_neutral_gw', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Fixture +/-', 'id': 'fix_effect', 'type': 'numeric', 'format': {'specifier': '+.2f'}},
            {'name': 'Fix x', 'id': 'xp_att_mult', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Mins', 'id': 'exp_mins_next', 'type': 'numeric', 'format': {'specifier': '.0f'}},
            {'name': 'Appear', 'id': 'xp_appear', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Goals', 'id': 'xp_goals', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Assists', 'id': 'xp_assists', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'CS', 'id': 'xp_cs', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'DEFCON', 'id': 'xp_defcon', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Saves', 'id': 'xp_saves', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Bonus', 'id': 'xp_bonus', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'GC pen', 'id': 'xp_gc', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        ],
        sort_action='native', page_size=20,
        style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
            {'if': {'filter_query': '{fix_effect} > 1', 'column_id': 'fix_effect'},
             'backgroundColor': '#e6fff2'},
            {'if': {'filter_query': '{fix_effect} < -1', 'column_id': 'fix_effect'},
             'backgroundColor': '#fde8ef'},
        ])
    return fig, table


# --- MODEL LAB ---
@callback(
    Output('lab-calibration', 'children'),
    Input('visit-model-lab', 'data'),
    prevent_initial_call=True
)
def render_lab_calibration(page):
    _need_visit(page)
    cal = compute_calibration()
    if not cal:
        return html.P("Nothing scoreable yet. Calibration appears once a logged gameweek "
                      "has finished. The logging is already running in the background.",
                      style={'color': COLORS['text_light']})
    rows = [{'gw': f"GW{r['gw']}", 'n': r['n'], 'mae_model': r['mae_model'],
             'mae_fpl': r['mae_fpl'],
             'edge': (round(r['mae_fpl'] - r['mae_model'], 3)
                      if r['mae_fpl'] is not None else None)}
            for r in cal['per_gw']]
    verdict = ("Model is beating FPL's xP overall — trust the Proj columns."
               if (cal['mae_fpl'] is not None and cal['mae_model'] < cal['mae_fpl'])
               else "FPL's xP is ahead overall — run the sweep and consider its suggested constants.")
    return html.Div([
        dash_table.DataTable(
            data=rows,
            columns=[
                {'name': 'GW', 'id': 'gw'},
                {'name': 'Players', 'id': 'n', 'type': 'numeric'},
                {'name': 'Model MAE', 'id': 'mae_model', 'type': 'numeric'},
                {'name': 'FPL xP MAE', 'id': 'mae_fpl', 'type': 'numeric'},
                {'name': 'Edge vs FPL', 'id': 'edge', 'type': 'numeric',
                 'format': {'specifier': '+.3f'}},
            ],
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
            style_data=TABLE_STYLE_DATA,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                {'if': {'filter_query': '{edge} > 0', 'column_id': 'edge'},
                 'backgroundColor': '#e6fff2'},
                {'if': {'filter_query': '{edge} < 0', 'column_id': 'edge'},
                 'backgroundColor': '#fde8ef'},
            ]),
        html.P([html.Strong(f"Overall: model {cal['mae_model']:.3f} vs FPL "
                            f"{cal['mae_fpl']:.3f} MAE over {cal['gws']} GW(s). "), verdict],
               style={'color': COLORS['text_dark'], 'marginTop': '12px'}),
    ])


@callback(
    Output('lab-sweep-result', 'children'),
    Input('lab-sweep-btn', 'n_clicks'),
    prevent_initial_call=True
)
def run_lab_sweep(n_clicks):
    sweep = run_parameter_sweep()
    if not sweep or not sweep['results']:
        return html.P("Not enough logged data yet — needs at least one finished gameweek "
                      "with stored features.",
                      style={'color': COLORS['text_light'], 'marginTop': '12px'})
    best = sweep['results'][0]
    live = sweep['live']
    cur = sweep['current']
    rows = [{'fdr_ratio': r['fdr_ratio'], 'shrink_k': r['shrink_k'],
             'mae': r['mae'],
             'tag': ('BEST' if r is best else '') +
                    (' LIVE' if (r['fdr_ratio'] == live['fdr_ratio'] and
                                 r['shrink_k'] == live['shrink_k']) else '')}
            for r in sweep['results']]
    advice = []
    if cur and best['mae'] < cur['mae'] - 0.005:
        advice.append(html.P([
            html.Strong("Suggested change: "),
            f"FDR ratio {best['fdr_ratio']}, shrinkage k {best['shrink_k']} would have cut MAE "
            f"from {cur['mae']:.4f} to {best['mae']:.4f} over {sweep['gws']} GW(s). Apply by "
            f"setting env vars FPL_FDR_RATIO={best['fdr_ratio']} and "
            f"FPL_SHRINK_K={best['shrink_k']} on Render, then redeploy."],
            style={'color': COLORS['text_dark'], 'marginTop': '12px',
                   'backgroundColor': '#f0e6f5', 'padding': '10px', 'borderRadius': '6px'}))
    else:
        advice.append(html.P("Your live constants are already at (or within noise of) the "
                             "best tested combination — no change recommended.",
                             style={'color': COLORS['success_text'], 'fontWeight': '600',
                                    'marginTop': '12px'}))
    if sweep['gws'] < 4:
        advice.append(html.P(f"Caution: only {sweep['gws']} gameweek(s) scored — treat this as "
                             f"directional until ~6 GWs are in. Early-season parameter fitting "
                             f"can chase noise.",
                             style={'color': COLORS['warning_text'], 'fontSize': '13px'}))
    return html.Div([
        dash_table.DataTable(
            data=rows,
            columns=[
                {'name': 'FDR Ratio', 'id': 'fdr_ratio', 'type': 'numeric'},
                {'name': 'Shrink k', 'id': 'shrink_k', 'type': 'numeric'},
                {'name': 'MAE', 'id': 'mae', 'type': 'numeric'},
                {'name': '', 'id': 'tag'},
            ],
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
            style_data=TABLE_STYLE_DATA,
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                {'if': {'filter_query': '{tag} contains "BEST"'},
                 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
            ]),
        *advice
    ])


# --- DEADLINE DASHBOARD ---
@callback(
    Output('dd-content', 'children'),
    Input('dd-load-btn', 'n_clicks'),
    State('dd-team-id', 'value'),
    prevent_initial_call=True
)
def run_deadline_check(n_clicks, team_id):
    if not team_id:
        return html.Div([html.P("Enter your team ID.", style={'color': COLORS['text_light'],
                                'textAlign': 'center', 'padding': '30px 0'})], style=CARD_STYLE)
    data = get_data()
    cur = data.get('current_gw')
    nxt = data.get('next_gw')
    if not cur:
        return html.Div([html.P("Season not started — nothing to check yet.",
                                style={'color': COLORS['text_light'], 'textAlign': 'center',
                                       'padding': '30px 0'})], style=CARD_STYLE)
    picks_data = fetch_team_picks(int(team_id), cur['id'])
    if not picks_data or 'picks' not in picks_data:
        return html.Div([html.P("Could not load your squad — check the team ID.",
                                style={'color': COLORS['danger_text']})], style=CARD_STYLE)
    dfa = data.get('df_active', pd.DataFrame())
    squad_ids = [pk['element'] for pk in picks_data['picks']]
    squad = dfa[dfa['id'].isin(squad_ids)].copy()
    cards = []

    # 1. Deadline
    if nxt and nxt.get('deadline_time'):
        dl = datetime.fromisoformat(
            nxt['deadline_time'].replace('Z', '+00:00')
        ).astimezone(ZoneInfo('Europe/London'))

        cards.append(html.Div([
            html.H4(
                f"Next deadline: {nxt['name'].replace('Gameweek ', 'GW')} — "
                f"{dl.strftime('%a %d %b, %H:%M')} UK",
                style={'color': COLORS['primary'], 'margin': 0}
            )
        ], style={**CARD_STYLE, 'backgroundColor': '#f0e6f5'}))

    # 2. Availability flags in squad
    flagged = squad[pd.to_numeric(squad.get('avail_pct'), errors='coerce').fillna(100) < 100]
    if len(flagged) > 0:
        items = [html.Li(f"{r.web_name} — {r.avail_pct:.0f}% ({r.news or 'no detail'})",
                         style={'marginBottom': '4px'}) for r in flagged.itertuples()]
        cards.append(html.Div([
            html.H4(f"\u26a0 Flagged players ({len(flagged)})", style={'color': COLORS['danger_text'],
                    'marginBottom': '8px'}),
            html.Ul(items, style={'paddingLeft': '18px', 'margin': 0})
        ], style=CARD_STYLE))
    else:
        cards.append(html.Div([html.P("\u2713 No availability flags in your squad.",
                              style={'color': COLORS['success_text'], 'fontWeight': '600', 'margin': 0})],
                              style=CARD_STYLE))

    # 3. Captain EV + ceiling from YOUR squad
    if 'proj_pts_next' in squad.columns and len(squad) > 0:
        ev = squad.nlargest(3, 'proj_pts_next')
        ceil = squad.nlargest(3, 'haul_pct') if 'haul_pct' in squad.columns else ev
        cards.append(html.Div([
            html.H4("Captaincy", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P([html.Strong("Protect (EV): "),
                    ',  '.join(f"{r.web_name} ({r.proj_pts_next:.1f})" for r in ev.itertuples())],
                   style={'marginBottom': '6px'}),
            html.P([html.Strong("Chase (ceiling): "),
                    ',  '.join(f"{r.web_name} ({r.haul_pct:.0f}% haul)" for r in ceil.itertuples())],
                   style={'marginBottom': '6px'}),
            html.P("Protecting a lead? Take EV. Chasing? The doubled captain is your variance lever.",
                   style={'color': COLORS['text_light'], 'fontSize': '13px', 'margin': 0}),
        ], style=CARD_STYLE))

    # 4. XI vs optimal + bench order
    try:
        players = []
        for r in squad.itertuples():
            proj = 0.0 if pd.isna(r.proj_pts_next) else float(r.proj_pts_next)
            sec = 1.0 if pd.isna(getattr(r, 'start_rate', np.nan)) else float(r.start_rate) / 100
            players.append({'id': int(r.id), 'name': r.web_name, 'position': r.position,
                            'proj': proj, 'order_score': proj * max(sec, 0.3)})
        if len(players) >= 11:
            xi, bench = pick_best_xi(players)
            xi_total = sum(pl['proj'] for pl in xi)
            bench_gk = [pl for pl in bench if pl['position'] == 'GKP']
            bench_out = sorted([pl for pl in bench if pl['position'] != 'GKP'],
                               key=lambda pl: -pl['order_score'])
            cards.append(html.Div([
                html.H4(f"Optimal XI projects {xi_total:.1f} pts", style={'color': COLORS['primary'],
                        'marginBottom': '8px'}),
                html.P([html.Strong("XI: "), ', '.join(pl['name'] for pl in xi)],
                       style={'marginBottom': '6px'}),
                html.P([html.Strong("Bench order: "),
                        '  \u2192  '.join(f"{i}. {pl['name']}"
                                          for i, pl in enumerate(bench_gk + bench_out, 1))],
                       style={'margin': 0}),
            ], style=CARD_STYLE))
    except Exception as _e:
        print(f"  Deadline XI check failed: {_e}")

    # 5. Price risk tonight
    if 'price_change_likelihood' in squad.columns:
        risk = squad[squad['price_change_likelihood'] <= -40]
        if len(risk) > 0:
            cards.append(html.Div([
                html.H4("Price-fall risk in your squad", style={'color': COLORS['warning_text'],
                        'marginBottom': '8px'}),
                html.P(', '.join(f"{r.web_name} ({r.price_change_likelihood:.0f})"
                                 for r in risk.itertuples()), style={'margin': 0})
            ], style=CARD_STYLE))

    return html.Div(cards)


# --- PRICE ALERTS (squad-aware) ---
@callback(
    Output('pa-result', 'children'),
    Input('pa-load-btn', 'n_clicks'),
    State('pa-team-id', 'value'),
    prevent_initial_call=True
)
def check_price_alerts(n_clicks, team_id):
    if not team_id:
        return html.P("Enter your team ID.", style={'color': COLORS['text_light'], 'marginTop': '10px'})
    data = get_data()
    cur = data.get('current_gw')
    if not cur:
        return html.P("Squads load after the GW1 deadline.",
                      style={'color': COLORS['text_light'], 'marginTop': '10px'})
    picks_data = fetch_team_picks(int(team_id), cur['id'])
    if not picks_data or 'picks' not in picks_data:
        return html.P("Could not load that squad — check the team ID.",
                      style={'color': COLORS['danger_text'], 'marginTop': '10px'})
    squad_ids = {pk['element'] for pk in picks_data['picks']}
    dfa = data.get('df_active', pd.DataFrame())
    if dfa.empty or 'price_change_likelihood' not in dfa.columns:
        return html.P("Price data still loading.", style={'color': COLORS['text_light']})

    mine = dfa[dfa['id'].isin(squad_ids)]
    fall_risk = mine[mine['price_change_likelihood'] <= -40].sort_values('price_change_likelihood')
    # Buy-before-rise: strong projections you DON'T own, near a rise
    others = dfa[~dfa['id'].isin(squad_ids)]
    rise_soon = others[(others['price_change_likelihood'] >= 40)]
    rise_soon = rise_soon.nlargest(8, 'proj_pts_5')

    def _mini_table(frame, extra_col, extra_name):
        cols = ['web_name', 'team_name', 'position', 'price', extra_col, 'proj_pts_5', 'own_delta_7d']
        return dash_table.DataTable(
            data=prepare_table_data(frame, cols),
            columns=[
                {'name': 'Player', 'id': 'web_name'}, {'name': 'Team', 'id': 'team_name'},
                {'name': 'Pos', 'id': 'position'},
                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {'name': extra_name, 'id': extra_col, 'type': 'numeric', 'format': {'specifier': '.0f'}},
                {'name': 'Proj Next 5', 'id': 'proj_pts_5', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {'name': 'Own \u03947d', 'id': 'own_delta_7d', 'type': 'numeric', 'format': {'specifier': '+.1f'}},
            ],
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
        )

    blocks = []
    if len(fall_risk) > 0:
        blocks.append(html.H4(f"\u26a0 Fall risk in YOUR squad ({len(fall_risk)})",
                              style={'color': COLORS['danger_text'], 'margin': '16px 0 8px 0'}))
        blocks.append(_mini_table(fall_risk, 'price_change_likelihood', 'Fall Risk'))
    else:
        blocks.append(html.P("\u2713 No imminent fall risk in your squad.",
                             style={'color': COLORS['success_text'], 'fontWeight': '600', 'marginTop': '12px'}))
    if len(rise_soon) > 0:
        blocks.append(html.H4("Rising soon (you don't own — buy before the price does)",
                              style={'color': COLORS['success_text'], 'margin': '16px 0 8px 0'}))
        blocks.append(_mini_table(rise_soon, 'price_change_likelihood', 'Rise Score'))
    return html.Div(blocks)


# --- SQUAD BUILDER chip-target options ---
@callback(
    Output('sq-chip-gw', 'options'),
    Input('visit-squad-builder', 'data'),
    prevent_initial_call=True
)
def populate_chip_gw_options(page):
    _need_visit(page)
    data = get_data()
    anchor_gw = data.get('fixture_anchor_gw')
    if anchor_gw is None:
        cur = data.get('current_gw')
        anchor_gw = cur['id'] if cur else 0
    gws = [g for g in range(anchor_gw + 1, anchor_gw + 11) if g <= 38]
    # Flag likely DGWs from fixture counts so the dropdown self-documents
    fixtures_data = data.get('fixtures_data', [])
    counts = {}
    for f in fixtures_data:
        g = f.get('event')
        if g in gws:
            counts[g] = counts.get(g, 0) + 1
    opts = []
    for g in gws:
        n_fix = counts.get(g, 0)
        tag = ' (DGW!)' if n_fix > 10 else (' (blanks)' if 0 < n_fix < 10 else '')
        opts.append({'label': f'GW{g}{tag}', 'value': g})
    return opts


# --- SQUAD BUILDER ---
@callback(
    Output('sq-results', 'children'),
    Input('sq-build-btn', 'n_clicks'),
    [State('sq-budget', 'value'),
     State('sq-objective', 'value'),
     State('sq-must-include', 'value'),
     State('sq-must-exclude', 'value'),
     State('sq-chip-gw', 'value')],
    prevent_initial_call=True
)
def build_squad(n_clicks, budget, objective, must_include, must_exclude, chip_gw):
    import traceback
    try:
        data = get_data()
        df_now = data['df_active'].copy()

        # Chip-target emphasis: add each player's projection in the target
        # GW to the objective, so the wildcard draft is pulled toward squads
        # that peak (bench included) exactly when you plan to Bench Boost.
        if chip_gw and 'proj_neutral_gw' in df_now.columns:
            try:
                lookup = build_gw_fixture_lookup(
                    data.get('fixtures_data', []), data.get('teams_df', pd.DataFrame()),
                    [int(chip_gw)]).get(int(chip_gw), {})
                df_now['chip_gw_proj'] = [
                    project_player_gw(
                        0 if pd.isna(r.proj_neutral_gw) else r.proj_neutral_gw,
                        r.position, r.team, lookup)
                    for r in df_now.itertuples()]
                base_obj = objective if objective in df_now.columns else 'ppg'
                df_now['chip_weighted'] = (
                    pd.to_numeric(df_now[base_obj], errors='coerce').fillna(0) +
                    df_now['chip_gw_proj'])
                objective = 'chip_weighted'
            except Exception as _e:
                print(f"  Chip emphasis failed (using base objective): {_e}")

        result = build_optimal_squad(
            df_now,
            budget=budget or 83,
            objective=objective or 'ppg',
            must_include=must_include or [],
            must_exclude=must_exclude or [],
            min_minutes=0
        )

        if result is None:
            return html.Div([
                html.Div([
                    html.P(
                        "⚠️ No feasible squad found. Try raising the budget or removing pinned players.",
                        style={'color': COLORS['danger_text'], 'fontSize': '15px', 'textAlign': 'center', 'margin': '0'}
                    )
                ], style=CARD_STYLE)
            ])

        obj_col = objective if objective in result.columns else 'ppg'
        obj_labels = {
            'ppg': 'Points Per Game',
            'form': 'Form',
            'expected_goal_involvements': 'xGI',
            'total_points': 'Total Points',
            'blended': 'Blended Score',
            'proj_pts_next': 'Projected Points (next GW)',
            'proj_pts_5': 'Projected Points (next 5 GWs)',
            'proj_pts_8': 'Projected Points (next 8 GWs)',
            'chip_weighted': 'Chip-Weighted Projection',
        }
        obj_label = obj_labels.get(objective, objective)

        total_cost = result['price'].sum()
        remaining = budget - total_cost
        total_score = result[obj_col].sum()
        teams_used = result['team_name'].nunique()

        # Summary cards
        summary = html.Div([
            html.Div([build_stat_card("Total Cost", f"£{total_cost:.1f}m", f"£{remaining:.1f}m remaining")],
                     style={'flex': '1', 'minWidth': '180px', 'padding': '0 10px'}),
            html.Div([build_stat_card(obj_label, f"{total_score:,.0f}", "Combined squad total")],
                     style={'flex': '1', 'minWidth': '180px', 'padding': '0 10px'}),
            html.Div([build_stat_card("Clubs Used", str(teams_used), "Max 3 players per club")],
                     style={'flex': '1', 'minWidth': '180px', 'padding': '0 10px'}),
            html.Div([build_stat_card("Squad Size", "15", "2 GKP · 5 DEF · 5 MID · 3 FWD")],
                     style={'flex': '1', 'minWidth': '180px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 24px -10px'})

        # Squad cards by position
        pos_order = ['GKP', 'DEF', 'MID', 'FWD']
        pos_colors = {
            'GKP': '#e5a823',
            'DEF': COLORS['primary'],
            'MID': COLORS['accent'],
            'FWD': COLORS['info'],
        }

        pos_cards = []
        for pos in pos_order:
            pos_df = result[result['position'] == pos].sort_values(obj_col, ascending=False)
            if pos_df.empty:
                continue
            rows = []
            for _, p in pos_df.iterrows():
                score_val = p.get(obj_col, 0)
                if pd.isna(score_val):
                    score_val = 0
                rows.append(html.Div([
                    html.Div([
                        html.Span(pos, style={
                            'backgroundColor': pos_colors[pos],
                            'color': 'white' if pos != 'GKP' else COLORS['primary'],
                            'padding': '2px 8px', 'borderRadius': '4px',
                            'fontSize': '11px', 'fontWeight': '700', 'marginRight': '8px'
                        }),
                        html.Span(p['web_name'],
                                  style={'fontWeight': '600', 'fontSize': '15px', 'color': COLORS['text_dark']}),
                    ]),
                    html.Div([
                        html.Span(p['team_name'],
                                  style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginRight': '10px'}),
                        html.Span(f"£{p['price']:.1f}m",
                                  style={'color': COLORS['primary'], 'fontWeight': '600', 'fontSize': '14px',
                                         'marginRight': '10px'}),
                        html.Span(f"{obj_label}: {score_val:.0f}",
                                  style={'color': COLORS['text_light'], 'fontSize': '13px'}),
                    ])
                ], style={
                    'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                    'padding': '10px 0', 'borderBottom': '1px solid #f0f0f0'
                }))
            pos_cards.append(html.Div([
                html.H4(f"{pos}  ({len(pos_df)})",
                        style={'color': pos_colors[pos], 'marginBottom': '12px', 'fontWeight': '700'}),
                html.Div(rows)
            ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '300px'}))

        squad_display = html.Div([
            html.Div([
                html.H3("Optimal Squad", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
                html.P(f"Optimised for: {obj_label}", style={'color': COLORS['text_light']})
            ], style={'marginBottom': '20px'}),
            html.Div(pos_cards, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'})
        ])

        # Score breakdown bar chart
        result_plot = result.assign(
            pos_order=result['position'].map({'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3})
        ).sort_values(['pos_order', obj_col], ascending=[True, False])

        bar_fig = px.bar(
            result_plot, x='web_name', y=obj_col, color='position',
            text=result_plot[obj_col].round(1),
            hover_data=['team_name', 'price', 'ppg', 'form'],
            color_discrete_map={
                'GKP': '#e5a823', 'DEF': COLORS['primary'],
                'MID': COLORS['accent'], 'FWD': COLORS['info']
            }
        )
        bar_fig.update_traces(textposition='outside')
        bar_fig.update_layout(
            template='plotly_white', height=420,
            xaxis_tickangle=-45, xaxis_title='',
            yaxis_title=obj_label,
            yaxis=dict(range=[0, result_plot[obj_col].max() * 1.22]),
            font=dict(family=FONT_FAMILY),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )

        # Club distribution chart
        club_counts = result['team_name'].value_counts().reset_index()
        club_counts.columns = ['team_name', 'count']
        club_fig = px.bar(
            club_counts, x='team_name', y='count',
            color='count', text='count',
            color_continuous_scale=['#c8e6c9', COLORS['primary']]
        )
        club_fig.update_traces(textposition='outside')
        club_fig.update_layout(
            template='plotly_white', height=320,
            xaxis_tickangle=-45, xaxis_title='', yaxis_title='Players Selected',
            yaxis=dict(range=[0, club_counts['count'].max() + 0.8]),
            coloraxis_showscale=False, showlegend=False,
            font=dict(family=FONT_FAMILY)
        )

        # Full table
        table_cols_config = [
            {'name': 'Player', 'id': 'web_name'},
            {'name': 'Team', 'id': 'team_name'},
            {'name': 'Pos', 'id': 'position'},
            {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Points', 'id': 'total_points', 'type': 'numeric'},
            {'name': 'PPG', 'id': 'ppg', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Form', 'id': 'form', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'xGI', 'id': 'expected_goal_involvements', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Own%', 'id': 'ownership', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        ]
        if objective == 'blended' and 'blended' in result.columns:
            table_cols_config.append(
                {'name': 'Blended', 'id': 'blended', 'type': 'numeric', 'format': {'specifier': '.2f'}}
            )

        display_col_ids = [c['id'] for c in table_cols_config if c['id'] in result.columns]
        table_data = prepare_table_data(
            result.assign(
                pos_order=result['position'].map({'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3})
            ).sort_values(['pos_order', obj_col], ascending=[True, False]),
            display_col_ids
        )

        full_table = html.Div([
            html.H4("Full Squad Details", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
            dash_table.DataTable(
                data=table_data,
                columns=table_cols_config,
                sort_action='native',
                style_cell=TABLE_STYLE_CELL,
                style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                    {'if': {'filter_query': '{position} = "GKP"', 'column_id': 'position'},
                     'backgroundColor': '#fff8e1', 'fontWeight': '600'},
                    {'if': {'filter_query': '{position} = "DEF"', 'column_id': 'position'}, 'color': COLORS['primary'],
                     'fontWeight': '600'},
                    {'if': {'filter_query': '{position} = "MID"', 'column_id': 'position'}, 'color': COLORS['accent'],
                     'fontWeight': '600'},
                    {'if': {'filter_query': '{position} = "FWD"', 'column_id': 'position'}, 'color': COLORS['info_text'],
                     'fontWeight': '600'},
                ]
            )
        ], style=CARD_STYLE)

        return html.Div([
            summary,
            squad_display,
            html.Div([
                html.H3("Score Breakdown", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                html.P(f"Each player's {obj_label} contribution to the squad total.",
                       style={'color': COLORS['text_light']}),
                dcc.Graph(figure=bar_fig, config={'displayModeBar': False})
            ], style=CARD_STYLE),
            html.Div([
                html.H3("Club Distribution", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                html.P("Players selected per club (max 3 enforced by solver).", style={'color': COLORS['text_light']}),
                dcc.Graph(figure=club_fig, config={'displayModeBar': False})
            ], style=CARD_STYLE),
            full_table,
        ])

    except Exception as e:
        return html.Div([
            html.Div([
                html.P(f"Error: {str(e)}", style={'color': COLORS['danger_text'], 'fontWeight': '600'}),
                html.Pre(traceback.format_exc(),
                         style={'fontSize': '12px', 'color': COLORS['text_light'], 'whiteSpace': 'pre-wrap'})
            ], style=CARD_STYLE)
        ])


# =============================================================================
# MY SQUAD CALLBACK
# =============================================================================

STATUS_LABELS = {
    'a': ('✓ Fit',        COLORS['success_text']),
    'd': ('⚠ Doubt',      COLORS['warning_text']),
    'i': ('✗ Injured',    COLORS['danger_text']),
    's': ('✗ Suspended',  COLORS['danger_text']),
    'n': ('✗ N/A',        COLORS['danger_text']),
    'u': ('✗ N/A',        COLORS['danger_text']),
}


@callback(
    Output('my-squad-content', 'children'),
    Input('squad-load-btn', 'n_clicks'),
    State('squad-team-id-input', 'value'),
    prevent_initial_call=True
)
def load_my_squad(n_clicks, team_id):
    if not team_id:
        return html.P("Please enter your FPL team ID above.",
                      style={'color': COLORS['text_light'], 'textAlign': 'center', 'padding': '40px 0'})

    team_id = int(team_id)

    entry = fetch_team_entry(team_id)
    if not entry:
        return html.Div([html.Div([
            html.P(f"Could not load team ID {team_id}. Please check the ID and try again.",
                   style={'color': COLORS['danger_text'], 'fontWeight': '600'})
        ], style=CARD_STYLE)])

    data = get_data()
    current_gw_info = data.get('current_gw')
    # Picks only exist once a deadline has passed. Pre-season there is no
    # current gameweek at all, so asking for GW1 picks just 404s.
    if not current_gw_info:
        target = data.get('next_gw_num', 1)
        return html.Div([html.Div([
            html.P(f"The season hasn't started yet — squads become available "
                   f"after the GW{target} deadline.",
                   style={'color': COLORS['text_light'], 'fontWeight': '600',
                          'textAlign': 'center', 'padding': '40px 0'})
        ], style=CARD_STYLE)])
    gw_num = current_gw_info['id']
    df_all = data.get('df', pd.DataFrame())
    fixtures_data = data.get('fixtures_data', [])
    fixture_difficulty = data.get('fixture_difficulty', {})

    picks_data = fetch_team_picks(team_id, gw_num)
    if not picks_data:
        return html.Div([html.Div([
            html.P("Could not load squad picks. The gameweek may not have started yet, "
                   "or the team ID is incorrect.",
                   style={'color': COLORS['danger_text'], 'fontWeight': '600'})
        ], style=CARD_STYLE)])

    picks = picks_data.get('picks', [])
    entry_history = picks_data.get('entry_history', {})
    active_chip = picks_data.get('active_chip')

    pick_map = {p['element']: p for p in picks}
    player_ids = list(pick_map.keys())

    squad_df = df_all[df_all['id'].isin(player_ids)].copy()
    if squad_df.empty:
        return html.Div([html.Div([
            html.P("No player data found for this squad. Try refreshing.",
                   style={'color': COLORS['danger_text']})
        ], style=CARD_STYLE)])

    squad_df['pick_position'] = squad_df['id'].map(lambda x: pick_map[x]['position'])
    squad_df['is_captain']    = squad_df['id'].map(lambda x: pick_map[x].get('is_captain', False))
    squad_df['is_vice']       = squad_df['id'].map(lambda x: pick_map[x].get('is_vice_captain', False))
    try:
        _prices = estimate_selling_prices(squad_df, fetch_entry_transfers(team_id),
                                          fetch_entry_chips(team_id))
    except Exception as _e:
        print(f"  Selling price calculation failed (non-fatal): {_e}")
        _prices = {}
    squad_df['selling_price'] = squad_df['id'].map(
        lambda x: _prices.get(int(x), (None, None))[0])
    squad_df['purchase_price'] = squad_df['id'].map(
        lambda x: _prices.get(int(x), (None, None))[1])
    squad_df['selling_price'] = squad_df['selling_price'].fillna(squad_df['price'])
    squad_df['form'] = pd.to_numeric(squad_df['form'], errors='coerce').fillna(0)
    squad_df['avg_fdr_5'] = squad_df['team'].map(
        lambda x: fixture_difficulty.get(x, {}).get('avg_fdr', 3.0)
    )
    squad_df['fixture_string'] = squad_df['team'].map(
        lambda x: fixture_difficulty.get(x, {}).get('fixture_string', '')
    )

    squad_team_ids = squad_df['team'].unique().tolist()
    bgw_dgw_flags = get_squad_fixture_flags(squad_team_ids, fixtures_data, gw_num, num_gws=5)
    squad_df['bgw_dgw'] = squad_df['team'].map(lambda x: bgw_dgw_flags.get(x, ''))
    squad_df['status_label'] = squad_df['status'].map(
        lambda x: STATUS_LABELS.get(x, ('? Unknown', COLORS['text_light']))[0]
    )
    squad_df = squad_df.sort_values('pick_position').reset_index(drop=True)

    # --- Manager info card ---
    mgr_name     = (entry.get('player_first_name', '') + ' ' + entry.get('player_last_name', '')).strip()
    team_name_e  = entry.get('name', 'My Team')
    overall_rank = entry.get('summary_overall_rank')
    total_pts    = entry.get('summary_overall_points', 0)
    gw_pts       = entry_history.get('points', 0)
    bank_m       = entry_history.get('bank', entry.get('last_deadline_bank', 0)) / 10
    team_value_m = entry_history.get('value', entry.get('last_deadline_value', 0)) / 10
    transfer_cost = entry_history.get('event_transfers_cost', 0)
    chip_str     = f" · Chip: {active_chip.upper()}" if active_chip else ''
    rank_str     = f"{overall_rank:,}" if overall_rank else 'N/A'

    manager_card = html.Div([
        html.Div([
            html.Div([
                html.H3(team_name_e,
                        style={'color': COLORS['primary'], 'margin': '0 0 4px 0', 'fontSize': '22px'}),
                html.P(f"{mgr_name}{chip_str}",
                       style={'color': COLORS['text_light'], 'margin': '0', 'fontSize': '14px'}),
            ], style={'flex': '2'}),
            *[
                html.Div([
                    html.Span(label, style={'color': COLORS['text_light'], 'fontSize': '12px', 'display': 'block'}),
                    html.Span(value, style={'color': color, 'fontSize': '22px', 'fontWeight': '700'}),
                ], style={'flex': '1', 'textAlign': 'center'})
                for label, value, color in [
                    ("GW Points",    f"{gw_pts}" + (f" (-{transfer_cost})" if transfer_cost else ''), COLORS['primary']),
                    ("Overall Rank", rank_str,                 COLORS['primary']),
                    ("Team Value",   f"£{team_value_m:.1f}m", COLORS['primary']),
                    ("In the Bank",  f"£{bank_m:.1f}m",       COLORS['success_text']),
                    ("Total Points", f"{total_pts}",           COLORS['primary']),
                ]
            ]
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'gap': '20px'}),
    ], style=CARD_STYLE)

    # --- Squad table builder ---
    pos_colors = {'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}

    def build_squad_section(title, rows_df):
        rows = []
        for _, r in rows_df.iterrows():
            status_text, status_color = STATUS_LABELS.get(r.get('status', 'a'), ('? Unknown', COLORS['text_light']))
            fdr = r.get('avg_fdr_5')
            fdr_color = (COLORS['success_text'] if fdr and fdr <= 2.5
                         else COLORS['warning_text'] if fdr and fdr <= 3.5
                         else COLORS['danger_text'])

            cap_badge = None
            if r.get('is_captain'):
                cap_badge = html.Span('C', style={
                    'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                    'borderRadius': '50%', 'width': '20px', 'height': '20px',
                    'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'fontWeight': '800', 'fontSize': '11px', 'marginLeft': '6px'
                })
            elif r.get('is_vice'):
                cap_badge = html.Span('V', style={
                    'backgroundColor': '#ccc', 'color': COLORS['primary'],
                    'borderRadius': '50%', 'width': '20px', 'height': '20px',
                    'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'fontWeight': '800', 'fontSize': '11px', 'marginLeft': '6px'
                })

            bgw_dgw = r.get('bgw_dgw', '')
            bgw_cell = []
            if 'BGW' in bgw_dgw:
                bgw_cell.append(html.Span('BGW', style={
                    'backgroundColor': '#d0d0d0', 'color': '#555',
                    'padding': '2px 6px', 'borderRadius': '10px',
                    'fontSize': '11px', 'fontWeight': '600', 'marginRight': '4px'
                }))
            if 'DGW' in bgw_dgw:
                bgw_cell.append(html.Span('DGW', style={
                    'backgroundColor': COLORS['success'], 'color': COLORS['primary'],
                    'padding': '2px 6px', 'borderRadius': '10px',
                    'fontSize': '11px', 'fontWeight': '600', 'marginRight': '4px'
                }))

            rows.append(html.Tr([
                html.Td(
                    html.Span(r.get('position', ''),
                              style={'color': pos_colors.get(r.get('position', ''), '#333'),
                                     'fontWeight': '700', 'fontSize': '12px'}),
                    style={'padding': '10px 12px', 'width': '50px'}
                ),
                html.Td(
                    html.Div(
                        [html.Span(r.get('web_name', ''), style={'fontWeight': '600'}),
                         cap_badge or ''],
                        style={'display': 'flex', 'alignItems': 'center'}
                    ),
                    style={'padding': '10px 12px'}
                ),
                html.Td(r.get('team_name', ''),
                        style={'padding': '10px 12px', 'color': COLORS['text_light'], 'fontSize': '13px'}),
                html.Td([
                    html.Div(f"£{r.get('selling_price', r.get('price', 0)):.1f}m"),
                    html.Div(f"bought £{r['purchase_price']:.1f}m",
                             style={'fontSize': '11px', 'color': COLORS['text_light']})
                    if pd.notna(r.get('purchase_price')) and
                    abs(r['purchase_price'] - r.get('selling_price', 0)) >= 0.05 else None,
                ], style={'padding': '10px 12px'}),
                html.Td(f"{r.get('form', 0):.1f}", style={'padding': '10px 12px'}),
                html.Td(
                    html.Span(f"{fdr:.2f}" if fdr else 'N/A',
                              style={'color': fdr_color, 'fontWeight': '700'}),
                    style={'padding': '10px 12px'}
                ),
                html.Td(r.get('fixture_string', ''),
                        style={'padding': '10px 12px', 'fontSize': '12px',
                               'color': COLORS['text_light']}),
                html.Td(
                    html.Span(status_text,
                              style={'color': status_color, 'fontWeight': '600', 'fontSize': '12px'}),
                    style={'padding': '10px 12px'}
                ),
                html.Td(html.Div(bgw_cell), style={'padding': '10px 12px'}),
            ], style={
                'borderBottom': '1px solid #e0e0e0',
                'backgroundColor': '#fff8e1' if r.get('pick_position', 0) > 11 else 'white'
            }))

        header = html.Tr([
            html.Th(col, style={**TABLE_STYLE_HEADER, 'padding': '10px 12px'})
            for col in ['Pos', 'Player', 'Club', 'Selling Price', 'Form',
                        'Avg FDR', 'Next 5 Fixtures', 'Status', 'Flags']
        ])

        return html.Div([
            html.H4(title, style={'color': COLORS['primary'], 'marginBottom': '12px'}),
            html.Div([
                html.Table(
                    [html.Thead(header), html.Tbody(rows)],
                    style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'}
                )
            ], style={'overflowX': 'auto'})
        ], style=CARD_STYLE)

    starters_section = build_squad_section(
        f"Starting XI as of: GW{gw_num}",
        squad_df[squad_df['pick_position'] <= 11]
    )
    bench_section = build_squad_section(
        "Bench",
        squad_df[squad_df['pick_position'] > 11]
    )

    # --- Fitness concerns ---
    concern_df = squad_df[squad_df['status'] != 'a']
    injury_section = None
    if len(concern_df) > 0:
        items = []
        for _, r in concern_df.iterrows():
            status_text, status_color = STATUS_LABELS.get(r.get('status', 'a'), ('? Unknown', COLORS['text_light']))
            news = r.get('news', '')
            items.append(html.Div([
                html.Span(r['web_name'],
                          style={'fontWeight': '700', 'color': COLORS['primary']}),
                html.Span(f" ({r['team_name']}, {r['position']}) — ",
                          style={'color': COLORS['text_light']}),
                html.Span(status_text,
                          style={'color': status_color, 'fontWeight': '600'}),
                html.Span(f" · {news}" if news else '',
                          style={'color': COLORS['text_light'], 'fontSize': '13px'}),
            ], style={'marginBottom': '8px'}))

        injury_section = html.Div([
            html.H4("Fitness Concerns",
                    style={'color': COLORS['danger_text'], 'marginBottom': '12px'}),
            html.Div(items)
        ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORS["danger"]}'})

    # --- Lineup advisor: optimal XI vs your picks + projected bench order ---
    lineup_section = html.Div()
    try:
        dfa = data.get('df_active', pd.DataFrame())
        squad_ids = [pk['element'] for pk in picks_data['picks']]
        adv = dfa[dfa['id'].isin(squad_ids)].copy()
        if len(adv) >= 11 and 'proj_pts_next' in adv.columns:
            players = []
            for r in adv.itertuples():
                proj = 0.0 if pd.isna(r.proj_pts_next) else float(r.proj_pts_next)
                sec = 1.0 if pd.isna(getattr(r, 'start_rate', np.nan)) else float(r.start_rate) / 100
                players.append({'id': int(r.id), 'name': r.web_name, 'position': r.position,
                                'proj': proj, 'order_score': proj * max(sec, 0.3)})
            xi, bench = pick_best_xi(players)
            xi_ids = {pl['id'] for pl in xi}
            current_xi_ids = {pk['element'] for pk in picks_data['picks']
                              if pk.get('multiplier', 0) > 0 or pk.get('position', 16) <= 11}
            promote = [pl for pl in xi if pl['id'] not in current_xi_ids]
            demote_ids = current_xi_ids - xi_ids
            demote = [pl for pl in players if pl['id'] in demote_ids]

            bench_gk = [pl for pl in bench if pl['position'] == 'GKP']
            bench_out = sorted([pl for pl in bench if pl['position'] != 'GKP'],
                               key=lambda pl: -pl['order_score'])
            bench_order = bench_gk + bench_out

            advice = []
            if promote:
                for pin, pout in zip(promote, demote):
                    advice.append(html.P([
                        "\u2192 Start ", html.Strong(pin['name']),
                        f" ({pin['proj']:.1f} proj) over ",
                        html.Strong(pout['name']), f" ({pout['proj']:.1f} proj)"],
                        style={'color': COLORS['text_dark'], 'marginBottom': '6px'}))
            else:
                advice.append(html.P("\u2713 Your XI already matches the projected-optimal lineup.",
                                     style={'color': COLORS['success_text'], 'fontWeight': '600',
                                            'marginBottom': '6px'}))
            advice.append(html.P([html.Strong("Recommended bench order: "),
                                  '  \u2192  '.join(
                                      f"{i}. {pl['name']}" for i, pl in enumerate(bench_order, 1))],
                                 style={'color': COLORS['text_dark'], 'marginTop': '10px'}))
            advice.append(html.P("Bench order decides which auto-subs you get — highest "
                                 "projection x start-security first (bench GK is fixed in slot 1).",
                                 style={'color': COLORS['text_light'], 'fontSize': '13px'}))
            lineup_section = html.Div([
                html.H3("Lineup Advisor", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
                *advice
            ], style=CARD_STYLE)
    except Exception as _e:
        print(f"  Lineup advisor failed (non-fatal): {_e}")

    try:
        rank_card = build_rank_card(fetch_entry_history(team_id), overall_rank)
    except Exception as _e:
        print(f"  Rank card failed (non-fatal): {_e}")
        rank_card = html.Div()

    return html.Div([
        rank_card,
        manager_card,
        lineup_section,
        starters_section,
        bench_section,
        injury_section or html.Div(),
    ])


# =============================================================================
# RUN
# =============================================================================

# =============================================================================
# TRANSFER PLANNER CALLBACKS
# =============================================================================

@callback(
    [Output('tp-player-out', 'options'), Output('tp-player-in', 'options')],
    Input('visit-transfer-planner', 'data'),
    prevent_initial_call=True
)
def populate_transfer_planner_options(page):
    """Refresh the player pools whenever the page is visited (cheap, and
    avoids serving a boot-time snapshot after a data refresh)."""
    _need_visit(page)
    data = get_data()
    dfa = data.get('df_active', pd.DataFrame())
    if dfa.empty:
        return [], []
    pool = dfa.sort_values(['team_name', 'web_name'])
    options = [
        {'label': f"{r.web_name} ({r.team_name} {r.position} \u00a3{r.price:.1f}m)",
         'value': int(r.id)}
        for r in pool.itertuples()
    ]
    return options, options


def _tp_stat_row(label, out_val, in_val):
    return html.Tr([
        html.Td(label, style={'padding': '8px 12px', 'fontWeight': '600',
                              'color': COLORS['text_light'], 'fontSize': '13px'}),
        html.Td(out_val, style={'padding': '8px 12px', 'textAlign': 'center'}),
        html.Td(in_val, style={'padding': '8px 12px', 'textAlign': 'center'}),
    ], style={'borderBottom': '1px solid #eee'})


@callback(
    Output('tp-result', 'children'),
    [Input('tp-player-out', 'value'), Input('tp-player-in', 'value'),
     Input('tp-horizon', 'value'), Input('tp-hit', 'value')]
)
def update_transfer_gain(out_id, in_id, horizon, hit):
    if not out_id or not in_id:
        return html.Div([
            html.P("Select the player you'd sell and the player you'd buy.",
                   style={'color': COLORS['text_light'], 'textAlign': 'center',
                          'padding': '30px 0'})
        ], style=CARD_STYLE)

    if out_id == in_id:
        return html.Div([
            html.P("That's the same player twice \u2014 the projected gain of doing nothing is reassuringly zero.",
                   style={'color': COLORS['text_light'], 'textAlign': 'center', 'padding': '30px 0'})
        ], style=CARD_STYLE)

    data = get_data()
    dfa = data.get('df_active', pd.DataFrame())
    out_rows = dfa[dfa['id'] == out_id]
    in_rows = dfa[dfa['id'] == in_id]
    if out_rows.empty or in_rows.empty:
        return html.Div([html.P("Player data not found \u2014 try reloading the page.",
                                style={'color': COLORS['danger_text']})], style=CARD_STYLE)
    p_out, p_in = out_rows.iloc[0], in_rows.iloc[0]

    proj_col = 'proj_pts_next' if horizon == 'next' else 'proj_pts_5'
    horizon_label = 'next gameweek' if horizon == 'next' else 'next 5 gameweeks'
    hit = int(hit or 0)

    def _proj(row):
        v = row.get(proj_col)
        return 0.0 if pd.isna(v) else float(v)

    out_proj, in_proj = _proj(p_out), _proj(p_in)
    raw_gain = in_proj - out_proj
    net_gain = raw_gain - hit

    if net_gain >= 2:
        verdict, colour = f"Worth it: projected +{net_gain:.1f} pts over the {horizon_label}", COLORS['success']
    elif net_gain > 0:
        verdict, colour = f"Marginal: projected +{net_gain:.1f} pts over the {horizon_label}", COLORS['warning']
    else:
        verdict, colour = f"Don't: projected {net_gain:+.1f} pts over the {horizon_label}", COLORS['danger']

    position_warning = None
    if p_out['position'] != p_in['position']:
        position_warning = html.P(
            f"\u26a0 Position mismatch: {p_out['web_name']} is a {p_out['position']}, "
            f"{p_in['web_name']} is a {p_in['position']} \u2014 this can't be a direct one-for-one swap.",
            style={'color': COLORS['danger_text'], 'fontWeight': '600', 'marginTop': '10px'})

    price_delta = p_in['price'] - p_out['price']

    def _fx(row):
        v = row.get('fixture_string')
        return v if isinstance(v, str) and v else '\u2014'

    def _avail(row):
        v = row.get('avail_pct')
        return '100%' if pd.isna(v) else f"{v:.0f}%"

    return html.Div([
        html.Div([
            html.Div(verdict, style={
                'backgroundColor': colour,
                # FPL green/amber are light fills: purple text on them, white on red
                'color': 'white' if colour == COLORS['danger'] else COLORS['primary'],
                'padding': '14px 24px',
                'borderRadius': '8px', 'fontSize': '18px', 'fontWeight': '700',
                'textAlign': 'center', 'marginBottom': '6px'
            }),
            html.P(
                f"{p_in['web_name']} projects {in_proj:.1f}, {p_out['web_name']} projects {out_proj:.1f}"
                + (f", minus the {hit}-point hit" if hit else "")
                + f". Money move: {'+' if price_delta >= 0 else ''}\u00a3{price_delta:.1f}m.",
                style={'color': COLORS['text_light'], 'textAlign': 'center', 'marginBottom': '16px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("", style={'padding': '8px 12px'}),
                    html.Th(f"OUT: {p_out['web_name']}",
                            style={'padding': '8px 12px', 'color': COLORS['danger_text'], 'textAlign': 'center'}),
                    html.Th(f"IN: {p_in['web_name']}",
                            style={'padding': '8px 12px', 'color': COLORS['success_text'], 'textAlign': 'center'}),
                ])),
                html.Tbody([
                    _tp_stat_row("Team", p_out['team_name'], p_in['team_name']),
                    _tp_stat_row("Position", p_out['position'], p_in['position']),
                    _tp_stat_row("Price", f"\u00a3{p_out['price']:.1f}m", f"\u00a3{p_in['price']:.1f}m"),
                    _tp_stat_row("Proj next GW",
                                 f"{0 if pd.isna(p_out.get('proj_pts_next')) else p_out['proj_pts_next']:.2f}",
                                 f"{0 if pd.isna(p_in.get('proj_pts_next')) else p_in['proj_pts_next']:.2f}"),
                    _tp_stat_row("Proj next 5",
                                 f"{0 if pd.isna(p_out.get('proj_pts_5')) else p_out['proj_pts_5']:.1f}",
                                 f"{0 if pd.isna(p_in.get('proj_pts_5')) else p_in['proj_pts_5']:.1f}"),
                    _tp_stat_row("Form", f"{p_out['form']:.1f}" if pd.notna(p_out['form']) else '\u2014',
                                 f"{p_in['form']:.1f}" if pd.notna(p_in['form']) else '\u2014'),
                    _tp_stat_row("Availability", _avail(p_out), _avail(p_in)),
                    _tp_stat_row("Next 5 fixtures", _fx(p_out), _fx(p_in)),
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'}),
            position_warning,
            html.P("Projections come from the expected-points engine (xG/xA, minutes security, "
                   "fixture-specific difficulty, DEFCON, availability). A projected gain under ~2 "
                   "points is within model noise \u2014 treat it as a coin flip, not a signal.",
                   style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginTop': '14px'})
        ], style=CARD_STYLE)
    ])


# =============================================================================
# CHIP PLANNER CALLBACK
# =============================================================================

@callback(
    Output('cp-content', 'children'),
    Input('cp-load-btn', 'n_clicks'),
    [State('cp-team-id', 'value'), State('cp-horizon', 'value'),
     State('cp-league-id', 'value')],
    prevent_initial_call=True
)
def analyse_chip_windows(n_clicks, team_id, horizon, league_id):
    if not team_id:
        return html.Div([html.P("Enter your FPL team ID above.",
                                style={'color': COLORS['text_light'], 'textAlign': 'center',
                                       'padding': '30px 0'})], style=CARD_STYLE)

    data = get_data()
    current_gw_info = data.get('current_gw')
    if not current_gw_info:
        target = data.get('next_gw_num', 1)
        return html.Div([html.Div([
            html.P(f"The season hasn't started \u2014 squads (and therefore chip planning) "
                   f"become available after the GW{target} deadline.",
                   style={'color': COLORS['text_light'], 'fontWeight': '600',
                          'textAlign': 'center', 'padding': '40px 0'})
        ], style=CARD_STYLE)])

    gw_num = current_gw_info['id']
    picks_data = fetch_team_picks(int(team_id), gw_num)
    if not picks_data or 'picks' not in picks_data:
        return html.Div([html.Div([
            html.P("Could not load your squad \u2014 check the team ID.",
                   style={'color': COLORS['danger_text'], 'fontWeight': '600'})
        ], style=CARD_STYLE)])

    squad_ids = [pk['element'] for pk in picks_data['picks']]
    dfa = data.get('df_active', pd.DataFrame())
    squad = dfa[dfa['id'].isin(squad_ids)].copy()
    missing = len(squad_ids) - len(squad)

    if squad.empty:
        return html.Div([html.P("No projection data found for this squad.",
                                style={'color': COLORS['danger_text']})], style=CARD_STYLE)

    # Neutral per-GW base: precomputed in the refresh (falls back to a
    # fresh engine run if the column predates this feature)
    if 'proj_neutral_gw' in squad.columns and squad['proj_neutral_gw'].notna().any():
        squad['neutral_base'] = squad['proj_neutral_gw'].fillna(0)
    else:
        neutral = squad.copy()
        neutral['next_att_fdr'] = 3.0
        neutral['next_def_fdr'] = 3.0
        squad['neutral_base'] = compute_expected_points(
            neutral, gw_elapsed=max(gw_num, 1))[1].values

    fixtures_data = data.get('fixtures_data', [])
    teams_df = data.get('teams_df', pd.DataFrame())
    horizon = int(horizon or 8)
    future_gws = [g for g in range(gw_num + 1, gw_num + 1 + horizon) if g <= 38]
    if not future_gws:
        return html.Div([html.P("No future gameweeks left this season.",
                                style={'color': COLORS['text_light']})], style=CARD_STYLE)

    gw_lookup = build_gw_fixture_lookup(fixtures_data, teams_df, future_gws)

    # --- Free Hit needs a priced pool and a budget -----------------------
    # Budget is squad selling value + bank, NOT £100m — the Free Hit team is
    # built with the money you actually have.
    _eh = (picks_data or {}).get('entry_history') or {}
    fh_budget = (float(_eh.get('value', 1000)) + float(_eh.get('bank', 0))) / 10.0
    fh_pool = dfa[dfa['position'].notna() & dfa['price'].notna()].copy()
    if 'proj_neutral_gw' in fh_pool.columns and fh_pool['proj_neutral_gw'].notna().any():
        fh_pool['neutral_base'] = fh_pool['proj_neutral_gw'].fillna(0)
    else:
        fh_pool = fh_pool.iloc[0:0]

    _team_short = (dict(zip(teams_df['id'], teams_df['short_name']))
                   if not teams_df.empty and 'short_name' in teams_df.columns else {})
    squad_id_set = set(int(i) for i in squad_ids)

    has_lam = 'xgi_lam_neutral' in squad.columns
    has_mins = 'exp_mins_next' in squad.columns
    rows = []
    for g in future_gws:
        players = []
        blanks = 0
        for r in squad.itertuples():
            info = gw_lookup.get(g, {}).get(r.team, {})
            proj = project_player_gw(r.neutral_base, r.position, r.team, gw_lookup.get(g, {}))
            if proj == 0:
                blanks += 1
            # Ceiling in THIS gameweek: Poisson P(2+ involvements) with the
            # neutral xGI rate scaled by this fixture's goal environment and
            # fixture count. Triple Captain multiplies a hoped-for haul, not
            # an average — so the TC pick is scored on EV x ceiling, which
            # is how an explosive striker at home to a promoted side beats a
            # steady accumulator whose mean is marginally higher.
            lam_n = float(getattr(r, 'xgi_lam_neutral', 0) or 0) if has_lam else 0.0
            env = float(np.clip(info.get('att_env', 1.0) or 1.0, *ATT_ENV_CLIP))
            lam_g = max(lam_n * env * max(info.get('count', 0), 0), 0.0)
            p_haul = float(1 - np.exp(-lam_g) * (1 + lam_g)) if lam_g > 0 else 0.0
            tc_score = proj * (0.6 + 0.8 * p_haul)
            players.append({'name': r.web_name, 'position': r.position, 'proj': proj,
                            'team_id': r.team, 'haul': round(p_haul * 100, 1),
                            'tc_score': round(tc_score, 2),
                            'exp_mins': (float(getattr(r, 'exp_mins_next', np.nan))
                                         if has_mins else np.nan)})
        xi, bench = pick_best_xi(players)
        xi_total = sum(p['proj'] for p in xi)
        bench_total = sum(p['proj'] for p in bench)

        # Bench Boost's TRUE value: bench points you wouldn't have banked
        # via autosubs anyway.
        autosub_pts = estimate_autosub_points(xi, bench)
        bb_value = round(max(bench_total - autosub_pts, 0.0), 1)

        # Free Hit's TRUE value: best XI money can buy this GW, minus the XI
        # you'd have fielded. Positive on blanks (your side is broken) AND on
        # doubles (the pool is better than what you own) — the old blank
        # count could only ever see the first case.
        fh_best, fh_xi, fh_meta = 0.0, [], {}
        if not fh_pool.empty:
            fh_pool['proj_gw'] = project_pool_for_gw(fh_pool, gw_lookup.get(g, {}))
            fh_best, fh_xi, fh_meta = optimise_free_hit_xi(fh_pool, fh_budget)
        fh_delta = round(max(fh_best - xi_total, 0.0), 1)
        # Keep the actual XI, not just the total — the whole point of a Free
        # Hit recommendation is knowing WHO you'd be fielding, and which of
        # them you already own (if most of them, transfers may do the job
        # without spending the chip).
        _pos_rank = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        fh_detail = [{
            'name': pk['web_name'], 'position': pk['position'],
            'team': _team_short.get(pk['team'], ''),
            'price': round(float(pk['price']), 1),
            'proj': round(float(pk['proj_gw']), 1),
            'owned': 'yes' if int(pk['id']) in squad_id_set else '',
        } for pk in sorted(fh_xi, key=lambda q: (_pos_rank.get(q['position'], 9),
                                                 -q['proj_gw']))]

        with_fixture = len(players) - blanks
        by_tc = sorted(players, key=lambda p: -p['tc_score'])
        best = by_tc[0]
        runner = by_tc[1] if len(by_tc) > 1 else None
        _binfo = gw_lookup.get(g, {}).get(best.get('team_id'), {})
        rows.append({
            'gw': g, 'xi': round(xi_total, 1), 'bench': round(bench_total, 1),
            'squad_total': round(xi_total + bench_total, 1),
            'bb_value': bb_value, 'autosub_pts': autosub_pts,
            'fh_best': fh_best, 'fh_delta': fh_delta, 'fh_detail': fh_detail,
            'fh_spend': fh_meta.get('xi_spend', round(sum(d['price'] for d in fh_detail), 1)),
            'fh_reserve': fh_meta.get('bench_reserve', 0.0),
            'fh_formation': fh_meta.get('formation', ''),
            'fh_formation_table': fh_meta.get('formation_table', []),
            'fh_margin': fh_meta.get('margin', 0.0),
            'fh_owned': sum(1 for d in fh_detail if d['owned']),
            'tc_name': best['name'], 'tc_pts': round(best['proj'], 1),
            'tc_haul': best['haul'], 'tc_score': best['tc_score'],
            'tc_alt': (f"{runner['name']} ({runner['haul']:.0f}%)" if runner else ''),
            'tc_opp': _binfo.get('opp', ''),
            'tc_env': _binfo.get('att_env', 1.0),
            'blanks': blanks, 'with_fixture': with_fixture,
        })

    best_bb = max(rows, key=lambda r: r['bb_value'])
    best_tc = max(rows, key=lambda r: r['tc_score'])

    # Free Hit ranked on EXCESS over the horizon median, not raw delta.
    # The best XI in the game beats yours by 15-25 pts in ANY week purely
    # because it is the best XI in the game — that constant squad-quality
    # gap is present every gameweek and swamps the structural signal, so
    # ranking on raw delta picks a week out of noise. The median IS that
    # baseline gap; the excess over it is what the chip actually buys.
    import statistics as _stats
    _fh_baseline = _stats.median(r['fh_delta'] for r in rows)
    for _r in rows:
        _r['fh_excess'] = round(_r['fh_delta'] - _fh_baseline, 1)
    best_fh = max(rows, key=lambda r: r['fh_excess'])

    # Chip EV in points, not rankings: value of the best window vs the
    # median window over the horizon — i.e. what perfect timing is WORTH.
    med_bb = _stats.median(r['bb_value'] for r in rows)
    med_tc = _stats.median(r['tc_pts'] for r in rows)
    med_fh = _stats.median(r['fh_delta'] for r in rows)
    bb_ev_delta = round(best_bb['bb_value'] - med_bb, 1)
    tc_ev_delta = round(best_tc['tc_pts'] - med_tc, 1)
    fh_ev_delta = best_fh['fh_excess']

    # --- League chip availability, fetched BEFORE the recommendations so it
    # can weight each one rather than being tacked on as a footnote.
    #
    # A chip's value to your SCORE is its point delta. Its value to your
    # RANK depends on whether the people around you can mirror it. If a
    # rival still holds his Free Hit he can answer yours in the same blank
    # and you net roughly nothing; if he has already spent it, you bank the
    # whole delta against him. So the league-relative figure is the delta
    # scaled by the share of rivals who cannot answer — a lower bound, since
    # a rival who CAN answer may still choose not to.
    league_chips, league_name_note, n_rivals_chips = None, '', 0
    chip_matrix_rows, my_league_rank, managers_above = [], None, []
    if league_id and team_id:
        try:
            _lname, _entries = fetch_league_standings(int(league_id))
            _me = next((e for e in _entries if e['entry'] == int(team_id)), None)
            my_league_rank = _me['rank'] if _me else None
            _rivals = [e for e in _entries if e['entry'] != int(team_id)]
            if _rivals:
                _snaps = {}
                with ThreadPoolExecutor(max_workers=8) as _ex:
                    _futs = {_ex.submit(fetch_entry_chips, e['entry']): e['entry']
                             for e in _entries}
                    for _f in as_completed(_futs):
                        _snaps[_futs[_f]] = {'chips_used': _f.result()}
                _summ, _matrix = summarise_league_chips(
                    _snaps, _rivals, get_data().get('bootstrap_data'),
                    gw_num, my_id=None)
                league_chips = {r['chip']: r for r in _summ}
                n_rivals_chips = len(_rivals)
                league_name_note = _lname

                # Full matrix INCLUDING you, so the table reads as a league
                _, chip_matrix_rows = summarise_league_chips(
                    _snaps, _entries, get_data().get('bootstrap_data'),
                    gw_num, my_id=int(team_id))
                chip_matrix_rows.sort(key=lambda r: r['rank'])
                # The managers you are CHASING — the only ones whose chips can
                # cost you a place. Rivals below you are a protect problem, not
                # a chase one, and averaging them together hides both.
                if my_league_rank:
                    managers_above = [r for r in chip_matrix_rows
                                      if r['rank'] < my_league_rank]
        except Exception as _e:
            print(f"  league chip check failed: {_e}")

    _CHIP_KEY = {'Wildcard': 'wildcard', 'Free Hit': 'freehit',
                 'Bench Boost': 'bboost', 'Triple Captain': '3xc'}

    def _league_note(chip_label, delta):
        """Name the managers above you who can and cannot answer this chip."""
        if not league_chips or chip_label not in league_chips or n_rivals_chips == 0:
            return ""
        spent = league_chips[chip_label]['spent']
        key = _CHIP_KEY.get(chip_label)

        detail = ""
        if managers_above and key:
            blocked = [r['manager'] for r in managers_above if r.get(key) != 'held']
            holding = [r['manager'] for r in managers_above if r.get(key) == 'held']
            if blocked:
                _names = ', '.join(blocked[:3]) + ('...' if len(blocked) > 3 else '')
                detail = (f" Above you, {_names} cannot answer it "
                          f"({len(blocked)}/{len(managers_above)} of the managers you are "
                          f"chasing)")
                if holding:
                    detail += f"; {', '.join(holding[:3])} still can."
                else:
                    detail += " — nobody above you can mirror this."
            elif holding:
                detail = (f" But every manager above you ({', '.join(holding[:3])}) still "
                          f"holds theirs and can mirror you.")

        if spent == 0:
            return (f" All {n_rivals_chips} rivals still hold this chip, so it is table "
                    f"stakes rather than an edge." + detail)
        frac = spent / n_rivals_chips
        return (f" {spent} of {n_rivals_chips} rivals have already spent theirs, worth "
                f"roughly {delta * frac:.1f} pts against the league on top of the raw "
                f"gain." + detail)

    rec_lines = [
        html.P([html.Strong("Bench Boost: "),
                f"GW{best_bb['gw']} \u2014 worth {best_bb['bb_value']:.1f} pts there "
                f"(bench projects {best_bb['bench']:.1f}, but ~{best_bb['autosub_pts']:.1f} "
                f"of that would arrive via autosubs anyway). "
                f"Timing it right is worth +{bb_ev_delta} pts vs an average window."
                + _league_note('Bench Boost', best_bb['bb_value'])],
               style={'color': COLORS['text_dark'], 'marginBottom': '8px'}),
        html.P([html.Strong("Triple Captain: "),
                f"{best_tc['tc_name']} in GW{best_tc['gw']}"
                + (f" vs {best_tc['tc_opp']}" if best_tc.get('tc_opp') else "")
                + f" \u2014 {best_tc['tc_pts']:.1f} projected with a "
                + f"{best_tc.get('tc_haul', 0):.0f}% haul probability "
                + f"(+{tc_ev_delta} vs an average week). Picks are ranked on "
                + f"EV \u00d7 ceiling, not average alone \u2014 TC multiplies a haul, "
                + f"not a mean."
                + (f" Alternative: {best_tc['tc_alt']}." if best_tc.get('tc_alt') else "")
                + _league_note('Triple Captain', best_tc['tc_pts'])],
               style={'color': COLORS['text_dark'], 'marginBottom': '8px'}),
    ]

    # Free Hit, priced the same way as the other two: what the best XI money
    # can buy that week beats the XI you'd otherwise field.
    if best_fh['fh_excess'] >= 4.0:
        _why = ("your squad is short of fixtures" if best_fh['with_fixture'] < 11
                else "that week's fixtures suit the pool far better than your squad")
        rec_lines.append(html.P([
            html.Strong("Free Hit: "),
            f"GW{best_fh['gw']} \u2014 {best_fh['fh_excess']:+.1f} pts better than a "
            f"typical week, because {_why}. Best available XI projects "
            f"{best_fh['fh_best']:.1f} vs your {best_fh['xi']:.1f} on a "
            f"\u00a3{fh_budget:.1f}m budget; {best_fh['with_fixture']}/15 of your squad "
            f"have a fixture. You already own {best_fh['fh_owned']}/11 of that XI."
            + _league_note('Free Hit', best_fh['fh_excess'])],
            style={'color': COLORS['text_dark'], 'marginBottom': '8px'}))
    else:
        rec_lines.append(html.P([
            html.Strong("Free Hit: "),
            f"hold \u2014 no week in the next {len(rows)} stands out. The best "
            f"(GW{best_fh['gw']}) is only {best_fh['fh_excess']:+.1f} pts better than "
            f"average, which is noise, not opportunity. Blanks and doubles form "
            f"from cup progression and rarely appear before late February."
            + _league_note('Free Hit', best_fh['fh_excess'])],
            style={'color': COLORS['text_dark'], 'marginBottom': '8px'}))
    rec_lines.append(html.P(
        f"Reading the Free Hit numbers: the best XI in the game beats yours by "
        f"~{_fh_baseline:.0f} pts in ANY week simply because it is the best XI in "
        f"the game. That baseline is not a reason to play the chip \u2014 only the "
        f"excess above it is.",
        style={'color': COLORS['text_light'], 'fontSize': '13px',
               'marginBottom': '8px'}))

    # One chip per gameweek — flag collisions rather than recommending both.
    _clash = {}
    for _label, _r in (('Bench Boost', best_bb), ('Triple Captain', best_tc),
                       ('Free Hit', best_fh)):
        _clash.setdefault(_r['gw'], []).append(_label)
    for _g, _names in _clash.items():
        if len(_names) > 1:
            rec_lines.append(html.P(
                f"Note: {' and '.join(_names)} both point at GW{_g}, and FPL allows "
                f"only one chip per gameweek. Play the higher-value one there and "
                f"take the next-best window for the other.",
                style={'color': COLORS['warning_text'], 'fontSize': '13px',
                       'fontWeight': '600', 'marginBottom': '8px'}))

    if missing:
        rec_lines.append(html.P(f"Note: {missing} squad player(s) had no projection data "
                                f"(usually zero minutes so far) and count as 0.",
                                style={'color': COLORS['text_light'], 'fontSize': '13px'}))

    # Rival chip-window collision: can anyone in your league answer your
    # planned window? A BB into a week the leader has already spent his on
    # is worth double its raw points in league terms.
    # League chip standings, as a summary line beneath the recommendations.
    # The old version counted every chip a rival had EVER played, which is
    # wrong under two chip sets: a Free Hit spent in GW5 does not stop anyone
    # playing another one after the split. summarise_league_chips scopes the
    # count to the window that actually applies.
    if league_chips and n_rivals_chips:
        _parts = ', '.join(
            f"{r['spent']}/{n_rivals_chips} spent {label}"
            for label, r in league_chips.items())
        _rank_note = (f" You are {my_league_rank}"
                      + {1: 'st', 2: 'nd', 3: 'rd'}.get(my_league_rank % 100 if my_league_rank else 0, 'th')
                      + f", chasing {len(managers_above)}."
                      if my_league_rank else "")
        rec_lines.append(html.P([
            html.Strong(f"League chip standings"
                        + (f" ({league_name_note})" if league_name_note else "") + ": "),
            _parts + "." + _rank_note
            + " Windows come from the game's own chip calendar, so a chip played in the "
              "first half does not count against the second."],
            style={'color': COLORS['text_dark'], 'marginBottom': '8px',
                   'backgroundColor': '#f0e6f5', 'padding': '10px', 'borderRadius': '6px'}))

    # All three chips on ONE points axis, so the comparison is direct.
    _x = [f"GW{r['gw']}" for r in rows]
    bench_fig = go.Figure()
    bench_fig.add_trace(go.Bar(
        name='Bench Boost', x=_x, y=[r['bb_value'] for r in rows],
        marker_color=[COLORS['success'] if r['gw'] == best_bb['gw'] else COLORS['info']
                      for r in rows],
        text=[f"{r['bb_value']:.1f}" for r in rows], textposition='outside',
        hovertemplate=('%{x}<br>Bench Boost gain: %{y:.1f} pts<br>'
                       'Raw bench %{customdata[0]:.1f} less ~%{customdata[1]:.1f} '
                       'autosubs<extra></extra>'),
        customdata=[[r['bench'], r['autosub_pts']] for r in rows],
    ))
    bench_fig.add_trace(go.Scatter(
        name='Free Hit', x=_x, y=[r['fh_delta'] for r in rows],
        mode='lines+markers', line=dict(color=COLORS['accent'], width=2),
        hovertemplate=('%{x}<br>Free Hit gain: %{y:.1f} pts<br>'
                       'Best available XI %{customdata:.1f}<extra></extra>'),
        customdata=[r['fh_best'] for r in rows],
    ))
    bench_fig.update_layout(template='plotly_white', height=340,
                            yaxis_title='Chip gain (points)',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                        xanchor='center', x=0.5),
                            margin=dict(t=50, b=30, l=50, r=20),
                            font=dict(family=FONT_FAMILY))

    table_rows = [{
        'gw': f"GW{r['gw']}", 'xi': r['xi'], 'bench': r['bb_value'],
        'fh_delta': r['fh_delta'], 'fh_excess': r['fh_excess'],
        'squad_total': r['squad_total'],
        'tc': f"{r['tc_name']} ({r['tc_pts']:.1f} pts, {r.get('tc_haul', 0):.0f}% haul"
              + (f" vs {r['tc_opp']}" if r.get('tc_opp') else "") + ")",
        'blanks': r['blanks'],
    } for r in rows]

    return html.Div([
        html.Div([
            html.H3("Chip Windows \u2014 Recommendations", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
            *rec_lines
        ], style={**CARD_STYLE, 'backgroundColor': '#f8f9fa'}),

        html.Div([
            html.H3("Who Can Answer You", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P(["Named, in league order. ", html.Strong("Green = still held"),
                    " means that manager can mirror your chip in the same week; a gameweek "
                    "number means they have already spent it and cannot. Rows above yours are "
                    "the ones that decide whether you close the gap \u2014 a chip that only "
                    "beats the people below you protects a position, it does not win one."],
                   style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=chip_matrix_rows,
                columns=[
                    {'name': 'Rank', 'id': 'rank'},
                    {'name': 'Manager', 'id': 'manager'},
                    {'name': 'Wildcard', 'id': 'wildcard'},
                    {'name': 'Free Hit', 'id': 'freehit'},
                    {'name': 'Bench Boost', 'id': 'bboost'},
                    {'name': 'Triple Captain', 'id': '3xc'},
                ],
                sort_action='native', page_size=25,
                style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'filter_query': '{manager} contains "(you)"'},
                     'backgroundColor': '#fff8e1', 'fontWeight': '700'},
                ] + [
                    {'if': {'filter_query': '{%s} = held' % c, 'column_id': c},
                     'backgroundColor': '#e6fff2', 'fontWeight': '600'}
                    for c in ('wildcard', 'freehit', 'bboost', '3xc')
                ] + [
                    {'if': {'filter_query': '{%s} != held' % c, 'column_id': c},
                     'color': COLORS['text_light']}
                    for c in ('wildcard', 'freehit', 'bboost', '3xc')
                ]),
        ], style=CARD_STYLE) if chip_matrix_rows else html.Div(),

        html.Div([
            html.H3(f"Free Hit XI \u2014 GW{best_fh['gw']}", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P([f"Formation {best_fh['fh_formation']}. XI costs "
                    f"\u00a3{best_fh['fh_spend']:.1f}m of your \u00a3{fh_budget:.1f}m "
                    f"(squad value + bank), with only \u00a3{best_fh['fh_reserve']:.1f}m held "
                    f"back for the four cheapest bench fillers in the game \u2014 on a Free "
                    f"Hit the bench is filler, so every remaining penny goes into the XI. ",
                    html.Strong(f"You already own {best_fh['fh_owned']} of these 11"),
                    " \u2014 if that number is high, transfers may get you most of the way "
                    "without spending the chip."],
                   style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=best_fh['fh_detail'],
                columns=[
                    {'name': 'Player', 'id': 'name'},
                    {'name': 'Pos', 'id': 'position'},
                    {'name': 'Team', 'id': 'team'},
                    {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Proj', 'id': 'proj', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Owned', 'id': 'owned'},
                ],
                style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                    {'if': {'filter_query': '{owned} = yes'}, 'backgroundColor': '#e6fff2'},
                ]),
        ], style=CARD_STYLE) if best_fh.get('fh_detail') else html.Div(),

        html.Div([
            html.H4("Formation Comparison", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P([
                (f"{best_fh['fh_formation']} wins by {best_fh['fh_margin']:.1f} pts over the "
                 f"next-best shape. " if best_fh.get('fh_margin') is not None else ""),
                html.Strong(
                    "That margin is noise \u2014 take whichever shape you prefer."
                    if best_fh.get('fh_margin', 0) < 1.5 else
                    "That is a real read on this week's fixtures, not a coin flip."),
                " The optimiser scores every legal shape and reports the highest total, but it "
                "maximises EXPECTED points and knows nothing about your league position. Five at "
                "the back is a tight, reliable distribution; three at the back with three forwards "
                "has a fatter tail both ways. If you are chasing, the bigger ceiling is usually "
                "worth a point or two of mean."],
                style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
            dash_table.DataTable(
                data=best_fh.get('fh_formation_table', []),
                columns=[
                    {'name': 'Formation', 'id': 'formation'},
                    {'name': 'Proj', 'id': 'total', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'XI cost', 'id': 'xi_spend', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Bench held back', 'id': 'bench_reserve', 'type': 'numeric',
                     'format': {'specifier': '.1f'}},
                ],
                style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                    {'if': {'row_index': 0}, 'backgroundColor': '#e6fff2', 'fontWeight': '600'},
                ]),
        ], style=CARD_STYLE) if best_fh.get('fh_formation_table') else html.Div(),

        html.Div([
            html.H3("Chip Gain by Gameweek", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P("Both chips priced in points so they compare directly. Bars are Bench Boost's gain "
                   "net of autosubs; the line is Free Hit's gain over the XI you'd otherwise field. "
                   "Re-run after wildcards or DGW announcements.",
                   style={'color': COLORS['text_light']}),
            dcc.Graph(figure=bench_fig, config={'displayModeBar': False})
        ], style=CARD_STYLE),

        html.Div([
            html.H4("Gameweek-by-Gameweek Squad Projection", style={'color': COLORS['primary'], 'marginBottom': '16px'}),
            dash_table.DataTable(
                data=table_rows,
                columns=[
                    {'name': 'GW', 'id': 'gw'},
                    {'name': 'Best XI Proj', 'id': 'xi', 'type': 'numeric'},
                    {'name': 'BB gain', 'id': 'bench', 'type': 'numeric'},
                    {'name': 'FH gain', 'id': 'fh_delta', 'type': 'numeric'},
                    {'name': 'FH vs typical', 'id': 'fh_excess', 'type': 'numeric',
                     'format': {'specifier': '+.1f'}},
                    {'name': 'Full Squad Proj', 'id': 'squad_total', 'type': 'numeric'},
                    {'name': 'Best TC Pick', 'id': 'tc'},
                    {'name': 'Players Blanking', 'id': 'blanks', 'type': 'numeric'},
                ],
                style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                    {'if': {'filter_query': '{blanks} > 0', 'column_id': 'blanks'},
                     'backgroundColor': '#fde8ef', 'fontWeight': '600'},
                ]
            ),
            html.P("Projections assume your current fifteen held over the horizon; DGWs count both fixtures, blanks count zero.",
                   style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginTop': '12px'})
        ], style=CARD_STYLE),
    ])


# =============================================================================
# MINI-LEAGUE RIVALS CALLBACK
# =============================================================================

def _rv_list_card(title, subtitle, items, accent):
    return html.Div([
        html.H4(title, style={'color': COLORS['primary'], 'marginBottom': '4px',
                              'borderLeft': f'4px solid {accent}', 'paddingLeft': '10px'}),
        html.P(subtitle, style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginBottom': '12px'}),
        html.Ul([
            html.Li(item, style={'marginBottom': '6px', 'fontSize': '14px', 'color': COLORS['text_dark']})
            for item in items
        ], style={'paddingLeft': '18px', 'margin': 0}) if items else
        html.P("\u2014 none \u2014", style={'color': COLORS['text_light']})
    ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '260px'})


@callback(
    Output('rv-content', 'children'),
    Input('rv-load-btn', 'n_clicks'),
    [State('rv-league-id', 'value'), State('rv-my-id', 'value')],
    prevent_initial_call=True
)
def load_rivals(n_clicks, league_id, my_id):
    if not league_id:
        return html.Div([html.P("Enter a league ID above.",
                                style={'color': COLORS['text_light'], 'textAlign': 'center',
                                       'padding': '30px 0'})], style=CARD_STYLE)

    league_name, entries = fetch_league_standings(int(league_id))
    if not entries:
        return html.Div([html.Div([
            html.P(f"Could not load league {league_id}. Check the ID \u2014 it must be a classic "
                   f"(not head-to-head) league.",
                   style={'color': COLORS['danger_text'], 'fontWeight': '600'})
        ], style=CARD_STYLE)])

    data = get_data()
    dfa = data.get('df_active', pd.DataFrame())
    df_all = data.get('df', pd.DataFrame())
    name_map = dict(zip(df_all['id'], df_all['web_name'])) if not df_all.empty else {}
    current_gw_info = data.get('current_gw')

    # Pre-season: no picks exist yet — show the table we do have, cleanly.
    if not current_gw_info:
        target = data.get('next_gw_num', 1)
        table = dash_table.DataTable(
            data=[{'rank': e['rank'] if e['rank'] is not None else '\u2014',
                   'player_name': e['player_name'],
                   'entry_name': e['entry_name'],
                   'total': e['total'] if e['rank'] is not None else '\u2014'} for e in entries],
            columns=[{'name': 'Rank', 'id': 'rank'}, {'name': 'Manager', 'id': 'player_name'},
                     {'name': 'Team', 'id': 'entry_name'}, {'name': 'Total', 'id': 'total'}],
            style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
        )
        return html.Div([
            html.Div([
                html.H3(league_name, style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                html.P(f"Squads, captains and chips appear after the GW{target} deadline \u2014 "
                       f"for now, here's who you're up against.",
                       style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
                table
            ], style=CARD_STYLE)
        ])

    gw_num = current_gw_info['id']
    snapshots = fetch_rival_snapshots(entries, gw_num)

    my_id = int(my_id) if my_id else None
    entry_ids = {e['entry'] for e in entries}
    my_snapshot = snapshots.get(my_id)
    if my_id and my_id not in entry_ids:
        # User's team isn't in the sampled slice — fetch it separately
        pk = fetch_team_picks(my_id, gw_num)
        if pk and 'picks' in pk:
            my_snapshot = {'picks': pk['picks'],
                           'entry_history': pk.get('entry_history', {}) or {},
                           'active_chip': pk.get('active_chip'),
                           'chips_used': fetch_entry_chips(my_id)}

    rival_snapshots = {eid: s for eid, s in snapshots.items() if eid != my_id}
    eo_pct, owner_count = build_league_ownership(rival_snapshots if my_snapshot else snapshots)
    n_rivals = len(rival_snapshots if my_snapshot else snapshots)

    def _proj(pid):
        row = dfa[dfa['id'] == pid]
        if row.empty:
            return 0.0
        v = row.iloc[0].get('proj_pts_next')
        return 0.0 if pd.isna(v) else float(v)

    # --- League table with captain / chips / value ---
    captain_counter = Counter()
    table_rows = []
    for e in entries:
        snap = snapshots.get(e['entry'])
        cap_name, chips_str, value_str, bank_str, active = '\u2014', '\u2014', '\u2014', '\u2014', ''
        if snap:
            cap_pick = next((pk for pk in snap['picks'] if pk.get('is_captain')), None)
            if cap_pick:
                cap_name = name_map.get(cap_pick['element'], f"#{cap_pick['element']}")
                captain_counter[cap_name] += 1
            used = snap['chips_used']
            chips_str = ', '.join(
                f"{chip_name_map.get(c['name'], c['name'])} (GW{c['event']})" for c in used
            ) if used else 'None yet'
            eh = snap['entry_history']
            if eh.get('value') is not None:
                value_str = f"\u00a3{eh['value'] / 10:.1f}m"
            if eh.get('bank') is not None:
                bank_str = f"\u00a3{eh['bank'] / 10:.1f}m"
            active = chip_name_map.get(snap['active_chip'], snap['active_chip']) if snap['active_chip'] else ''
        is_me = (my_id is not None and e['entry'] == my_id)
        table_rows.append({
            'rank': e['rank'],
            'player_name': e['player_name'] + (' (you)' if is_me else ''),
            'entry_name': e['entry_name'],
            'event_total': e['event_total'], 'total': e['total'],
            'captain': cap_name + (f" \u2605 {active}" if active else ''),
            'chips': chips_str, 'value': value_str, 'bank': bank_str,
        })

    league_table = dash_table.DataTable(
        data=table_rows,
        columns=[
            {'name': 'Rank', 'id': 'rank'},
            {'name': 'Manager', 'id': 'player_name'},
            {'name': 'Team', 'id': 'entry_name'},
            {'name': 'GW Pts', 'id': 'event_total', 'type': 'numeric'},
            {'name': 'Total', 'id': 'total', 'type': 'numeric'},
            {'name': f'GW{gw_num} Captain', 'id': 'captain'},
            {'name': 'Chips Used', 'id': 'chips'},
            {'name': 'Team Value', 'id': 'value'},
            {'name': 'Bank', 'id': 'bank'},
        ],
        style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
            {'if': {'filter_query': '{player_name} contains "(you)"'},
             'backgroundColor': '#e6fff2', 'fontWeight': '600'},
        ],
    )

    # --- Captain tracker ---
    cap_items = [f"{name}: {cnt} manager{'s' if cnt != 1 else ''}"
                 for name, cnt in captain_counter.most_common()]

    # --- Threats / leverage / shared (needs your squad) ---
    if my_snapshot:
        my_ids = {pk['element'] for pk in my_snapshot['picks']}
        rival_owned = set(owner_count.keys())

        def _fmt(pid, with_owners=True):
            nm = name_map.get(pid, f"#{pid}")
            pj = _proj(pid)
            if with_owners:
                cnt = owner_count.get(pid, 0)
                return f"{nm} \u2014 {cnt}/{n_rivals} rivals, proj {pj:.1f}"
            return f"{nm} \u2014 proj {pj:.1f}"

        threats = sorted([pid for pid in rival_owned if pid not in my_ids],
                         key=lambda pid: (-owner_count.get(pid, 0), -_proj(pid)))[:12]
        leverage = sorted([pid for pid in my_ids if pid not in rival_owned],
                          key=lambda pid: -_proj(pid))[:12]
        shared = sorted([pid for pid in my_ids if pid in rival_owned],
                        key=lambda pid: -owner_count.get(pid, 0))[:12]

        versus_section = html.Div([
            _rv_list_card("Threats", "They own, you don't \u2014 every point scores against you",
                          [_fmt(pid) for pid in threats], COLORS['danger']),
            _rv_list_card("Your Leverage", "You own, no rival does \u2014 your rank-movers",
                          [_fmt(pid, with_owners=False) for pid in leverage], COLORS['success']),
            _rv_list_card("Shared (cancels out)", "Owned by you and rivals \u2014 moves nothing between you",
                          [_fmt(pid) for pid in shared], COLORS['info']),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '20px'})
    else:
        versus_section = html.Div([
            html.P("Add your Team ID and reload to unlock Threats / Leverage / Shared analysis.",
                   style={'color': COLORS['text_light'], 'textAlign': 'center', 'padding': '20px 0'})
        ], style=CARD_STYLE)

    # --- League EO table ---
    eo_rows = []
    for pid, eo_val in sorted(eo_pct.items(), key=lambda kv: -kv[1])[:25]:
        row = dfa[dfa['id'] == pid]
        glob = row.iloc[0]['ownership'] if not row.empty and pd.notna(row.iloc[0]['ownership']) else None
        eo_rows.append({
            'player': name_map.get(pid, f"#{pid}"),
            'owners': f"{owner_count.get(pid, 0)}/{n_rivals}",
            'league_eo': eo_val,
            'global_own': round(glob, 1) if glob is not None else None,
            'proj': round(_proj(pid), 2),
            'mine': '\u2713' if (my_snapshot and pid in {pk['element'] for pk in my_snapshot['picks']}) else '',
        })

    eo_table = dash_table.DataTable(
        data=eo_rows,
        columns=[
            {'name': 'Player', 'id': 'player'},
            {'name': 'Rival Owners', 'id': 'owners'},
            {'name': 'League EO%', 'id': 'league_eo', 'type': 'numeric'},
            {'name': 'Global Own%', 'id': 'global_own', 'type': 'numeric'},
            {'name': 'Proj Pts', 'id': 'proj', 'type': 'numeric'},
            {'name': 'You Own', 'id': 'mine'},
        ],
        sort_action='native', page_size=25,
        style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER, style_data=TABLE_STYLE_DATA,
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
            {'if': {'filter_query': '{league_eo} >= 100', 'column_id': 'league_eo'},
             'backgroundColor': '#fde8ef', 'fontWeight': '600'},
        ],
    )

    # Chip availability across the league — the version that moves your rank
    chip_summary, chip_matrix = summarise_league_chips(
        snapshots, entries, get_data().get('bootstrap_data'), gw_num, my_id)

    loaded_note = (f"Loaded {len(snapshots)}/{len(entries)} squads for GW{gw_num}. "
                   f"League EO counts captains double and triple captains treble \u2014 "
                   f"100%+ means effectively more than one copy per rival squad.")

    return html.Div([
        html.Div([
            html.H3(league_name, style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P(loaded_note, style={'color': COLORS['text_light'], 'marginBottom': '16px'}),
            league_table
        ], style=CARD_STYLE),

        versus_section,

        html.Div([
            html.H4("Chip Availability in This League",
                    style={'color': COLORS['primary'], 'marginBottom': '8px'}),
            html.P(["Global chip usage tells you nothing about your rank \u2014 a mini-league is "
                    "scored in gaps. What matters is whether the people around you can ",
                    html.Strong("answer"), " your chip. A rival who has already spent his Free Hit "
                    "is defenceless in a blank, so yours becomes a swing rather than a "
                    "like-for-like trade. Windows come from the game's own chip calendar, so the "
                    "two-set structure is handled."],
                   style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
            html.Div([
                html.Div([
                    build_stat_card(
                        r['chip'], f"{r['held']}/{len(entries)}",
                        f"still hold it \u00b7 {r['window']}",
                        color=COLORS['success'] if r['held_pct'] < 50 else COLORS['primary'])
                ], style={'flex': '1', 'minWidth': '180px', 'padding': '0 10px'})
                for r in chip_summary
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 20px -10px'}),
            dash_table.DataTable(
                data=chip_matrix,
                columns=[
                    {'name': 'Rank', 'id': 'rank'},
                    {'name': 'Manager', 'id': 'manager'},
                    {'name': 'Wildcard', 'id': 'wildcard'},
                    {'name': 'Free Hit', 'id': 'freehit'},
                    {'name': 'Bench Boost', 'id': 'bboost'},
                    {'name': 'Triple Captain', 'id': '3xc'},
                ],
                sort_action='native', page_size=20,
                style_cell=TABLE_STYLE_CELL, style_header=TABLE_STYLE_HEADER,
                style_data=TABLE_STYLE_DATA,
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                ] + [
                    {'if': {'filter_query': '{%s} = held' % c, 'column_id': c},
                     'backgroundColor': '#e6fff2', 'fontWeight': '600'}
                    for c in ('wildcard', 'freehit', 'bboost', '3xc')
                ] + [
                    {'if': {'filter_query': '{%s} != held' % c, 'column_id': c},
                     'color': COLORS['text_light']}
                    for c in ('wildcard', 'freehit', 'bboost', '3xc')
                ]),
        ], style=CARD_STYLE),

        html.Div([
            html.H4(f"GW{gw_num} Captain Picks", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
            html.P("Behind the leader and sharing his captain? You can't catch him this week. "
                   "Ahead? Covering the chasers' captain removes their biggest weapon.",
                   style={'color': COLORS['text_light'], 'fontSize': '13px', 'marginBottom': '12px'}),
            html.Ul([html.Li(item, style={'marginBottom': '6px', 'fontSize': '14px'})
                     for item in cap_items]) if cap_items else
            html.P("No captain data loaded.", style={'color': COLORS['text_light']}),
        ], style=CARD_STYLE),

        html.Div([
            html.H4("Effective Ownership Within This League", style={'color': COLORS['primary'], 'marginBottom': '12px'}),
            eo_table
        ], style=CARD_STYLE),
    ])


# =============================================================================
# EXPECTED CLEAN SHEETS CALLBACK
# =============================================================================

@callback(
    [Output('xcs-bar', 'figure'), Output('xcs-team-table', 'data'),
     Output('xcs-player-table', 'data'), Output('xcs-chart-title', 'children'),
     Output('xcs-form-note', 'children')],
    [Input('xcs-horizon', 'value'), Input('visit-xcs', 'data')],
    prevent_initial_call=True
)
def update_expected_clean_sheets(horizon, n):
    _need_visit(n)
    data = get_data()
    fixtures_data = data.get('fixtures_data', [])
    teams_df = data.get('teams_df', pd.DataFrame())
    dfa = data.get('df_active', pd.DataFrame())

    anchor = data.get('fixture_anchor_gw')
    if anchor is None:
        cur = data.get('current_gw')
        anchor = cur['id'] if cur else 0

    horizon = int(horizon or 5)

    def _empty(msg):
        fig = go.Figure()
        fig.add_annotation(text=msg, xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color=COLORS['text_light']))
        fig.update_layout(template='plotly_white', height=420)
        return [fig, [], [], "Expected Clean Sheets", None]

    if teams_df.empty or not fixtures_data:
        return _empty("Data loading \u2014 please wait...")

    xcs = calculate_expected_clean_sheets(fixtures_data, teams_df, anchor, num_gws=horizon,
                                          odds_lambdas=data.get('odds_lambdas'),
                                          xg_ledger=data.get('xg_ledger'))
    if not xcs:
        return _empty("Team strength data unavailable.")

    name_map = dict(zip(teams_df['id'], teams_df['name']))
    rows = []
    for tid, v in xcs.items():
        rows.append({
            'team': name_map.get(tid, str(tid)),
            'xcs': v['xcs'], 'avg_cs_prob': v['avg_cs_prob'], 'count': v['count'],
            'recent_conceded': v['recent_conceded_pm'],
            'recent_goals_conceded': v['recent_goals_conceded_pm'],
            'recent_cs': (f"{v['recent_cs']}/{v['recent_played']}"
                          if v['recent_cs'] is not None else '\u2014'),
            'fixture_string': v['fixture_string'],
        })
    rows.sort(key=lambda r: -r['xcs'])

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=[r['team'] for r in rows],
        y=[r['xcs'] for r in rows],
        marker_color=[COLORS['success'] if r['avg_cs_prob'] >= 40 else
                      (COLORS['warning'] if r['avg_cs_prob'] >= 25 else COLORS['danger'])
                      for r in rows],
        text=[f"{r['xcs']:.2f}" for r in rows], textposition='outside',
        customdata=[r['fixture_string'] for r in rows],
        hovertemplate='<b>%{x}</b><br>xCS: %{y:.2f}<br>%{customdata}<extra></extra>',
    ))
    bar_fig.update_layout(template='plotly_white', height=420, xaxis_tickangle=-45,
                          yaxis_title=f'Expected clean sheets (next {horizon} GW{"s" if horizon > 1 else ""})',
                          showlegend=False, font=dict(family=FONT_FAMILY),
                          margin=dict(t=30, b=80, l=50, r=20))

    # Defender/keeper picks joined to team xCS
    player_rows = []
    if not dfa.empty:
        team_xcs_map = {tid: v['xcs'] for tid, v in xcs.items()}
        picks = dfa[dfa['position'].isin(['GKP', 'DEF'])].copy()
        picks['team_xcs'] = picks['team'].map(team_xcs_map).round(2)
        picks = picks.sort_values(['team_xcs', 'proj_pts_5'], ascending=[False, False]).head(60)
        cols = ['web_name', 'team_name', 'position', 'price', 'team_xcs',
                'proj_pts_5', 'cs_per_90', 'minutes', 'ownership']
        player_rows = prepare_table_data(picks, cols)

    # Form-blend status note: how much of the number is form vs reputation
    played = [v['recent_played'] for v in xcs.values()]
    max_played = max(played) if played else 0
    if max_played == 0:
        note = html.Span("Pre-season: no results yet, so projections are 100% team-strength "
                         "based. Form blends in automatically from the first finished fixture.",
                         style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})
    else:
        note = html.Span(f"Form-aware: blending last-{min(max_played, 6)}-game results with team "
                         f"strengths (up to 60% form weight).",
                         style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                'padding': '8px 16px', 'borderRadius': '20px', 'fontWeight': '600'})

    title = f"Expected Clean Sheets \u2014 Next {horizon} Gameweek{'s' if horizon > 1 else ''}"
    return [bar_fig, rows, player_rows, title, note]


# =============================================================================
# SHOT INTELLIGENCE — callbacks
# =============================================================================

_SHOT_COMP_COLOURS = {'set_piece': COLORS['accent'], 'cross': COLORS['info'],
                      'op_centre': COLORS['primary'], 'op_left': COLORS['primary'],
                      'op_right': COLORS['primary']}
_SIDE_LABEL = {'left': 'Left', 'right': 'Right', 'centre': 'Central'}


def _shot_note(text, tone='info'):
    colours = {'info': ('#e3f2fd', '#90caf9'), 'warn': ('#fff8e1', '#ffe082')}
    bg, border = colours.get(tone, colours['info'])
    return html.Div(text, style={'backgroundColor': bg, 'border': f'1px solid {border}',
                                 'borderRadius': '8px', 'padding': '12px 16px',
                                 'marginBottom': '20px', 'fontSize': '14px',
                                 'color': COLORS['text_dark']})


def _shot_intel_status(data):
    """(intel or None, status banner). Kicks a sync if one is due."""
    check_shots_sync()
    intel = data.get('shot_intel')
    st = data.get('shots_status') or {}
    if intel is None:
        if data.get('shots_syncing') or not st:
            msg = ("Syncing shot data from Understat. The first sync downloads every match played "
                   "so far (a minute or two); this page fills in automatically within a couple of "
                   "minutes of it finishing.")
        else:
            msg = (f"Shot data isn't available right now ({st.get('error') or 'no matches yet'}). "
                   f"It retries automatically every 10 minutes.")
        return None, _shot_note(msg, 'warn')
    bits = [f"Shot data: {intel['n_matches']} matches"]
    if intel.get('last_match_date'):
        bits.append(f"latest {intel['last_match_date']}")
    if st.get('pending'):
        bits.append(f"{st['pending']} more still downloading")
    bits.append(f"profiles shrunk towards league average over {SHOT_PROFILE_K:.0f} matches")
    tone = 'info'
    extra = []
    if intel.get('unmatched_teams'):
        extra.append(f" Couldn't match Understat team(s) {', '.join(intel['unmatched_teams'])} to FPL.")
        tone = 'warn'
    if str(intel.get('orientation_note', '')).startswith('default'):
        extra.append(" Left/right orientation not yet verified from the data (needs a few more matches).")
    if st.get('error'):
        extra.append(f" Last sync issue: {st['error']}.")
        tone = 'warn'
    return intel, _shot_note(' · '.join(bits) + '.' + ''.join(extra), tone)


def _blank_fig(msg='', height=300):
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color=COLORS['text_light']))
    fig.update_layout(template='plotly_white', height=height, xaxis=dict(visible=False),
                      yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _pitch_base(left_label, right_label):
    """Empty vertical half pitch, attacked goal at the top."""
    line = dict(color='#bdbdbd', width=1.5)
    fig = go.Figure()
    shapes = [
        dict(type='rect', x0=0, y0=0.5, x1=1, y1=1, line=line),
        dict(type='rect', x0=SHOT_BOX_Y[0], y0=SHOT_BOX_X, x1=SHOT_BOX_Y[1], y1=1, line=line),
        dict(type='rect', x0=SHOT_CENTRE_Y[0], y0=SHOT_SIX_X, x1=SHOT_CENTRE_Y[1], y1=1, line=line),
        dict(type='rect', x0=0.446, y0=1, x1=0.554, y1=1.015,
             line=dict(color='#9e9e9e', width=2)),
    ]
    fig.add_annotation(x=0.02, y=0.505, xanchor='left', yanchor='bottom', showarrow=False,
                       text=left_label, font=dict(size=11, color=COLORS['text_light']))
    fig.add_annotation(x=0.98, y=0.505, xanchor='right', yanchor='bottom', showarrow=False,
                       text=right_label, font=dict(size=11, color=COLORS['text_light']))
    fig.update_layout(
        template='plotly_white', height=420, shapes=shapes,
        xaxis=dict(range=[-0.02, 1.02], visible=False, fixedrange=True),
        yaxis=dict(range=[0.49, 1.03], visible=False, fixedrange=True,
                   scaleanchor='x', scaleratio=105 / 68),
        legend=dict(orientation='h', y=-0.02, x=0.5, xanchor='center'),
        margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor='#fbfdf9')
    return fig


def _pitch_arcs(fig):
    """Penalty arc (outside the box only) and centre-circle arc, 9.15m radius.
    Added after any heatmap so the markings sit on top of it."""
    line = dict(color='#bdbdbd', width=1.5)
    t = np.linspace(0, np.pi, 60)
    rx, ry = 9.15 / 68, 9.15 / 105
    arc_x, arc_y = 0.5 + rx * np.cos(t), 0.885 - ry * np.sin(t)
    keep = arc_y < SHOT_BOX_X
    fig.add_trace(go.Scatter(x=arc_x[keep], y=arc_y[keep], mode='lines', line=line,
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=0.5 + rx * np.cos(t), y=0.5 + ry * np.sin(t), mode='lines',
                             line=line, hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=[0.5], y=[0.885], mode='markers', marker=dict(size=4, color='#bdbdbd'),
                             hoverinfo='skip', showlegend=False))


def _half_pitch_fig(shots, left_label, right_label):
    """Vertical half pitch with every shot, sized by xG; goals are stars."""
    fig = _pitch_base(left_label, right_label)
    _pitch_arcs(fig)
    groups = [('Open play', shots['component'].isin(['op_centre', 'op_left', 'op_right']), COLORS['primary']),
              ('Crosses', shots['component'] == 'cross', COLORS['info']),
              ('Set pieces', shots['component'] == 'set_piece', COLORS['accent'])]
    for name, mask, colour in groups:
        d = shots[mask & (shots['x'] >= 0.5)]
        if d.empty:
            continue
        # Misses first, goals last: goals draw on top of the pile, and the
        # legend (which copies the first point's symbol) shows a circle.
        d = d.sort_values('is_goal', kind='stable')
        fig.add_trace(go.Scatter(
            x=d['hx'], y=d['x'], mode='markers', name=name,
            marker=dict(size=5 + 28 * np.sqrt(d['xg'].clip(lower=0)), color=colour,
                        symbol=np.where(d['is_goal'], 'star', 'circle'),
                        opacity=0.72, line=dict(width=0.5, color='white')),
            customdata=np.stack([d['player'].fillna(''), d['xg'], d['result'].fillna(''),
                                 d['last_action'].fillna('')], axis=-1),
            hovertemplate='%{customdata[0]}<br>xG %{customdata[1]:.2f} · %{customdata[2]}'
                          '<br>Set up by: %{customdata[3]}<extra></extra>'))
    return fig


_HEAT_XEDGES = np.linspace(0, 1, 11)        # ~6.8m wide cells
_HEAT_YEDGES = np.linspace(0.5, 1, 9)       # ~6.6m deep cells


def _heat_grid(shots, n_matches):
    d = shots[shots['x'] >= 0.5]
    grid, _, _ = np.histogram2d(d['hx'], d['x'], bins=[_HEAT_XEDGES, _HEAT_YEDGES],
                                weights=d['xg'])
    return grid.T / max(n_matches, 1)        # rows = depth, cols = width


def _half_pitch_heat(grid, zmax, left_label, right_label):
    """Where the xG comes from, per match, on a coarse grid. Stays readable
    however many shots pile up."""
    fig = _pitch_base(left_label, right_label)
    xc = (_HEAT_XEDGES[:-1] + _HEAT_XEDGES[1:]) / 2
    yc = (_HEAT_YEDGES[:-1] + _HEAT_YEDGES[1:]) / 2
    z = np.where(grid > 0, grid, np.nan)
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=z, zmin=0, zmax=max(zmax, 0.05),
        colorscale=[[0, 'rgba(55,0,60,0.08)'], [1, 'rgba(55,0,60,0.95)']],
        colorbar=dict(title=dict(text='xG/match', side='right'), thickness=10, len=0.6),
        hovertemplate='%{z:.2f} xG per match<extra></extra>', xgap=1, ygap=1))
    _pitch_arcs(fig)
    fig.update_layout(showlegend=False)
    return fig


def _profile_bars(profile, league, which, labels, high_is_good):
    comps = SHOT_COMPONENTS
    idx = [100 * profile[which][c] / league[c] if league[c] > 0 else 100 for c in comps]
    good = [(v >= 100) == high_is_good for v in idx]
    fig = go.Figure(go.Bar(
        y=[labels[c] for c in comps], x=idx, orientation='h',
        marker_color=[COLORS['success'] if g else COLORS['danger'] for g in good],
        text=[f"{v:.0f}" for v in idx], textposition='outside',
        customdata=[profile[which][c] for c in comps],
        hovertemplate='%{y}: %{x:.0f} (%{customdata:.2f} xG per match)<extra></extra>'))
    fig.add_vline(x=100, line_dash='dash', line_color='#999')
    fig.update_layout(template='plotly_white', height=280, showlegend=False,
                      xaxis=dict(range=[0, max(160, max(idx) * 1.15)], title='Index (100 = league average)',
                                 fixedrange=True),
                      yaxis=dict(autorange='reversed', fixedrange=True),
                      margin=dict(l=10, r=20, t=10, b=40))
    return fig


_SHOW_LABEL = {'all': 'shots', 'on_target': 'shots on target', 'goals': 'goals',
               'big': 'big chances'}


def _filter_map_shots(s, show, routes):
    comp_ok = []
    if 'open' in routes:
        comp_ok += ['op_centre', 'op_left', 'op_right']
    if 'cross' in routes:
        comp_ok.append('cross')
    if 'sp' in routes:
        comp_ok.append('set_piece')
    s = s[s['component'].isin(comp_ok)]
    if show == 'goals':
        s = s[s['is_goal']]
    elif show == 'on_target':
        s = s[s['result'].isin(['Goal', 'SavedShot'])]
    elif show == 'big':
        s = s[s['xg'] >= SHOT_BIG_CHANCE_XG]
    return s


@callback(
    [Output('sp-status', 'children'), Output('sp-map-created', 'figure'),
     Output('sp-map-conceded', 'figure'), Output('sp-bars-att', 'figure'),
     Output('sp-bars-def', 'figure'), Output('sp-team-table', 'data'),
     Output('sp-map-count', 'children')],
    [Input('visit-shot-profiles', 'data'), Input('sp-team', 'value'),
     Input('sp-show', 'value'), Input('sp-routes', 'value'),
     Input('sp-recent', 'value'), Input('sp-display', 'value')],
    prevent_initial_call=True
)
def update_shot_profiles(visit, team_name, show, routes, recent, display):
    _need_visit(visit)
    data = get_data()
    intel, note = _shot_intel_status(data)
    if intel is None:
        b = _blank_fig('Waiting for shot data')
        return note, b, b, b, b, [], ''
    teams_df = data.get('teams_df')
    tid = None
    if teams_df is not None and team_name:
        m = teams_df.loc[teams_df['name'] == team_name, 'id']
        tid = int(m.iloc[0]) if len(m) else None
    prof = intel['profiles'].get(tid) if tid is not None else None
    table = prepare_table_data(intel['team_table'], list(intel['team_table'].columns))
    if prof is None:
        b = _blank_fig('No shot data for this team yet')
        return note, b, b, b, b, table, ''

    s = intel['shots']
    s = s[s['component'] != 'penalty']
    team_s = s[(s['team_id'] == tid) | (s['opp_id'] == tid)]
    # Recency: this team's last N matches (maps only; the profile bars stay
    # season-long and shrunk, so a hot or cold patch can't swing them)
    order = (team_s[['match_id', 'kickoff']].drop_duplicates()
             .sort_values(['kickoff', 'match_id']))
    try:
        recent = int(recent or 0)
    except (TypeError, ValueError):
        recent = 0
    match_ids = order['match_id'].tolist()
    if recent > 0:
        match_ids = match_ids[-recent:]
    # Matches with zero shots on one side still count as matches played
    n_matches = len(match_ids) if recent > 0 else max(prof['n'], len(match_ids))
    team_s = team_s[team_s['match_id'].isin(match_ids)]
    team_s = _filter_map_shots(team_s, show or 'all', routes or [])
    made = team_s[team_s['team_id'] == tid]
    conc = team_s[team_s['opp_id'] == tid]

    # Opponents attack the top goal; the attacker's right (picture right) is
    # the defending team's left.
    if display == 'heat':
        g_made, g_conc = _heat_grid(made, n_matches), _heat_grid(conc, n_matches)
        zmax = float(max(g_made.max(initial=0), g_conc.max(initial=0)))
        created = _half_pitch_heat(g_made, zmax, 'Their left', 'Their right')
        conceded = _half_pitch_heat(g_conc, zmax, 'Their right', 'Their left')
    else:
        created = _half_pitch_fig(made, 'Their left', 'Their right')
        conceded = _half_pitch_fig(conc, 'Their right', 'Their left')

    def _summ(d):
        return (f"{len(d)} {_SHOW_LABEL.get(show, 'shots')}, "
                f"{int(d['is_goal'].sum())} goals, {d['xg'].sum():.1f} xG")
    scope = f"last {n_matches} matches" if recent > 0 else f"{n_matches} matches this season"
    count = (f"Showing {scope}. Created: {_summ(made)}. Conceded: {_summ(conc)}. "
             "Filters change the pitch maps only; the profile bars below always use the "
             "whole season.")
    att = _profile_bars(prof, intel['league'], 'created', COMP_LABEL_ATT, high_is_good=True)
    dfn = _profile_bars(prof, intel['league'], 'conceded', COMP_LABEL_DEF, high_is_good=False)
    return note, created, conceded, att, dfn, table, count


def _upcoming_by_team(data, n_gws):
    """{team_id: [(gw, opp_id, venue), ...]} over the next n gameweeks."""
    anchor = data.get('fixture_anchor_gw')
    if anchor is None:
        anchor = (data.get('next_gw_num') or 1) - 1
    scheduled = {f.get('event') for f in data.get('fixtures_data') or []}
    gws = [g for g in range(anchor + 1, anchor + 1 + n_gws) if g <= 38 and g in scheduled]
    out = {}
    for f in data.get('fixtures_data') or []:
        g = f.get('event')
        if g in gws:
            out.setdefault(f['team_h'], []).append((g, f['team_a'], 'H'))
            out.setdefault(f['team_a'], []).append((g, f['team_h'], 'A'))
    return gws, out


def _matchup_heatmap(intel, data, gws, fixtures_by_team):
    teams_df = data.get('teams_df')
    names = dict(zip(teams_df['id'], teams_df['name']))
    shorts = dict(zip(teams_df['id'], teams_df['short_name']))
    prof, league = intel['profiles'], intel['league']
    rows = []
    for tid in names:
        cells, fits = [], []
        for g in gws:
            fx = [f for f in fixtures_by_team.get(tid, []) if f[0] == g]
            if not fx:
                cells.append((None, 'BLANK', 'No fixture'))
                continue
            vals, texts, hovers = [], [], []
            for _, opp, venue in fx:
                res = shot_matchup(prof.get(tid), prof.get(opp), league)
                opp_s = shorts.get(opp, '?') + ('' if venue == 'H' else ' (A)')
                if res is None:
                    texts.append(opp_s)
                    hovers.append(f"vs {names.get(opp, '?')}: no shot data")
                    continue
                base, style, comps, edges = res
                fit = 100 * (style / base - 1) if base > 0 else 0.0
                vals.append(fit)
                best = max(edges, key=edges.get)
                worst = min(edges, key=edges.get)
                tag = f" {COMP_TAG[best]}" if edges[best] >= 0.03 else ''
                texts.append(f"{opp_s}<br>{fit:+.0f}%{tag}")
                hovers.append(
                    f"vs {names.get(opp, '?')} ({venue}): style fit {fit:+.0f}%<br>"
                    f"Expected npxG {style:.2f} vs {base:.2f} on strength alone<br>"
                    f"Biggest edge: {COMP_LABEL_ATT[best]} ({edges[best]:+.2f} xG)<br>"
                    f"Biggest drag: {COMP_LABEL_ATT[worst]} ({edges[worst]:+.2f} xG)")
            z = float(np.mean(vals)) if vals else None
            if z is not None:
                fits.append(z)
            if len(texts) == 1:
                text = texts[0]
            else:   # double gameweek: both opponents, averaged fit
                text = ' + '.join(t.split('<br>')[0] for t in texts)
                if z is not None:
                    text += f"<br>{z:+.0f}%"
            cells.append((z, text, '<br>—<br>'.join(hovers)))
        rows.append((names[tid], cells, float(np.mean(fits)) if fits else -999))
    rows.sort(key=lambda r: r[2], reverse=True)
    z = [[c[0] for c in r[1]] for r in rows]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"GW{g}" for g in gws], y=[r[0] for r in rows],
        text=[[c[1] for c in r[1]] for r in rows],
        customdata=[[c[2] for c in r[1]] for r in rows],
        texttemplate='%{text}', textfont=dict(size=11, color='#333333'),
        colorscale=[[0, '#ef9a9a'], [0.5, '#f7f7f7'], [1, '#81c784']],
        zmin=-25, zmax=25, zmid=0, showscale=False, xgap=2, ygap=2,
        hovertemplate='<b>%{y}</b> %{x}<br>%{customdata}<extra></extra>'))
    cell_w = 84 if len(gws) > 3 else 104
    fig.update_layout(
        template='plotly_white', height=40 * len(rows) + 70,
        width=118 + cell_w * max(len(gws), 1) + 28,
        xaxis=dict(side='top', fixedrange=True), yaxis=dict(autorange='reversed', fixedrange=True),
        font=dict(family=FONT_FAMILY, size=12), margin=dict(l=110, r=18, t=40, b=10))
    return fig


def _fixture_index_text(fx, shorts, idx_fn):
    parts = []
    for g, opp, venue in fx:
        parts.append(f"{shorts.get(opp, '?')}{'' if venue == 'H' else '(A)'} {idx_fn(opp):.2f}")
    return ', '.join(parts)


@callback(
    [Output('mu-status', 'children'), Output('mu-validation', 'children'),
     Output('mu-heatmap', 'figure'), Output('mu-sp-table', 'data'),
     Output('mu-flank-table', 'data')],
    [Input('visit-matchups', 'data'), Input('mu-gws', 'value'),
     Input('mu-position', 'value'), Input('mu-minutes', 'value')],
    prevent_initial_call=True
)
def update_matchups(visit, n_gws, position, min_minutes):
    _need_visit(visit)
    data = get_data()
    intel, note = _shot_intel_status(data)
    if intel is None:
        return note, None, _blank_fig('Waiting for shot data'), [], []
    try:
        n_gws = max(1, min(6, int(n_gws or 4)))
    except (TypeError, ValueError):
        n_gws = 4
    min_minutes = 0 if min_minutes is None else min_minutes
    position = position or 'All'

    v = intel.get('validation')
    if v:
        better = v['mae_style'] < v['mae_base']
        val = html.P([
            html.Strong("Does it work? "),
            f"Tested out-of-sample on {v['n']} team-matches (each gameweek predicted only from earlier "
            f"ones): style-aware error {v['mae_style']:.3f} xG vs {v['mae_base']:.3f} for strength alone; "
            f"closer in {v['closer_pct']:.0f}% of cases. ",
            ("The style model is currently adding accuracy." if better else
             "The style model isn't beating strength alone yet, so treat these as tie-breakers, "
             "not headline calls."),
        ], style={'fontSize': '14px', 'color': COLORS['text_dark'], 'margin': '0'})
    else:
        val = html.P("Accuracy check: appears once most teams have 3+ matches of data, so each gameweek "
                     "can be predicted from earlier ones and scored. Until then, treat style fit as a "
                     "tie-breaker.", style={'fontSize': '14px', 'color': COLORS['text_light'], 'margin': '0'})

    gws, fx_by_team = _upcoming_by_team(data, n_gws)
    if not gws:
        return note, val, _blank_fig('No upcoming fixtures'), [], []
    heat = _matchup_heatmap(intel, data, gws, fx_by_team)

    teams_df = data.get('teams_df')
    shorts = dict(zip(teams_df['id'], teams_df['short_name']))
    prof, league = intel['profiles'], intel['league']
    dfa = data.get('df_active', pd.DataFrame())
    pl = intel['players']
    if pl is None or pl.empty or dfa.empty:
        return note, val, heat, [], []
    fpl_cols = ['id', 'web_name', 'team', 'team_name', 'position', 'price', 'ownership']
    j = pl.dropna(subset=['fpl_id']).merge(dfa[fpl_cols], left_on='fpl_id', right_on='id', how='inner')
    j = j[j['minutes'] >= min_minutes]
    j = j[j['position'].isin(['DEF', 'MID', 'FWD'])]
    if position != 'All':
        j = j[j['position'] == position]

    def sp_idx(opp):
        p = prof.get(opp)
        return p['conceded']['set_piece'] / league['set_piece'] if p and league['set_piece'] > 0 else 1.0

    sp_rows = []
    for r in j.itertuples(index=False):
        fx = fx_by_team.get(r.team, [])
        if not fx or not (r.sp_threat_90 > 0):
            continue
        sp_rows.append({
            'web_name': r.web_name, 'team_name': r.team_name, 'position': r.position,
            'price': r.price, 'sp_threat_90': r.sp_threat_90, 'head_pct': r.head_pct,
            'sp_created': int(r.sp_created), 'ownership': r.ownership,
            'fixtures': _fixture_index_text(fx, shorts, sp_idx),
            'horizon': r.sp_threat_90 * sum(sp_idx(o) for _, o, _ in fx),
        })
    sp_rows = sorted(sp_rows, key=lambda d: d['horizon'], reverse=True)[:40]

    s = intel['shots']
    np_s = s[s['component'] != 'penalty']
    ch_counts = np_s.groupby(['player_id', 'channel']).size().unstack(fill_value=0)
    flank_rows = []
    for r in j[j['side'].isin(['left', 'right'])].itertuples(index=False):
        fx = fx_by_team.get(r.team, [])
        if not fx:
            continue
        comp = 'op_' + r.side

        def fl_idx(opp, comp=comp):
            p = prof.get(opp)
            return p['conceded'][comp] / league[comp] if p and league[comp] > 0 else 1.0

        n_sh = int(ch_counts.loc[r.player_id].sum()) if r.player_id in ch_counts.index else 0
        n_side = int(ch_counts.loc[r.player_id].get(r.side, 0)) if n_sh else 0
        share = n_side / n_sh if n_sh >= 5 else 0.5   # too few shots: assume half
        xgi90 = (r.npxg_90 if pd.notna(r.npxg_90) else 0) + (r.xa_90 if pd.notna(r.xa_90) else 0)
        mults = [1 + share * (fl_idx(o) - 1) for _, o, _ in fx]
        flank_rows.append({
            'web_name': r.web_name, 'team_name': r.team_name, 'position': r.position,
            'side_label': _SIDE_LABEL.get(r.side, ''), 'price': r.price, 'xgi_90': xgi90,
            'channel_pct': 100 * share, 'ownership': r.ownership,
            'fixtures': _fixture_index_text(fx, shorts, fl_idx),
            'edge_pct': 100 * (float(np.mean(mults)) - 1), 'horizon': xgi90 * sum(mults),
        })
    flank_rows = sorted(flank_rows, key=lambda d: d['horizon'], reverse=True)[:40]
    sp_df = pd.DataFrame(sp_rows)
    fl_df = pd.DataFrame(flank_rows)
    return (note, val, heat,
            prepare_table_data(sp_df, list(sp_df.columns)) if not sp_df.empty else [],
            prepare_table_data(fl_df, list(fl_df.columns)) if not fl_df.empty else [])


@callback(
    [Output('cq-status', 'children'), Output('cq-scatter', 'figure'), Output('cq-table', 'data')],
    [Input('visit-chance-quality', 'data'), Input('cq-position', 'value'),
     Input('cq-team', 'value'), Input('cq-minutes', 'value')],
    prevent_initial_call=True
)
def update_chance_quality(visit, position, team, min_minutes):
    _need_visit(visit)
    data = get_data()
    intel, note = _shot_intel_status(data)
    if intel is None:
        return note, _blank_fig('Waiting for shot data'), []
    dfa = data.get('df_active', pd.DataFrame())
    pl = intel['players']
    if pl is None or pl.empty or dfa.empty:
        return note, _blank_fig('No player data yet'), []
    min_minutes = 0 if min_minutes is None else min_minutes
    fpl_cols = ['id', 'web_name', 'team_name', 'position', 'price', 'ownership']
    j = pl.dropna(subset=['fpl_id']).merge(dfa[fpl_cols], left_on='fpl_id', right_on='id', how='inner')
    j = j[(j['minutes'] >= min_minutes) & j['position'].isin(['DEF', 'MID', 'FWD'])]
    if position and position != 'All':
        j = j[j['position'] == position]
    if team and team != 'All':
        j = j[j['team_name'] == team]
    j = j.copy()
    j['side_label'] = j['side'].map(_SIDE_LABEL)
    sc = j[(j['shots'] >= 3) & j['xg_per_shot'].notna()]
    if sc.empty:
        fig = _blank_fig('Not enough shots for these filters')
    else:
        fig = px.scatter(sc, x='shots_90', y='xg_per_shot', color='position', size='npxg_90',
                         size_max=26, hover_name='web_name',
                         hover_data={'team_name': True, 'npxg_90': ':.2f', 'box_pct': ':.0f',
                                     'big_90': ':.2f', 'shots_90': ':.2f', 'xg_per_shot': ':.3f',
                                     'position': False},
                         labels={'shots_90': 'Shots per 90', 'xg_per_shot': 'xG per shot',
                                 'npxg_90': 'npxG/90', 'box_pct': 'Box %', 'big_90': 'Big chances/90',
                                 'team_name': 'Team'},
                         color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                             'FWD': COLORS['info']})
        fig.add_hline(y=float(sc['xg_per_shot'].median()), line_dash='dash', line_color='#999')
        fig.add_vline(x=float(sc['shots_90'].median()), line_dash='dash', line_color='#999')
        fig.update_layout(template='plotly_white', height=420, font=dict(family=FONT_FAMILY),
                          legend=dict(orientation='h', y=1.02, yanchor='bottom', x=0.5, xanchor='center'))
    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'shots_90', 'npxg_90',
            'xg_per_shot', 'box_pct', 'big_90', 'head_pct', 'chances_90', 'xa_90', 'cross_pct',
            'sp_threat_90', 'side_label', 'ownership']
    table = prepare_table_data(j.sort_values('npxg_90', ascending=False).head(150), cols)
    return note, fig, table


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Fantasy Premier League Analytics Hub")
    print("=" * 60)
    print(f"  Current Gameweek: {current_gw['name'] if current_gw else 'N/A'}")
    print(f"  Players loaded: {len(df)}")
    print(f"  Active players: {len(df_active)}")
    print("  Player histories loading in background...")
    print("  Features: Home, DefCon Bonus, Consistency, Defensive,")
    print("  xG/xA, Value, Form, Clean Sheets, Fixtures,")
    print("  Differentials, Captain Optimiser, Transfer Trends,")
    print("  Squad Builder")
    print("=" * 60)
    print("\n  Starting server...")
    print("  Open http://127.0.0.1:8053 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8053)