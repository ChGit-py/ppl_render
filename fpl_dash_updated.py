"""
Fantasy Premier League Analytics Dashboard
Track Defensive Contributions, Expected Metrics, and Value Picks
"""

import requests
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, ctx, clientside_callback
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


def fetch_player_history(player_id):
    """Fetch individual player's match-by-match history."""
    try:
        response = requests.get(f"{FPL_BASE_URL}/element-summary/{player_id}/", timeout=10)
        response.raise_for_status()
        return response.json().get('history', [])
    except Exception as e:
        print(f"Error fetching player {player_id}: {e}")
        return []


def calculate_bonus_consistency(player_ids, player_thresholds, min_minutes=60):
    """
    Calculate defcon bonus hit rate for multiple players.
    player_thresholds: dict of player_id -> bonus threshold (10 for DEF, 12 for MID/FWD)
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

        # Count games hitting bonus threshold (position-aware)
        threshold = player_thresholds.get(player_id, 10)
        bonus_games = [g for g in qualifying_games if g.get('defensive_contribution', 0) >= threshold]

        # Calculate stats
        defcon_values = [g.get('defensive_contribution', 0) for g in qualifying_games]

        stats = {
            'qualifying_games': len(qualifying_games),
            'bonus_games': len(bonus_games),
            'hit_rate': (len(bonus_games) / len(qualifying_games)) * 100 if qualifying_games else 0,
            'avg_defcon': sum(defcon_values) / len(defcon_values) if defcon_values else 0,
            'max_defcon': max(defcon_values) if defcon_values else 0,
            'min_defcon': min(defcon_values) if defcon_values else 0,
            'threshold': threshold,
        }
        return player_id, stats

    # Use threading for faster fetching (capped at 8 to limit memory on free tier)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_player, pid): pid for pid in player_ids}
        for future in as_completed(futures):
            player_id, stats = future.result()
            if stats:
                results[player_id] = stats

    return results


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

    position_defcon_rates = df[df['minutes'] > 450].groupby('position')['defcon_per_90'].mean()
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

    # --- Set-piece duties (P=penalties, C=corners/indirect FKs, F=direct FKs) ---
    # First or second in a duty order is a real xGI driver hiding in the payload.
    def _duty(col, tag):
        if col not in df.columns:
            return pd.Series('', index=df.index)
        order = pd.to_numeric(df[col], errors='coerce')
        return order.map(lambda v: f"{tag}{int(v)}" if pd.notna(v) and v <= 2 else '')

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
    """True once at least one gameweek has been played."""
    return any(e.get('finished') for e in data.get('events', []))


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
# FETCH AND PROCESS DATA €” WRAPPED FOR AUTO-REFRESH
# =============================================================================

import time
import threading

# Global data store
DATA = {
    'last_refresh': 0,
    'refreshing': False,
}
DATA_LOCK = threading.Lock()
REFRESH_INTERVAL = 3 * 60 * 60  # 3 hours in seconds
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fpl_cache.pkl')

# Keys to persist in cache (excludes transient flags like 'refreshing')
_CACHE_KEYS = [
    'bootstrap_data', 'df', 'df_active', 'current_gw', 'next_gw',
    'total_managers', 'fixtures_data', 'teams_df', 'fixture_difficulty',
    'player_histories', 'sorted_teams', 'next_gw_num', 'last_refresh',
    'heavy_loaded', 'fixture_anchor_gw', 'season_started', 'season_label',
]

# Bump whenever the shape of cached data changes, or at a season rollover.
# A mismatch (or an over-age cache) forces a clean fetch instead of serving
# last season's teams and players from disk.
CACHE_VERSION = 3
MAX_CACHE_AGE = REFRESH_INTERVAL


def save_cache():
    """Persist current DATA to disk so next startup is instant."""
    try:
        payload = {k: DATA[k] for k in _CACHE_KEYS if k in DATA}
        payload['_cache_version'] = CACHE_VERSION
        payload['_season_label'] = SEASON['label']
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

        print(f"  Cache found ({age_hrs:.1f}h old, season {cached.get('_season_label')}) — loading...")
        cached.pop('_cache_version', None)
        cached.pop('_season_label', None)
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
        for col in ['qualifying_games', 'bonus_games', 'hit_rate',
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
        fixture_difficulty = calculate_fixture_difficulty(fixtures_data, teams_df, fixture_anchor_gw)
        print(f"  Calculated fixture difficulty for {len(fixture_difficulty)} teams")

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

        # Transfer trend / price prediction
        print("Computing price change likelihood scores...")
        df_active['price_change_likelihood'] = df_active.apply(
            lambda r: estimate_price_change_likelihood(r, total_managers), axis=1
        )

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
                print(f"  Trend deltas computed vs snapshot from {baseline_date}")
            else:
                df_active['own_delta_7d'] = np.nan
                df_active['price_delta_7d'] = np.nan
                print("  No prior snapshot yet — trend deltas will appear from tomorrow")
        except Exception as e:
            print(f"  Snapshot store unavailable: {e}")
            df_active['own_delta_7d'] = np.nan
            df_active['price_delta_7d'] = np.nan

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
        defcon_positions = SEASON['defcon_positions']
        consistency_players = df_active[
            (df_active['minutes'] >= 200) &
            (df_active['position'].isin(defcon_positions))
            ]['id'].tolist()

        print(f"  Fetching data for {len(consistency_players)} players...")
        consistency_thresholds = dict(zip(
            df_active.loc[df_active['id'].isin(consistency_players), 'id'],
            df_active.loc[df_active['id'].isin(consistency_players), 'position'].map(SEASON['thresholds'])
        ))
        consistency_data = calculate_bonus_consistency(consistency_players, consistency_thresholds)
        print(f"  Retrieved data for {len(consistency_data)} players")

        df_active['qualifying_games'] = df_active['id'].map(
            lambda x: consistency_data.get(x, {}).get('qualifying_games'))
        df_active['bonus_games'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('bonus_games'))
        df_active['hit_rate'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('hit_rate'))
        df_active['avg_defcon_qualifying'] = df_active['id'].map(
            lambda x: consistency_data.get(x, {}).get('avg_defcon'))
        df_active['max_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('max_defcon'))
        df_active['min_defcon_game'] = df_active['id'].map(lambda x: consistency_data.get(x, {}).get('min_defcon'))

        # Free consistency data to reclaim memory before next batch
        del consistency_data, consistency_thresholds, consistency_players
        gc.collect()

        # Home/Away splits
        print("Fetching player histories for captain & home/away analysis...")
        captain_candidates = df_active[
            (df_active['minutes'] >= 450) &
            (df_active['position'].isin(SEASON['outfield_positions']))
            ].nlargest(100, 'form')['id'].tolist()

        print(f"  Fetching match history for {len(captain_candidates)} captain candidates...")
        player_histories = fetch_player_history_batch(captain_candidates)
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
        <title>FPL Analytics Hub</title>
        {%favicon%}
        {%css%}
        <style>
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
                height: calc(100vh - 64px);
                position: relative;
                overflow: hidden;
            }

            /* --- Sidebar --- */
            #sidebar {
                width: 230px;
                min-width: 230px;
                background: #ffffff;
                border-right: 1px solid #e0e0e0;
                height: 100%;
                overflow-y: auto;
                overflow-x: hidden;
                z-index: 500;
                transition: transform 0.28s ease;
                flex-shrink: 0;
                padding-bottom: 24px;
            }

            /* Group label */
            .nav-group-label {
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #37003c;
                padding: 18px 20px 6px 20px;
                margin: 0;
            }

            /* Nav item button */
            .nav-item {
                display: flex;
                align-items: center;
                gap: 10px;
                width: 100%;
                background: none;
                border: none;
                border-left: 3px solid transparent;
                padding: 9px 16px 9px 17px;
                font-size: 13.5px;
                font-family: Arial, sans-serif;
                font-weight: 500;
                color: #444;
                cursor: pointer;
                text-align: left;
                transition: background 0.15s, color 0.15s, border-color 0.15s;
                line-height: 1.3;
            }

            .nav-item:hover {
                background: #f5f5f5;
                color: #37003c;
            }

            .nav-item.active {
                background: #f0e6f6;
                color: #37003c;
                border-left-color: #37003c;
                font-weight: 700;
            }

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
            #hamburger-btn {
                display: none;
                background: none;
                border: none;
                color: white;
                font-size: 22px;
                cursor: pointer;
                padding: 4px 10px 4px 0;
                line-height: 1;
            }

            /* ================================================================
               TABLET & BELOW  (≤ 900px)
            ================================================================ */
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

                .dash-dropdown { min-width: 100% !important; }

                [style*="display: flex"] > div[style*="minWidth"] {
                    min-width: 100% !important;
                    flex: 1 1 100% !important;
                    padding-left: 0 !important;
                    padding-right: 0 !important;
                    margin-bottom: 10px;
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


def build_stat_card(title, value, subtitle=None, color=COLORS['primary'], image_code=None):
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
        }) if subtitle else None
    ], style=STAT_CARD_STYLE)


def build_player_spotlight(player, title, metric_label, metric_value):
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
        ])
    ], style={'flex': '1'})

    image_section = player_photo_img(player, style={
        'width': '70px', 'height': '90px',
        'borderRadius': '8px', 'alignSelf': 'center'
    })

    return html.Div([text_section, image_section], style={
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

app.layout = html.Div([
    # Interval + stores
    dcc.Interval(id='refresh-interval', interval=2 * 60 * 1000, n_intervals=0),
    dcc.Store(id='active-page', data='home'),
    dcc.Store(id='active-page-local'),
    dcc.Store(id='sidebar-open', data=False),
    dcc.Store(id='my-squad-store', data=None),

    # Header
    html.Div([
        html.Div([
            html.Div([
                # Hamburger (visible on mobile only via CSS)
                html.Button('☰', id='hamburger-btn', n_clicks=0),
                html.Img(src="/assets/premier_league_logo.png",
                         style={'height': '40px', 'marginRight': '12px'}),
                html.Span(f"Fantasy Premier League {SEASON['label']}",
                          style={'backgroundColor': COLORS['secondary'], 'color': COLORS['primary'],
                                 'padding': '6px 12px', 'borderRadius': '6px', 'fontWeight': '800',
                                 'fontSize': '18px', 'marginRight': '12px'}),
                html.Span("Analytics Hub", style={'color': 'white', 'fontSize': '20px', 'fontWeight': '600'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Span(id='gw-status-text',
                          style={'color': 'rgba(255,255,255,0.8)', 'fontSize': '13px'}),
                html.Span(" | ", style={'color': 'rgba(255,255,255,0.5)', 'fontSize': '13px'}),
                html.Span(id='last-updated-text',
                          style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '12px'})
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                  'maxWidth': '100%', 'margin': '0 auto', 'padding': '0 20px'})
    ], style={'backgroundColor': COLORS['primary'], 'padding': '12px 0', 'position': 'sticky',
              'top': '0', 'zIndex': '1000', 'boxShadow': '0 2px 8px rgba(0,0,0,0.15)'}),

    # Body: sidebar + content
    html.Div([

        # Mobile overlay
        html.Div(id='sidebar-overlay', n_clicks=0),

        # Sidebar nav
        html.Div(
            id='sidebar',
            className='sidebar',
            children=[
                # OVERVIEW
                html.P('Overview', className='nav-group-label'),
                html.Button('Home',
                            id='nav-home', className='nav-item active', n_clicks=0),

                # DEFENSIVE
                html.P('Defensive', className='nav-group-label'),
                html.Button('DEFCON Bonus',
                            id='nav-defcon-bonus', className='nav-item', n_clicks=0),
                html.Button('DEFCON: Consistency',
                            id='nav-bonus-consistency', className='nav-item', n_clicks=0),
                html.Button('DEFCONS',
                            id='nav-defcon', className='nav-item', n_clicks=0),
                html.Button('Clean Sheets',
                            id='nav-cs', className='nav-item', n_clicks=0),

                # ATTACKING
                html.P('Attacking', className='nav-group-label'),
                html.Button('Expected Goals & Assists',
                            id='nav-xg', className='nav-item', n_clicks=0),
                html.Button('Underlying Numbers',
                            id='nav-underlying', className='nav-item', n_clicks=0),

                # VALUE & FORM
                html.P('Value & Form', className='nav-group-label'),
                html.Button('Value Analysis',
                            id='nav-value', className='nav-item', n_clicks=0),
                html.Button('Form Tracker',
                            id='nav-form', className='nav-item', n_clicks=0),

                # PLANNING
                html.P('Squad Planning', className='nav-group-label'),
                html.Button('Fixture Ticker',
                            id='nav-fixture-ticker', className='nav-item', n_clicks=0),
                html.Button('Fixture Difficulty',
                            id='nav-fixtures', className='nav-item', n_clicks=0),
                html.Button('Differentials',
                            id='nav-differentials', className='nav-item', n_clicks=0),
                html.Button('Captain Optimiser',
                            id='nav-captain', className='nav-item', n_clicks=0),
                html.Button('Transfer Trends',
                            id='nav-transfers', className='nav-item', n_clicks=0),
                html.Button('My Squad',
                            id='nav-my-squad', className='nav-item', n_clicks=0),
                html.Button('Squad Builder',
                            id='nav-squad-builder', className='nav-item', n_clicks=0),
            ]
        ),

        # Content area — all pages live here, show/hide via callback
        html.Div(id='content-area', children=[

            # Carried-over-stats warning; populated by callback, empty in-season.
            # Must live INSIDE the content column — as a direct child of the
            # #app-body flex row it renders as a third column and shoves every
            # page off the right-hand edge of the viewport.
            html.Div(id='stale-stats-banner'),

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
                                dcc.Input(id='bonus-minutes', type='number', value=450, min=0, step=50,
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{defcon_vs_bonus} < 0', 'column_id': 'defcon_vs_bonus'},
                                 'backgroundColor': '#ffebee'}
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
                            "A player averaging the threshold per 90 minutes might be inconsistent (20 one week, 0 the next) vs someone who reliably hits it every game."
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
                                dcc.Input(id='consistency-games', type='number', value=5, min=1, step=1,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                            html.Div([
                                html.Label("Min. minutes",
                                           style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}),
                                dcc.Input(id='consistency-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Chart - Hit Rate Distribution
                    html.Div([
                        html.H3("Bonus Hit Rate by Player", style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            f"Percentage of qualifying games (60+ mins) where player hit their bonus threshold "
                            f"(DEF: {DEF_THR}+, MID/FWD: {MID_THR}+).",
                            style={'color': COLORS['text_light']}),
                        dcc.Graph(id='consistency-bar')
                    ], style=CARD_STYLE),

                    # Scatter - Hit Rate vs Avg Defcon
                    html.Div([
                        html.H3("Consistency vs Average Output",
                                style={'color': COLORS['primary'], 'marginBottom': '8px'}),
                        html.P(
                            "Compare hit rate (consistency) against average defcon in qualifying games. Top right = high output AND consistent.",
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
                                {'name': 'Games', 'id': 'qualifying_games', 'type': 'numeric'},
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{hit_rate} >= 25 && {hit_rate} < 50', 'column_id': 'hit_rate'},
                                 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{hit_rate} < 25', 'column_id': 'hit_rate'},
                                 'backgroundColor': '#ffebee'}
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
                                dcc.Input(id='defcon-minutes', type='number', value=200, min=0, step=50,
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{defcon_diff} < 0', 'column_id': 'defcon_diff'},
                                 'backgroundColor': '#ffebee'}
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
                                dcc.Input(id='xg-minutes', type='number', value=200, min=0, step=50,
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
                                 'backgroundColor': '#e8f5e9'}
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
                                dcc.Input(id='under-minutes', type='number', value=450, min=0, step=50,
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
                                {'if': {'filter_query': '{xgi_diff_per_90} < 0.00', 'column_id': 'xgi_diff_per_90'}, 'backgroundColor': '#ffebee'},
                                {'if': {'filter_query': '{xgi_diff_per_90} > 0.00', 'column_id': 'xgi_diff_per_90'}, 'backgroundColor': '#e8f5e9'},
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
                                dcc.Input(id='value-minutes', type='number', value=200, min=0, step=50,
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
                                dcc.Input(id='form-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{form_vs_season} < -1', 'column_id': 'form_vs_season'},
                                 'backgroundColor': '#ffebee'}
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
                                dcc.Input(id='cs-minutes', type='number', value=200, min=0, step=50,
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
                                    min=3, max=38, step=1, value=6,
                                    marks={3: '3', 5: '5', 8: '8', 10: '10', 15: '15',
                                           20: '20', 30: '30', 38: 'All'},
                                    tooltip={'placement': 'bottom', 'always_visible': False}
                                )
                            ], style={'flex': '2', 'minWidth': '260px', 'padding': '0 10px'}),
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
                               "cell text is hidden to keep the grid readable — hover any "
                               "cell for the fixture.",
                               style={'color': COLORS['text_light'], 'marginBottom': '12px'}),
                        dcc.Graph(id='ticker-heatmap', config={'displayModeBar': False})
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
                            html.Span("Remaining Season", style={'backgroundColor': COLORS['secondary'],
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
                                dcc.Input(id='fdr-minutes', type='number', value=200, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
                    ], style=CARD_STYLE),

                    # Team FDR Chart
                    html.Div([
                        html.H3("Team Fixture Difficulty (Remaining Season)",
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
                                {'name': 'Remaining Fixtures', 'id': 'fixture_string'},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{avg_fdr_5} <= 2.5', 'column_id': 'avg_fdr_5'},
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{avg_fdr_5} > 2.5 && {avg_fdr_5} <= 3.5',
                                        'column_id': 'avg_fdr_5'}, 'backgroundColor': '#fff8e1'},
                                {'if': {'filter_query': '{avg_fdr_5} > 3.5', 'column_id': 'avg_fdr_5'},
                                 'backgroundColor': '#ffebee'}
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
                                dcc.Input(id='diff-minutes', type='number', value=450, min=0, step=50,
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
                                {'name': 'Avg FDR', 'id': 'avg_fdr_5', 'type': 'numeric',
                                 'format': {'specifier': '.2f'}},
                                {'name': 'Remaining', 'id': 'fixture_string'},
                            ],
                            sort_action='native',
                            page_size=20,
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data=TABLE_STYLE_DATA,
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
                                {'if': {'filter_query': '{ownership} <= 5', 'column_id': 'ownership'},
                                 'backgroundColor': '#e8f5e9'},
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
                                dcc.Input(id='cap-minutes', type='number', value=450, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
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
                                {'name': 'Captain Score', 'id': 'captain_score', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{next_att_fdr} >= 3.7', 'column_id': 'next_att_fdr'},
                                 'backgroundColor': '#ffebee'},
                                {'if': {'filter_query': '{next_venue} = H', 'column_id': 'next_venue'},
                                 'color': COLORS['success'], 'fontWeight': '600'},
                                {'if': {'filter_query': '{next_venue} = A', 'column_id': 'next_venue'},
                                 'color': COLORS['danger'], 'fontWeight': '600'},
                                {'if': {'filter_query': '{avail_pct} < 100', 'column_id': 'avail_pct'},
                                 'backgroundColor': '#ffebee', 'fontWeight': '600'},
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
                                dcc.Input(id='xfer-minutes', type='number', value=0, min=0, step=50,
                                          style={'width': '100%', 'padding': '8px', 'borderRadius': '4px',
                                                 'border': '1px solid #ccc'})
                            ], style={'flex': '1', 'minWidth': '100px', 'padding': '0 10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'flex-end'})
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
                                style={'color': COLORS['primary'], 'marginBottom': '16px'}),
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
                                {'name': 'Price Chg', 'id': 'price_change_likelihood', 'type': 'numeric',
                                 'format': {'specifier': '.1f'}},
                                {'name': 'Own \u03947d', 'id': 'own_delta_7d', 'type': 'numeric',
                                 'format': {'specifier': '+.1f'}},
                                {'name': '\u00a3 \u03947d', 'id': 'price_delta_7d', 'type': 'numeric',
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
                                 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{net_transfers_gw} < 0', 'column_id': 'net_transfers_gw'},
                                 'backgroundColor': '#ffebee'},
                                {'if': {'filter_query': '{price_change_likelihood} >= 50',
                                        'column_id': 'price_change_likelihood'}, 'backgroundColor': '#e8f5e9'},
                                {'if': {'filter_query': '{price_change_likelihood} <= -50',
                                        'column_id': 'price_change_likelihood'}, 'backgroundColor': '#ffebee'},
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
                                    id='sq-budget', min=75, max=105, step=0.5, value=83,
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
                                    ],
                                    value='ppg', clearable=False
                                )
                            ], style={'flex': '2', 'minWidth': '220px', 'padding': '0 10px'}),

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
               style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '13px', 'margin': '0'})
    ], style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'textAlign': 'center'})

], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': COLORS['background'], 'margin': '0', 'padding': '0'})


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
            "Player stats shown are ",
            html.Strong("final 2025/26 numbers"),
            ". FPL carries them over until the "
            f"{next_gw.get('name', 'Gameweek 1')} deadline"
            f"{f' ({deadline})' if deadline else ''}, when they reset to zero. "
            "Prices, clubs and fixtures are current — minutes, points, clean "
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
    'home', 'defcon-bonus', 'bonus-consistency', 'defcon',
    'xg', 'underlying', 'value', 'form', 'cs',
    'fixture-ticker', 'fixtures', 'differentials',
    'captain', 'transfers', 'my-squad', 'squad-builder',
]

# --- NAV: clicks → active-page store ---
@callback(
    Output('active-page', 'data'),
    [Input(f'nav-{p}', 'n_clicks') for p in ALL_PAGES],
    Input('active-page-local', 'data'),
    prevent_initial_call=True
)
def set_active_page(*args):
    # Last arg is the localStorage value (from active-page-local)
    local_val = args[-1]
    if ctx.triggered_id == 'active-page-local':
        # On load: restore from localStorage if valid
        return local_val if local_val in ALL_PAGES else 'home'
    if ctx.triggered_id:
        return ctx.triggered_id[4:]  # strip 'nav-' prefix
    return 'home'


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


# --- PAGES: active-page store → show/hide each page div ---
@callback(
    [Output(f'page-{p}', 'style') for p in ALL_PAGES],
    Input('active-page', 'data')
)
def show_active_page(active):
    return [
        {'display': 'block'} if p == active else {'display': 'none'}
        for p in ALL_PAGES
    ]


# --- NAV ITEMS: highlight active nav button ---
@callback(
    [Output(f'nav-{p}', 'className') for p in ALL_PAGES],
    Input('active-page', 'data')
)
def highlight_nav(active):
    return [
        'nav-item active' if p == active else 'nav-item'
        for p in ALL_PAGES
    ]


# --- SIDEBAR: toggle open/closed on mobile ---
@callback(
    Output('sidebar-open', 'data'),
    [Input('hamburger-btn', 'n_clicks'),
     Input('sidebar-overlay', 'n_clicks'),
     Input('active-page', 'data')],
    State('sidebar-open', 'data'),
    prevent_initial_call=True
)
def toggle_sidebar(hamburger_clicks, overlay_clicks, active_page, is_open):
    trigger = ctx.triggered_id
    if trigger == 'hamburger-btn':
        return not is_open
    return False  # overlay click or page navigation → close


@callback(
    [Output('sidebar', 'className'),
     Output('sidebar-overlay', 'className')],
    Input('sidebar-open', 'data')
)
def apply_sidebar_classes(is_open):
    if is_open:
        return 'sidebar sidebar-open', 'overlay-open'
    return 'sidebar', ''



@callback(
    [Output('gw-status-text', 'children'), Output('last-updated-text', 'children')],
    Input('refresh-interval', 'n_intervals')
)
def update_refresh_status(n):
    """Check data freshness every 2 minutes. Trigger background refresh if stale."""
    check_and_refresh()
    current_gw_now = DATA.get('current_gw')
    gw_text = f"Data as of {current_gw_now['name']}" if current_gw_now else "N/A"
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
    Input('refresh-interval', 'n_intervals')
)
def update_home_tab(n):
    """Rebuild the entire Home tab from current DATA so it reflects refreshed data."""
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
    best_value_now = df_now[df_now['minutes'] > 450].nlargest(1, 'points_per_million').iloc[0] if len(
        df_now[df_now['minutes'] > 450]) > 0 else None
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

    # Chip usage
    chips = current_gw_now.get('chip_plays', []) if current_gw_now else []
    chip_sum = ', '.join(
        [f"{chip_name_map.get(c['chip_name'], c['chip_name'])}: {c['num_played']:,}" for c in chips]
    ) if chips else "No data yet"
    total_chips = sum(c['num_played'] for c in chips) if chips else 0

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
                               font=dict(family='Arial, sans-serif'),
                               yaxis=dict(range=[0, max(c_counts) * 1.15]))
    else:
        chip_fig = go.Figure()
        chip_fig.add_annotation(text="No chip data available yet", xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=COLORS['text_light']))
        chip_fig.update_layout(template='plotly_white', height=300)

    # Position breakdown chart
    pos_data = df_now[df_now['minutes'] > 450]
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
                              yaxis_title='Value', xaxis_title='Position', font=dict(family='Arial, sans-serif'),
                              yaxis=dict(range=[0, position_stats['points_per_million'].max() * 10 * 1.2]))
    else:
        pos_fig = go.Figure()

    # Build and return layout
    return html.Div([
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
            html.Div([build_stat_card("Highest GW Score", f"{highest_gw}", "This gameweek", color=COLORS['success'])],
                     style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card("Next Deadline",
                                      next_gw_now['name'].replace('Gameweek ', 'GW') if next_gw_now else "N/A",
                                      datetime.fromisoformat(
                                          next_gw_now['deadline_time'].replace('Z', '+00:00')).strftime(
                                          '%a %d %b, %H:%M') if next_gw_now else "")],
                     style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}),

        html.Div([
            html.Div([build_stat_card(
                "Most Captained",
                most_cap['web_name'] if most_cap is not None else "N/A",
                f"{most_cap['team_name']} - £{most_cap['price']:.1f}m" if most_cap is not None else "Data available after deadline",
                color=COLORS['primary'],
                image_code=most_cap if most_cap is not None else None
            )], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card(
                "Most Vice-Captained",
                most_vice['web_name'] if most_vice is not None else "N/A",
                f"{most_vice['team_name']} - £{most_vice['price']:.1f}m" if most_vice is not None else "Data available after deadline",
                color=COLORS['accent'],
                image_code=most_vice if most_vice is not None else None
            )], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
            html.Div([build_stat_card(
                "Chips Used This GW",
                f"{total_chips:,}" if total_chips > 0 else "N/A",
                chip_sum,
                color=COLORS['info']
            )], style={'flex': '1', 'minWidth': '200px', 'padding': '0 10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '0 -10px 40px -10px'}),

        html.Div([
            html.H2("Chip Usage This Gameweek", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Number of managers activating each chip", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([dcc.Graph(figure=chip_fig, config={'displayModeBar': False})], style=CARD_STYLE),

        html.Div([
            html.H2("Player Spotlights", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Top performers across key metrics", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([
            build_player_spotlight(top_scorer_now, "Top Scorer", "Total Points",
                                   f"{int(top_scorer_now['total_points'])}" if top_scorer_now is not None else "N/A"),
            build_player_spotlight(most_selected_now, "Most Selected", "Ownership",
                                   f"{most_selected_now['ownership']:.1f}%" if most_selected_now is not None else "N/A"),
            build_player_spotlight(best_value_now, "Best Value", "Points/£m",
                                   f"{best_value_now['points_per_million']:.2f}" if best_value_now is not None else "N/A"),
            build_player_spotlight(top_form_now, "In Form", "Form Rating",
                                   f"{top_form_now['form']:.1f}" if top_form_now is not None else "N/A"),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'marginBottom': '40px'}),

        html.Div([
            html.H2("Position Breakdown", style={'color': COLORS['primary'], 'margin': '0 0 4px 0'}),
            html.P("Average points and value by position", style={'color': COLORS['text_light']})
        ], style={'marginBottom': '24px'}),

        html.Div([dcc.Graph(figure=pos_fig, config={'displayModeBar': False})], style=CARD_STYLE),

    ], style={'padding': '20px 0'})


# RANK CONGESTION TOOL
@callback(
    Output('rank-result', 'children'),
    Input('rank-check-btn', 'n_clicks'),
    State('rank-your', 'value'),
    State('rank-rival', 'value'),
    prevent_initial_call=True
)
def check_rank_gap(n_clicks, your_rank, rival_rank):
    if not your_rank or not rival_rank:
        return html.P("Please enter both ranks.", style={'color': COLORS['danger']})

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
                      style={'color': COLORS['danger'], 'marginTop': '12px'})

    gap = abs(rival_points - your_points)
    rank_gap = abs(your_rank - rival_rank)
    pts_per_1k_ranks = round((gap / rank_gap) * 1000, 1) if rank_gap > 0 else 0
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
                html.H3(f"{pts_per_1k_ranks} pts", style={'color': COLORS['success'], 'margin': '0',
                                                           'fontSize': '28px', 'fontWeight': '700'}),
                html.P("per 1,000 rank places", style={'color': COLORS['text_light'],
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
     Input('bonus-minutes', 'value')]
)
def update_bonus(position, team, max_price, min_minutes):
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
                              font=dict(family='Arial, sans-serif'))

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
                          font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'defcon_vs_bonus',
            'bonus_rate', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'defcon_vs_bonus'), cols)

    return scatter_fig, bar_fig, table_data


# BONUS CONSISTENCY
@callback(
    [Output('consistency-bar', 'figure'), Output('consistency-scatter', 'figure'), Output('consistency-table', 'data')],
    [Input('consistency-position', 'value'), Input('consistency-team', 'value'), Input('consistency-price', 'value'),
     Input('consistency-games', 'value'), Input('consistency-minutes', 'value'),
     Input('refresh-interval', 'n_intervals')]
)
def update_consistency(position, team, max_price, min_games, min_minutes, _n):
    # Filter to players with consistency data
    data = get_data()
    filtered = data['df_active'][data['df_active']['qualifying_games'].notna()].copy()

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
                          font=dict(family='Arial, sans-serif'))

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
        font=dict(family='Arial, sans-serif')
    )

    # Table data
    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'qualifying_games', 'bonus_games', 'hit_rate',
            'avg_defcon_qualifying', 'max_defcon_game', 'min_defcon_game', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'hit_rate'), cols)

    return bar_fig, scatter_fig, table_data


# DEFCON
@callback(
    [Output('defcon-scatter', 'figure'), Output('defcon-table', 'data')],
    [Input('defcon-position', 'value'), Input('defcon-team', 'value'), Input('defcon-price', 'value'),
     Input('defcon-minutes', 'value')]
)
def update_defcon(position, team, max_price, min_minutes):
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
    fig.update_layout(template='plotly_white', height=400, font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'defcon', 'defcon_per_90', 'expected_defcon',
            'defcon_diff', 'ownership']
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
                     labels={'expected_goals': 'Expected Goals', 'goals_scored': 'Goals Scored'},
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                         'FWD': COLORS['info']})
    if len(filtered) > 0:
        max_val = max(filtered['goals_scored'].max(), filtered['expected_goals'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color='#999'),
                                 name='Expected'))
    fig.update_layout(template='plotly_white', height=400, font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'goals_scored', 'expected_goals', 'xg_diff', 'assists',
            'expected_assists', 'xa_diff', 'ownership']
    table_data = prepare_table_data(filtered.sort_values('xg_diff').head(50), cols)

    return fig, table_data


# UNDERLYING NUMBERS
@callback(
    [Output('under-scatter', 'figure'), Output('under-table', 'data')],
    [Input('under-position', 'value'), Input('under-team', 'value'), Input('under-price', 'value'),
     Input('under-minutes', 'value')]
)
def update_underlying(position, team, max_price, min_minutes):
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
    fig.update_layout(template='plotly_white', height=400, font=dict(family='Arial, sans-serif'))

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
     Input('value-minutes', 'value')]
)
def update_value(position, team, max_price, min_minutes):
    filtered = filter_data(position, team, max_price, min_minutes)
    filtered = filtered.dropna(subset=['points_per_million'])

    fig = px.scatter(filtered, x='price', y='total_points', color='position', size='ownership',
                     hover_name='web_name', hover_data=['team_name', 'points_per_million', 'form'],
                     labels={'price': 'Price', 'total_points': 'Total Points'},
                     color_discrete_map={'GKP': '#666', 'DEF': COLORS['primary'], 'MID': COLORS['accent'],
                                         'FWD': COLORS['info']})
    fig.update_layout(template='plotly_white', height=400, font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'points_per_million', 'form', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'points_per_million'), cols)

    return fig, table_data


# FORM
@callback(
    [Output('form-chart', 'figure'), Output('form-table', 'data')],
    [Input('form-position', 'value'), Input('form-team', 'value'), Input('form-price', 'value'),
     Input('form-minutes', 'value')]
)
def update_form(position, team, max_price, min_minutes):
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
                      font=dict(family='Arial, sans-serif'))

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
                     hover_name='web_name', hover_data=['price', 'clean_sheets', 'goals_conceded'],
                     labels={'team_name': 'Club', 'gc_per_90': 'Goals Conceded per 90',
                             'cs_per_90': 'Clean Sheet per 90'})
    fig.update_layout(template='plotly_white', height=400, font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'minutes', 'clean_sheets', 'cs_per_90', 'goals_conceded',
            'gc_per_90', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'cs_per_90'), cols)

    return fig, table_data


# FIXTURE DIFFICULTY
@callback(
    [Output('fdr-team-bar', 'figure'), Output('fdr-scatter', 'figure'), Output('fdr-table', 'data')],
    [Input('fdr-position', 'value'), Input('fdr-team', 'value'), Input('fdr-price', 'value'),
     Input('fdr-minutes', 'value')]
)
def update_fdr(position, team, max_price, min_minutes):
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
                          yaxis_title='Average FDR (Remaining Season)', showlegend=False,
                          yaxis=dict(range=[0, 5.5]),
                          font=dict(family='Arial, sans-serif'))

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
        font=dict(family='Arial, sans-serif')
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
     Input('refresh-interval', 'n_intervals')]
)
def update_fixture_ticker(sort_by, num_gws, n):
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
    num_gws = int(num_gws or 6)
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

    fig.update_layout(
        template='plotly_white',
        height=height,
        font=dict(family='Arial, sans-serif', size=12),
        xaxis=dict(side='top', tickangle=0 if n_cols <= 15 else -45,
                   fixedrange=True, tickfont=dict(size=tick_font)),
        yaxis=dict(autorange='reversed', fixedrange=True, tickfont=dict(size=14)),
        margin=dict(l=110, r=20, t=60, b=10),
    )

    if showing_all:
        title = "Fixture Ticker: Remainder of the Season"
    else:
        title = (f"Fixture Ticker: Next {len(remaining_gws)} Gameweeks "
                 f"(GW{remaining_gws[0]}\u2013GW{remaining_gws[-1]})")

    return [fig, title]


# --- OWNERSHIP DIFFERENTIALS ---
@callback(
    [Output('diff-scatter', 'figure'), Output('diff-bar', 'figure'), Output('diff-table', 'data')],
    [Input('diff-position', 'value'), Input('diff-team', 'value'), Input('diff-price', 'value'),
     Input('diff-max-own', 'value'), Input('diff-minutes', 'value')]
)
def update_differentials(position, team, max_price, max_own, min_minutes):
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
        median_ppg = get_data()['df_active'][get_data()['df_active']['minutes'] >= 450]['ppg'].median()
        scatter_fig.add_hline(y=median_ppg, line_dash='dash', line_color='#999',
                              annotation_text=f'Median PPG ({median_ppg:.1f})', annotation_position='top right')
    scatter_fig.update_layout(template='plotly_white', height=400, xaxis_title='Ownership %',
                              yaxis_title='Points per Game',
                              font=dict(family='Arial, sans-serif'))

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
                          font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'total_points', 'form', 'ppg',
            'expected_goal_involvements', 'ownership', 'top_eo', 'own_delta_7d',
            'differential_score', 'avg_fdr_5', 'fixture_string']
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


# --- CAPTAIN Optimiser ---
@callback(
    [Output('cap-bar', 'figure'), Output('cap-ha-scatter', 'figure'), Output('cap-table', 'data')],
    [Input('cap-position', 'value'), Input('cap-team', 'value'), Input('cap-price', 'value'),
     Input('cap-minutes', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_captain(position, team, max_price, min_minutes, _n):
    filtered = filter_data(position, team, max_price, min_minutes, positions_allowed=SEASON['outfield_positions'])
    filtered = filtered.dropna(subset=['captain_score'])
    filtered = filtered[filtered['captain_score'] > 0]

    if len(filtered) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No captain candidates found. Try reducing the min. minutes",
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                 font=dict(size=14, color=COLORS['text_light']))
        empty_fig.update_layout(template='plotly_white', height=400)
        return empty_fig, empty_fig, []

    top_20 = filtered.nlargest(20, 'captain_score')
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=top_20['web_name'], y=top_20['captain_score'],
        marker_color=[COLORS['success'] if v == 'H' else COLORS['info'] for v in top_20['next_venue']],
        text=[f"{s:.1f}" for s in top_20['captain_score']],
        textposition='outside',
        hovertemplate=(
            '%{x}<br>'
            'Score: %{y:.2f}<br>'
            'vs %{customdata[0]} (%{customdata[1]})<br>'
            'FDR: %{customdata[2]}<br>'
            'Form: %{customdata[3]:.1f}<extra></extra>'
        ),
        customdata=top_20[['next_opponent', 'next_venue', 'next_fdr', 'form']].values
    ))
    bar_fig.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45,
                          yaxis_title='Captain Score', showlegend=False,
                          yaxis=dict(range=[0, top_20['captain_score'].max() * 1.1]),
                          font=dict(family='Arial, sans-serif'))

    ha_filtered = filtered.dropna(subset=['home_ppg', 'away_ppg'])
    ha_scatter = px.scatter(
        ha_filtered, x='away_ppg', y='home_ppg', color='position',
        hover_name='web_name',
        hover_data=['team_name', 'next_opponent', 'next_venue', 'price'],
        color_discrete_map={'DEF': COLORS['primary'], 'MID': COLORS['accent'], 'FWD': COLORS['info']}
    )
    if len(ha_filtered) > 0:
        max_val = max(ha_filtered['home_ppg'].max(), ha_filtered['away_ppg'].max(), 1)
        ha_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                        line=dict(dash='dash', color='#999'), name='Equal'))
    ha_scatter.update_layout(template='plotly_white', height=400, xaxis_title='Away PPG', yaxis_title='Home PPG',
                             font=dict(family='Arial, sans-serif'))

    cols = ['web_name', 'team_name', 'position', 'price', 'captain_score', 'ep_next', 'form', 'ppg',
            'xgi_per_90', 'next_att_fdr', 'venue_ppg', 'start_rate', 'avail_pct',
            'set_pieces', 'top_eo',
            'expected_goal_involvements', 'next_opponent', 'next_venue', 'next_fdr',
            'home_ppg', 'away_ppg', 'bps_per_90', 'ownership']
    table_data = prepare_table_data(filtered.nlargest(50, 'captain_score'), cols)

    return bar_fig, ha_scatter, table_data


# --- TRANSFER TRENDS ---
@callback(
    [Output('xfer-risers-bar', 'figure'), Output('xfer-fallers-bar', 'figure'),
     Output('xfer-scatter', 'figure'), Output('xfer-table', 'data')],
    [Input('xfer-position', 'value'), Input('xfer-team', 'value'),
     Input('xfer-price', 'value'), Input('xfer-minutes', 'value')]
)
def update_transfers(position, team, max_price, min_minutes):
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
                             font=dict(family='Arial, sans-serif'))

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
                              font=dict(family='Arial, sans-serif'))

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
                              font=dict(family='Arial, sans-serif'))

    sorted_by_activity = filtered.copy()
    sorted_by_activity['abs_net'] = sorted_by_activity['net_transfers_gw'].abs()
    cols = ['web_name', 'team_name', 'position', 'price', 'transfers_in_gw', 'transfers_out_gw',
            'net_transfers_gw', 'transfer_ratio', 'price_change_likelihood', 'own_delta_7d',
            'price_delta_7d', 'cost_change_start', 'form', 'ownership']
    table_data = prepare_table_data(sorted_by_activity.nlargest(50, 'abs_net'), cols)

    return risers_fig, fallers_fig, scatter_fig, table_data


# --- SQUAD BUILDER ---
@callback(
    Output('sq-results', 'children'),
    Input('sq-build-btn', 'n_clicks'),
    [State('sq-budget', 'value'),
     State('sq-objective', 'value'),
     State('sq-must-include', 'value'),
     State('sq-must-exclude', 'value')],
    prevent_initial_call=True
)
def build_squad(n_clicks, budget, objective, must_include, must_exclude):
    import traceback
    try:
        data = get_data()
        df_now = data['df_active'].copy()

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
                        style={'color': COLORS['danger'], 'fontSize': '15px', 'textAlign': 'center', 'margin': '0'}
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
            font=dict(family='Arial, sans-serif'),
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
            font=dict(family='Arial, sans-serif')
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
                    {'if': {'filter_query': '{position} = "FWD"', 'column_id': 'position'}, 'color': COLORS['info'],
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
                html.P(f"Error: {str(e)}", style={'color': COLORS['danger'], 'fontWeight': '600'}),
                html.Pre(traceback.format_exc(),
                         style={'fontSize': '12px', 'color': COLORS['text_light'], 'whiteSpace': 'pre-wrap'})
            ], style=CARD_STYLE)
        ])


# =============================================================================
# MY SQUAD CALLBACK
# =============================================================================

STATUS_LABELS = {
    'a': ('✓ Fit',        COLORS['success']),
    'd': ('⚠ Doubt',      COLORS['warning']),
    'i': ('✗ Injured',    COLORS['danger']),
    's': ('✗ Suspended',  COLORS['danger']),
    'n': ('✗ N/A',        COLORS['danger']),
    'u': ('✗ N/A',        COLORS['danger']),
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
                   style={'color': COLORS['danger'], 'fontWeight': '600'})
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
                   style={'color': COLORS['danger'], 'fontWeight': '600'})
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
                   style={'color': COLORS['danger']})
        ], style=CARD_STYLE)])

    squad_df['pick_position'] = squad_df['id'].map(lambda x: pick_map[x]['position'])
    squad_df['is_captain']    = squad_df['id'].map(lambda x: pick_map[x].get('is_captain', False))
    squad_df['is_vice']       = squad_df['id'].map(lambda x: pick_map[x].get('is_vice_captain', False))
    squad_df['selling_price'] = squad_df['id'].map(
        lambda x: pick_map[x].get('selling_price', pick_map[x].get('now_cost', 0))
    ) / 10
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
                    ("In the Bank",  f"£{bank_m:.1f}m",       COLORS['success']),
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
            fdr_color = (COLORS['success'] if fdr and fdr <= 2.5
                         else COLORS['warning'] if fdr and fdr <= 3.5
                         else COLORS['danger'])

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
                html.Td(f"£{r.get('selling_price', r.get('price', 0)):.1f}m",
                        style={'padding': '10px 12px'}),
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
                    style={'color': COLORS['danger'], 'marginBottom': '12px'}),
            html.Div(items)
        ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORS["danger"]}'})

    return html.Div([
        manager_card,
        starters_section,
        bench_section,
        injury_section or html.Div(),
    ])


# =============================================================================
# RUN
# =============================================================================

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