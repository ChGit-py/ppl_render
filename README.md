# Fantasy Premier League Analytics Hub

An FPL analytics app built with **Dash** and **Plotly**. It projects every player's points, plans transfers and chips, finds captains and differentials, breaks down fixtures and underlying stats, and keeps a running check of its own projections against FPL's expected points (xP).

It runs as a website on desktop and mobile, and can be added to a phone's home screen as an app ("FPL Room").

---

## Contents

- [What's in the app](#whats-in-the-app)
- [How the projections work](#how-the-projections-work)
- [Model Lab: how accuracy is tracked](#model-lab-how-accuracy-is-tracked)
- [Data sources](#data-sources)
- [Running it locally](#running-it-locally)
- [Deploying on Render](#deploying-on-render)
- [Configuration](#configuration)
- [How the data stays fresh](#how-the-data-stays-fresh)
- [Project files](#project-files)
- [Disclaimer](#disclaimer)

---

## What's in the app

Pages are grouped into five sections. On desktop the sections run down a slim rail on the left; on a phone they sit in a tab bar along the bottom. Each section's pages are tabs across the top.

### 🏠 Home

| Page | What it does |
|---|---|
| **Overview** | Next deadline with a live countdown (days, hours, minutes) and shortcuts to the key tools; a Model vs FPL summary; season numbers (managers, gameweek average and highest score, chips played); player spotlights; chip usage; and the **Rank Tracker**, which shows the points of managers at five distances above and below your rank (scaled to your rank, e.g. 25k–250k around 268k) and how far you are from a target rank. |
| **Deadline** | A pre-deadline check for your team: lineup vs the best possible lineup, captain expected points and ceiling, availability flags and price risk, all in one pass. |
| **Model Lab** | How close the app's projections and FPL's xP have come to the points players actually scored, gameweek by gameweek and player by player. See [Model Lab](#model-lab-how-accuracy-is-tracked). |

### 👕 My Team

| Page | What it does |
|---|---|
| **My Squad** | Load your squad by FPL team ID to see form, fixtures and injury flags, with suggested moves that use only your free transfers (never a hit). |
| **Transfer Planner** | Compare any two players over the next gameweek or next five, including the cost of a hit. Plus a **multi-week plan** that works out when to transfer, when to roll a free transfer and when a −4 hit pays for itself. |
| **Chip Planner** | Projects your actual squad gameweek by gameweek (doubles and blanks included) to find the best weeks for Bench Boost and Triple Captain, and flags Free Hit weeks. |
| **Squad Builder** | Builds the highest-scoring legal 15-man squad for a budget, with players you want to include or exclude, optionally aimed at a chip gameweek. |
| **Mini-League Rivals** | Loads the top squads in a classic league and shows threats (they own, you don't), leverage (you own, they don't), everyone's captain and the chips each rival has used. |

### 👥 Players

| Page | What it does |
|---|---|
| **Projections** | Projected points for every player: the next gameweek with the parts behind each number (minutes, goals, assists, clean sheet, bonus and so on) next to FPL's xP, or the next five gameweeks week by week. |
| **Captaincy** | Ranks captain picks by projected points, with **Protect**, **Chase** and **Ceiling** modes (the last ranks by the chance of a 15+ point haul). |
| **Differentials** | Low-owned players ranked by projected points, with a **Swing** column (what owning him gains you on everyone who doesn't), chance of a big week and start security. Max ownership can be set to any value from 0 to 100. |
| **Value** | Points returned per £1m spent. |
| **Form** | Who is scoring above or below their season average right now. |
| **Transfer Trends** | Net transfers, likely price rises and falls, ownership change since the last deadline, and price alerts for your own squad. |

### 📅 Fixtures

| Page | What it does |
|---|---|
| **Ticker** | Every team's upcoming fixtures, colour-coded by difficulty, with blanks and doubles marked. |
| **Outlook** | Expected goals scored and clean-sheet chances per fixture, so attack and defence runs are judged separately. |
| **Difficulty** | Fixture difficulty by team and player, plus **Fixture Swings**: teams whose fixtures are about to get easier (positive, green) or harder (negative, pink). |
| **Matchups** | Whether an attack's style suits the opponent's weaknesses (set pieces, crosses, central play, each flank). |

### 📊 Stats

| Page | What it does |
|---|---|
| **xG & xA** | Goals and assists against the chances behind them. |
| **Underlying** | Per-90 underlying numbers: who is involved in the most and best chances. |
| **Chance Quality** | xG per shot, box share and big chances: who gets into the positions that produce hauls. |
| **Shot Profiles** | Shot maps and attack/defence profiles for every team, from Understat shot locations. |
| **Clean Sheets** | Clean sheets and goals conceded per 90. |
| **Expected CS** | Projected clean sheets per team over a chosen number of gameweeks. |
| **DEFCON Bonus** | Defensive contribution rates against the bonus thresholds (10+ for DEF, 12+ for MID/FWD). |
| **DEFCON Consistency** | How often each player actually hits the threshold in a match. |
| **DEFCONs** | Defensive contributions against what minutes and position would predict. |

---

## How the projections work

Every player gets a projected points figure for the next gameweek and the next five, built from separate parts:

1. **Minutes** – how likely he is to start and how long he plays, from recent matches and FPL's availability flags.
2. **Goals and assists** – from his xG and xA, adjusted for the opponent.
3. **Fixtures** – team attack and defence ratings fitted from xG and, where available, bookmaker prices.
4. **Clean sheets and goals conceded** – from the expected goals of each fixture.
5. **Defensive contributions, saves and bonus.**
6. **Upside** – a simulation that gives each player's chance of a big score (used by Ceiling and the haul columns).

Doubles count twice and blanks count zero. Each part was tested on a replay of the 2025/26 season. The replay engine is still in the code for offline tuning (see the comment above `_gather_backtest_inputs` in `fpl_dash_updated.py`) but isn't shown in the app.

---

## Model Lab: how accuracy is tracked

Before every deadline the app saves its projection and **FPL's xP** for every player. The save repeats on each refresh in the last 24 hours and stops at the deadline, so the version that's scored is the last one before it. Once FPL marks the gameweek finished, both are checked against the points awarded.

The page shows:

- who is ahead overall, and in how many gameweeks;
- **typical error** (root-mean-square miss), **closer to the score** (share of players), **top 10 picks** (average points of each side's ten highest projections) and **captain pick** (points from each side's number-one projection);
- a chart and table by gameweek;
- **player by player** for any scored gameweek, with search.

The saves go to a separate **private GitHub repository** (see [Configuration](#configuration)), so they survive Render's restarts and deploys. Only gameweeks locked in at the deadline are scored, and nothing is recalculated afterwards. Tracking began with **Gameweek 10** of 2026/27.

---

## Data sources

| Source | Used for | Key needed? |
|---|---|---|
| [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Players, prices, fixtures, ownership, live points, squads, leagues, standings | No |
| [Understat](https://understat.com/) | Shot-level xG with locations and situations (Shot Profiles, Matchups, Chance Quality) | No |
| [football-data.co.uk](https://www.football-data.co.uk/) | Pre-match bookmaker odds for team ratings (this season and last) | No |
| [The Odds API](https://the-odds-api.com/) | Live match odds and anytime-scorer odds | Yes, `ODDS_API_KEY` (optional) |
| GitHub API | Saving and reading Model Lab projections | Yes, `FPL_LOG_GITHUB_TOKEN` (optional) |

Without the optional keys the app still works: projections use the model's own goal chances, and Model Lab saves stay on the server, which Render clears on every deploy.

---

## Running it locally

Requires **Python 3.9+**.

```bash
git clone https://github.com/ChGit-py/ppl_render.git
cd ppl_render
pip install -r requirements.txt
python fpl_dash_updated.py
```

Then open **http://127.0.0.1:8053**. The first start downloads everything, which takes a few minutes; after that it starts from the saved cache.

---

## Deploying on Render

The app is deployed on [Render](https://render.com/) and **auto-deploys from `main`**.

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | for example `gunicorn fpl_dash_updated:server --workers 1 --timeout 120` |

`server` is the Flask server exposed by the Dash app (`server = app.server`).

> [!NOTE]
> On Render's **free plan**:
> - The instance sleeps after inactivity, which delays the first request by up to a minute. An uptime monitor (e.g. UptimeRobot) pinging the site every few minutes keeps it awake.
> - Memory is capped at **512 MB**. Use **one worker**: every extra gunicorn worker loads its own full copy of the data.
> - The disk is reset on every deploy, so anything worth keeping (Model Lab projections) is saved to GitHub.

---

## Configuration

Set these as environment variables (in Render: **Service → Environment**).

| Variable | Default | Purpose |
|---|---|---|
| `FPL_LOG_GITHUB_TOKEN` | – | GitHub fine-grained token with **Contents: Read and write** on the log repository. Enables permanent Model Lab tracking. |
| `FPL_LOG_REPO` | – | The log repository, e.g. `ChGit-py/fpl-model-log`. Use a separate repo so saves don't trigger deploys. |
| `FPL_LOG_BRANCH` | repo default | Branch to save to. |
| `ODDS_API_KEY` | – | The Odds API key for live match and anytime-scorer odds. |
| `ODDS_REFRESH_INTERVAL_HOURS` | `6` | How often match odds are refreshed. |
| `FOOTBALL_DATA_REFRESH_HOURS` | `12` | How often football-data.co.uk odds are refreshed. |
| `SHOTS_REFRESH_INTERVAL_HOURS` | `6` | How often new Understat shot data is fetched. |
| `FPL_PRICE_TRACKER_MINUTES` | `20` | How often the price-change tracker runs. |
| `FPL_REFRESH_CHECK_SECONDS` | `300` | How often the background timer checks whether a refresh is due. |
| `FPL_MAX_FREE_TRANSFERS` | `5` | Most free transfers that can be banked. |
| `FPL_SNAPSHOT_DB` | `fpl_snapshots.db` | Path of the local SQLite database. |

<details>
<summary><strong>Model tuning parameters</strong> (advanced; defaults are the tested values)</summary>

These change how the projections are calculated. Leave them alone unless you've tested a change on the season replay.

| Variable | Default |
|---|---|
| `FPL_MINUTES_MODEL` | `v2` |
| `FPL_MINS_HALFLIFE` | `3` |
| `FPL_MINS_WINDOW` | `8` |
| `FPL_SHRINK_K` | `450` |
| `FPL_TEAM_FORM_K` | `6` |
| `FPL_SHOT_PROFILE_K` | `6` |
| `FPL_FDR_RATIO` | `1.18` |
| `FPL_RATING_FDR_SCALE` | `0.2` |
| `FPL_RATING_HALF_LIFE_DAYS` | `150` |
| `FPL_RATING_LAST_SEASON_WEIGHT` | `0.5` |
| `FPL_RATING_PRIOR_MATCHES` | `8` |
| `FPL_MARKET_RATING_PAST` | `2` |
| `FPL_MARKET_RATING_UPCOMING` | `6` |
| `MARKET_WEIGHT` | `0.7` |
| `MARKET_WEIGHT_ANCHORED` | `0.3` |
| `PLAYER_ODDS` | `1` |
| `PLAYER_ODDS_WEIGHT` | `0.5` |
| `PLAYER_ODDS_REGIONS` | `uk` |
| `PLAYER_ODDS_MIN_CREDITS` | `60` |
| `FPL_CAPTAIN_TIEBREAK_PTS` | `1.0` |
| `FPL_LEAGUE_GAP_SD` | `12` |
| `FPL_PLANNER_DECAY` | `0.9` |
| `FPL_PLANNER_FT_VALUE` | `1.5` |
| `FPL_PRICE_LIKELY_PCT` | `95` |
| `FPL_PRICE_PRIOR` | `1` |
| `FPL_BACKTEST_BUDGET` | `45` |
| `FPL_BACKTEST_CHUNK` | `120` |
| `FPL_BACKTEST_MIN_MINS` | `90` |
| `FPL_SWEEP_BUDGET` | `70` |
| `SHOTS_MAX_MATCHES_PER_RUN` | `60` |
| `FPL_PHOTO_PREFIX` | – |

</details>

---

## How the data stays fresh

- **Every 3 hours** a background timer refreshes everything, even with no visitors. It runs in two phases: core FPL data first, then the heavier stats and projections.
- **Every 20 minutes** the price tracker updates; Understat shots every 6 hours; odds every 6–12 hours.
- Pages render only when opened, and again only if the data has changed since.
- Data is cached in `fpl_cache.pkl`, so restarts are quick; the cache is rebuilt after each deploy.
- Daily player snapshots, projection logs and stored shots live in `fpl_snapshots.db` (SQLite).

---

## Project files

```text
ppl_render/
├── fpl_dash_updated.py   # The app: data loading, models and all pages
├── requirements.txt      # Python dependencies
├── assets/               # Logo, favicon and home-screen app icons/manifest
├── fpl_cache.pkl         # Data cache (rebuilt automatically)
├── fpl_snapshots.db      # Local SQLite store (snapshots, logs, shots)
├── fpl_dash.py           # Older version, not used by the deployed app
├── test.py               # Older copy, not used by the deployed app
└── test2.py              # Older copy, not used by the deployed app
```

---

## Disclaimer

This is an independent project. It is not affiliated with or endorsed by the Premier League or Fantasy Premier League. All data belongs to its respective sources.
