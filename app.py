from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import requests
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import re
from zoneinfo import ZoneInfo
import os

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app) # Enable CORS for all routes, important for frontend/backend communication

# --- Configuration ---
BOOK_WEIGHTS_PATH = "book_weights.csv" # Path to your book_weights.csv
OUTPUT_DIR = "." # Current directory for output files

# Ensure output directory exists (though for current directory, it's always true)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions (adapted for Flask environment) ---
def fetch_json(url):
    """Fetch JSON data from a URL with headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.actionnetwork.com",
        "Origin": "https://www.actionnetwork.com",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def load_book_mapping(csv_path):
    """Load book ID to name and weight mapping from CSV."""
    try:
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in ['Book ID', 'Name', 'Weight']):
            book_id_mapping = dict(zip(df['Book ID'].astype(str), df['Name']))
            book_weight_mapping = dict(zip(df['Book ID'].astype(str), df['Weight'].astype(str)))
            print(f"Loaded {len(book_id_mapping)} book mappings from {csv_path}")
            return book_id_mapping, book_weight_mapping
        print(f"Warning: Expected columns missing in {csv_path}. Available: {list(df.columns)}")
        return {}, {}
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found.")
        return {}, {}
    except Exception as e:
        print(f"Error loading book mappings: {e}")
        return {}, {}

def extract_outcomes_and_events(data):
    """Extract betting outcomes and build event lookup in one pass."""
    outcomes = []
    event_lookup = {}

    def recursive_extract(obj, parent_keys=None):
        parent_keys = parent_keys or []
        if isinstance(obj, dict):
            # Extract outcomes
            if "outcome_id" in obj or ("odds" in obj and "book_id" in obj):
                outcomes.append(obj)

            # Extract event data
            if all(k in obj for k in ["id", "home_team_id", "away_team_id"]) and "teams" in obj:
                home = next((t for t in obj["teams"] if t["id"] == obj["home_team_id"]), None)
                away = next((t for t in obj["teams"] if t["id"] == obj["away_team_id"]), None)
                if home and away:
                    event_lookup[obj["id"]] = {
                        "home_team": home.get("full_name", home.get("name", "")),
                        "away_team": away.get("full_name", away.get("name", "")),
                        "home_logo": home.get("logo", ""),
                        "away_logo": away.get("logo", "")
                    }

            for key, value in obj.items():
                if key in ["odds", "lines", "markets", "betting", "books", "sportsbooks", "outcomes"]:
                    recursive_extract(value, parent_keys + [key])
                elif isinstance(value, (dict, list)):
                    recursive_extract(value, parent_keys + [key])
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item, parent_keys)

    recursive_extract(data)
    return outcomes, event_lookup

def save_to_csv(outcomes, event_lookup, book_id_mapping, book_weight_mapping, filename):
    """Save outcomes to CSV with team data, book names, and weights, filtering by market_type."""
    fields = [
        "event_id", "home_team", "away_team", "home_logo", "away_logo", "book_id", "book_name",
        "market_type", "outcome_id", "event_type", "type", "side", "period", "team_id", "odds",
        "value", "is_live", "line_status", "tickets_value", "tickets_percent", "money_value",
        "money_percent", "weight"
    ]

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for outcome in outcomes:
            market_type = outcome.get("type", outcome.get("market_type", "")).lower()
            if market_type not in ["moneyline", "total", "spread"]:
                continue

            event_id = outcome.get("event_id", "")
            team_data = event_lookup.get(event_id, {"home_team": "", "away_team": "", "home_logo": "", "away_logo": ""})
            bet_info = outcome.get("bet_info", {})
            tickets = bet_info.get("tickets", {}) if isinstance(bet_info, dict) else outcome.get("tickets", {})
            money = bet_info.get("money", {}) if isinstance(bet_info, dict) else outcome.get("money", {})
            book_id = str(outcome.get("book_id", ""))

            writer.writerow({
                "event_id": event_id,
                "home_team": team_data["home_team"],
                "away_team": team_data["away_team"],
                "home_logo": team_data["home_logo"],
                "away_logo": team_data["away_logo"],
                "book_id": book_id,
                "book_name": book_id_mapping.get(book_id, f"Unknown_{book_id}"),
                "market_type": outcome.get("type", outcome.get("market_type", "")),
                "outcome_id": outcome.get("outcome_id", ""),
                "event_type": outcome.get("event_type", ""),
                "type": outcome.get("type", ""),
                "side": outcome.get("side", ""),
                "period": outcome.get("period", ""),
                "team_id": outcome.get("team_id", ""),
                "odds": outcome.get("odds", ""),
                "value": outcome.get("value", ""),
                "is_live": outcome.get("is_live", ""),
                "line_status": outcome.get("line_status", ""),
                "tickets_value": tickets.get("value", 0) if isinstance(tickets, dict) else 0,
                "tickets_percent": tickets.get("percent", 0) if isinstance(tickets, dict) else 0,
                "money_value": money.get("value", 0) if isinstance(money, dict) else 0,
                "money_percent": money.get("percent", 0) if isinstance(money, dict) else 0,
                "weight": book_weight_mapping.get(book_id, "N/A")
            })
    return filepath

def filter_by_book_ids(df, allowed_book_ids):
    """Filter dataframe to only include rows with specified book IDs."""
    # Convert book_ids to strings for consistent comparison
    allowed_book_ids_str = [str(book_id) for book_id in allowed_book_ids]
    
    # Filter the dataframe
    filtered_df = df[df['book_id'].astype(str).isin(allowed_book_ids_str)]
    
    print(f"Original records: {len(df)}")
    print(f"Filtered records (book IDs {allowed_book_ids}): {len(filtered_df)}")
    
    return filtered_df

def format_date(date_input=None):
    """Convert date to YellowstoneMMDD format."""
    if date_input is None:
        return datetime.now().strftime("%Y%m%d")
    if isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%d")

    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]:
        try:
            return datetime.strptime(date_input, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_input}")

# --- Data Fetching ---
def fetch_mlb_data_internal(target_date=None):
    """Fetch MLB betting data and save to CSV."""
    formatted_date = format_date(target_date)
    print(f"Processing data for date: {formatted_date}")

    book_id_mapping, book_weight_mapping = load_book_mapping(BOOK_WEIGHTS_PATH)
    book_ids = ",".join(book_id_mapping.keys()) if book_id_mapping else ""

    if not book_ids:
        print("Warning: No book mappings loaded, using empty book_ids")
    print(f"Fetching data for book IDs: {book_ids}")

    url = f"https://api.actionnetwork.com/web/v2/scoreboard/publicbetting/mlb?bookIds={book_ids}&date={formatted_date}&periods=event"
    print("Fetching MLB data...")
    data = fetch_json(url)

    outcomes, event_lookup = extract_outcomes_and_events(data)
    print(f"Found {len(outcomes)} raw outcomes, {len(event_lookup)} events")

    filename = f"mlb_betting_data_weighted_{formatted_date}.csv"
    filepath = save_to_csv(outcomes, event_lookup, book_id_mapping, book_weight_mapping, filename)
    print(f"Data saved to {filepath}")
    return filepath, formatted_date

# --- Odds Processing ---
def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds is None:
        return None
    if american_odds > 0:
        return (american_odds / 100) + 1
    elif american_odds < 0:
        return (100 / abs(american_odds)) + 1
    else:
        return 1.0 # Even money, usually +100 or -100

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds."""
    if decimal_odds is None:
        return "N/A"
    if decimal_odds >= 2.0:
        return f"+{(decimal_odds - 1) * 100:.0f}"
    elif decimal_odds > 1.0:
        return f"{-100 / (decimal_odds - 1):.0f}"
    else:
        return "+100" # Should not happen for valid odds

def parse_file(file_path):
    """Parse file as CSV with multiple separator attempts."""
    required_cols = ['event_id', 'home_team', 'away_team', 'type', 'side', 'odds', 'weight', 'value']

    # Try different separators
    for sep in [',', '\t']:
        try:
            df = pd.read_csv(file_path, sep=sep)
            if all(col in df.columns for col in required_cols):
                print(f"Successfully loaded {'CSV' if sep == ',' else 'tab-separated'} from {file_path}")
                return df
        except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
            continue

    print(f"Could not parse {file_path} with standard CSV methods. Custom parsing for concatenated records is not robustly supported in this web app.")
    return pd.DataFrame()

def calculate_weighted_averages(df):
    """Calculate weighted averages for odds and include the most common 'value'."""
    print("Starting weighted average calculation...")

    # Clean and validate data
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(0)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').clip(0, 10).fillna(0)

    if 'value' in df.columns:
        if pd.api.types.is_numeric_dtype(df['value']):
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
    else:
        df['value'] = ''

    # Filter out invalid data
    df = df[(df['odds'] != 0) & (df['weight'] >= 0) & (df['weight'] <= 10)]
    print(f"After filtering: {len(df)} valid records")

    grouped = df.groupby(['event_id', 'home_team', 'away_team', 'type', 'side'])
    results = []

    for name, group in grouped:
        event_id, home_team, away_team, bet_type, side = name
        decimal_odds = [american_to_decimal(odds) for odds in group['odds']]
        weights = group['weight'].tolist()

        if sum(weights) > 0:
            weighted_avg_decimal = sum(d * w for d, w in zip(decimal_odds, weights)) / sum(weights)
            weighted_avg_american = decimal_to_american(weighted_avg_decimal)
        else:
            weighted_avg_american = "+100"

        # Get most common value
        most_common_value = None
        if 'value' in group.columns and not group['value'].empty:
            mode_values = group['value'].mode()
            if len(mode_values) > 0:
                most_common_value = mode_values.iloc[0]

        results.append({
            'event_id': event_id,
            'home_team': home_team,
            'away_team': away_team,
            'type': bet_type,
            'side': side,
            'weighted_average_odds': weighted_avg_american,
            'most_common_value': most_common_value,
            'total_records': len(group),
            'total_weight': sum(weights)
        })

    return pd.DataFrame(results)

def process_odds_file_internal(input_path, output_path, filtered_output_path=None, allowed_book_ids=None):
    """Process odds file and save weighted averages, with optional book ID filtering."""
    print(f"Processing {input_path}")
    df = parse_file(input_path)

    if df.empty:
        print("Error: No data parsed or loaded.")
        return None, None

    print(f"Parsed {len(df)} records")
    
    # Filter by book IDs if specified
    if allowed_book_ids is not None and 'book_id' in df.columns:
        filtered_df = filter_by_book_ids(df, allowed_book_ids)
        
        # Save filtered raw data if path provided
        if filtered_output_path:
            filtered_filepath = os.path.join(OUTPUT_DIR, filtered_output_path)
            filtered_df.to_csv(filtered_filepath, index=False)
            print(f"Filtered raw data saved to {filtered_filepath}")
        
        # Use filtered data for weighted averages
        df = filtered_df

    results_df = calculate_weighted_averages(df)

    if results_df.empty:
        print("No valid groups found for weighted average calculation.")
        return None, None

    print(f"Generated {len(results_df)} weighted average groups.")
    output_filepath = os.path.join(OUTPUT_DIR, output_path)
    results_df.to_csv(output_filepath, index=False)
    print(f"Results saved to {output_filepath}")

    positive_odds = sum(1 for x in results_df['weighted_average_odds'] if isinstance(x, str) and x.startswith('+'))
    print(f"Summary: {len(results_df)} groups, {positive_odds} positive odds, {len(results_df) - positive_odds} negative odds")
    return output_filepath, filtered_filepath if filtered_output_path else None


def american_to_implied_probability(odds):
    """Converts American odds to implied win probability."""
    if odds is None: return 0.0
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 0.5

def calculate_expected_value(fair_odds_american, sportsbook_odds_american):
    """
    Calculates Expected Value (EV) given fair odds and sportsbook odds.
    EV = (Probability_of_Win * Payout_on_Win) - (Probability_of_Loss * Stake)
    """
    if fair_odds_american is None or sportsbook_odds_american is None:
        return None

    try:
        # Convert fair_odds_american to numeric if it's a string from the CSV
        if isinstance(fair_odds_american, str):
            fair_odds_american = float(fair_odds_american)
        if isinstance(sportsbook_odds_american, str):
            sportsbook_odds_american = float(sportsbook_odds_american)

        fair_prob = american_to_implied_probability(fair_odds_american)
        sportsbook_payout_decimal = american_to_decimal(sportsbook_odds_american)
        
        if sportsbook_payout_decimal is None: # Handle cases where sportsbook_odds_american conversion fails
            return None

        expected_value = (fair_prob * (sportsbook_payout_decimal - 1)) - (1 - fair_prob)
        return expected_value
    except Exception as e:
        print(f"Error calculating EV: {e}")
        return None

def load_and_clean_data(book_odds_filepath):
    """Load CSV file and clean team names."""
    try:
        df_book_odds = pd.read_csv(book_odds_filepath)

        # Clean team names by replacing underscores with spaces
        for col in ['home_team', 'away_team']:
            if col in df_book_odds.columns:
                df_book_odds[col] = df_book_odds[col].astype(str).str.replace('_', ' ')

        return df_book_odds
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def format_odds(odds):
    """Format odds with + or - sign for readability."""
    if pd.isna(odds):
        return "N/A"
    return f"{odds:+.0f}" if odds > 0 else f"{odds:.0f}"

def get_best_odds_row(odds_df):
    """
    Find the row with the best odds (most favorable to bettor).
    For American odds:
    - Negative odds: -100 is better than -120 (closer to 0)
    - Positive odds: +120 is better than +100 (higher number)
    - Overall: +100 is better than -100
    """
    if odds_df.empty:
        return None
    
    # Convert odds to a comparable value where higher = better for bettor
    def odds_to_comparable_value(odds):
        if pd.isna(odds):
            return float('-inf')  # NaN gets lowest priority
        if odds > 0:
            return odds  # Positive odds: higher is better
        elif odds < 0:
            return 10000 + odds  # Negative odds: closer to 0 is better, so -100 becomes 9900, -120 becomes 9880
        else:
            return 0  # Even odds, e.g., 0 for some data points, treat as neutral

    odds_df = odds_df.copy()
    # Ensure 'odds' column is numeric before applying comparison logic
    odds_df['odds'] = pd.to_numeric(odds_df['odds'], errors='coerce')
    odds_df['comparable_value'] = odds_df['odds'].apply(odds_to_comparable_value)
    
    # Get the row with the highest comparable value (best odds)
    best_idx = odds_df['comparable_value'].idxmax()
    return odds_df.loc[best_idx]

def get_favorability(open_odds, consensus_odds):
    """Determine if odds movement is more or less favorable."""
    if pd.isna(open_odds) or pd.isna(consensus_odds):
        return "N/A"

    # Ensure odds are numeric for comparison
    open_odds = float(open_odds)
    consensus_odds = float(consensus_odds)

    if consensus_odds == open_odds:
        return "NO change"

    # Convert to decimal for consistent comparison of favorability
    open_dec = american_to_decimal(open_odds)
    consensus_dec = american_to_decimal(consensus_odds)

    if consensus_dec is None or open_dec is None:
        return "N/A" # Handle conversion failures

    # Higher decimal odds (larger payout for same stake) are more favorable
    if consensus_dec > open_dec:
        return "MORE favorable"
    else:
        return "LESS favorable"

def detect_reverse_line_movement(open_odds, consensus_odds, tickets_percent, side):
    """Detect reverse line movement based on odds change and betting percentages."""
    if pd.isna(open_odds) or pd.isna(consensus_odds) or pd.isna(tickets_percent):
        return False, None, 0.0

    # Ensure odds are numeric
    open_odds = float(open_odds)
    consensus_odds = float(consensus_odds)
    tickets_percent = float(tickets_percent)

    # If the fade side odds are becoming more favorable (e.g., -150 to -140, or +120 to +130)
    # AND the tickets on the public side are high (e.g., > 50%).
    
    # Calculate favorability for the fade side
    fade_favorability = get_favorability(open_odds, consensus_odds)

    if fade_favorability == "MORE favorable" and tickets_percent > 50: # tickets_percent is for the PUBLIC side
        return True, 'fade_side_became_more_favorable_despite_public_betting', tickets_percent
    
    return False, None, 0.0


def get_market_sides_and_labels(market_type, home_team, away_team):
    """Get sides and team labels for different market types."""
    if market_type in ['spread', 'moneyline']:
        return ['home', 'away'], {'home': home_team, 'away': away_team}
    elif market_type == 'totals':
        return ['over', 'under'], {'over': 'Over', 'under': 'Under'}
    else:
        return [], {}

def extract_side_data(event_data, market_type, sides):
    """Extract open, consensus, sportsbook odds data for each side."""
    # Handle totals market type mapping
    filter_market = 'total' if market_type == 'totals' else market_type
    market_data = event_data[event_data['market_type'].astype(str).str.lower() == filter_market]

    if market_data.empty:
        return None

    # Ensure 'book_name' is string type to prevent error with .str.contains
    market_data['book_name'] = market_data['book_name'].astype(str)

    open_data = market_data[market_data['book_name'].str.contains('Open', na=False)]
    consensus_data = market_data[market_data['book_name'].str.contains('Consensus', na=False)]
    sportsbook_data = market_data[~market_data['book_name'].isin(['Open', 'Consensus'])]

    if open_data.empty or consensus_data.empty:
        print(f"Missing Open or Consensus data for {market_type}")
        return None

    side_data = {}
    for side in sides:
        # Using .lower() for case-insensitive matching
        side_filter = lambda df: df[df['side'].astype(str).str.lower().str.contains(side.lower(), na=False)]

        side_data[side] = {
            'open': side_filter(open_data),
            'consensus': side_filter(consensus_data),
            'sportsbook': side_filter(sportsbook_data),
            'fair_odds_value': None
        }

        # Get fair odds value from consensus (since you mentioned fair odds are in consensus)
        consensus_side_data = side_filter(consensus_data)
        if not consensus_side_data.empty and not pd.isna(consensus_side_data['odds'].iloc[0]):
            side_data[side]['fair_odds_value'] = consensus_side_data['odds'].iloc[0]

        # Check if we have required data
        if (side_data[side]['open'].empty or side_data[side]['consensus'].empty or
            side_data[side]['open']['odds'].isnull().any() or
            side_data[side]['consensus']['odds'].isnull().any()):
            print(f"Incomplete data for side {side} in {market_type}. Skipping.")
            return None

    return side_data

def determine_public_and_fade_sides(side_data, sides):
    """Determine which side is public (more tickets) and which to fade."""
    side1, side2 = sides
    side1_tickets = side_data[side1]['consensus']['tickets_percent'].iloc[0] if not side_data[side1]['consensus'].empty else 0
    side2_tickets = side_data[side2]['consensus']['tickets_percent'].iloc[0] if not side_data[side2]['consensus'].empty else 0

    if side1_tickets > side2_tickets:
        return side1, side2
    else:
        return side2, side1

def create_side_labels(market_type, side, side_teams, side_data, total_value=None):
    """Create appropriate labels for different market types and sides."""
    if market_type == 'spread':
        value_raw = side_data[side]['consensus']['value'].iloc[0]
        value = value_raw if not pd.isna(value_raw) else 0.0 # Default if NaN
        return f"{side_teams[side]} {value:+.1f}"
    elif market_type == 'totals':
        # For totals, the 'value' typically represents the over/under line
        # This value should be consistent across both 'over' and 'under' for the same market.
        # We can take it from any side's consensus data.
        if total_value is None: # Fallback in case total_value wasn't passed from process_market
            total_value = side_data[side]['consensus']['value'].iloc[0]
        
        total_value = total_value if not pd.isna(total_value) else 0.0 # Default if NaN
        return f"{side_teams[side]} {total_value:.1f}"
    elif market_type == 'moneyline':
        return f"{side_teams[side]} to win"
    else:
        return f"{side_teams[side]}"

def add_best_line_with_ev(report_lines, row, side_key, side_teams, fair_odds_val, market_type):
    """Add best sportsbook line with EV calculation to report."""
    if row is None:
        return

    book = row['book_name']
    odds = pd.to_numeric(row['odds'], errors='coerce') # Ensure odds is numeric
    ev_str = ""

    if fair_odds_val is not None and not pd.isna(odds):
        ev = calculate_expected_value(fair_odds_val, odds)
        if ev is not None:
            ev_str = f" (EV: {ev:.2%})"

    if market_type == 'spread':
        value = pd.to_numeric(row['value'], errors='coerce')
        value = value if not pd.isna(value) else 0.0
        label = f"{side_teams[side_key]} {value:+.1f}"
    elif market_type == 'totals':
        value = pd.to_numeric(row['value'], errors='coerce')
        value = value if not pd.isna(value) else 0.0
        label = f"{side_teams[side_key]} {value:.1f}"
    else:  # moneyline
        label = f"{side_teams[side_key]} to win"

    report_lines.append(f"    {book}: {label} ({format_odds(odds)}){ev_str}")

def process_market(event_data, market_type, home_team, away_team):
    """Process data for a specific market type and return report lines."""
    sides, side_teams = get_market_sides_and_labels(market_type, home_team, away_team)
    if not sides:
        return []

    side_data = extract_side_data(event_data, market_type, sides)
    if side_data is None:
        return []

    public_side, fade_side = determine_public_and_fade_sides(side_data, sides)

    # Get betting percentages
    public_tickets = side_data[public_side]['consensus']['tickets_percent'].iloc[0]
    public_money = side_data[public_side]['consensus']['money_percent'].iloc[0]
    fade_tickets = side_data[fade_side]['consensus']['tickets_percent'].iloc[0]
    fade_money = side_data[fade_side]['consensus']['money_percent'].iloc[0]

    # Get total value for totals market (should be the line value, e.g., 8.5 for over/under)
    total_value = None
    if market_type == 'totals':
        # The 'value' column for totals will hold the O/U line, which should be consistent for both sides
        if not side_data[sides[0]]['consensus'].empty:
             total_value = side_data[sides[0]]['consensus']['value'].iloc[0]


    # Create labels
    public_label = create_side_labels(market_type, public_side, side_teams, side_data, total_value)
    fade_label = create_side_labels(market_type, fade_side, side_teams, side_data, total_value)

    # Get odds and calculate movements
    public_open_odds = side_data[public_side]['open']['odds'].iloc[0]
    public_consensus_odds = side_data[public_side]['consensus']['odds'].iloc[0]
    fade_open_odds = side_data[fade_side]['open']['odds'].iloc[0]
    fade_consensus_odds = side_data[fade_side]['consensus']['odds'].iloc[0]

    # Ensure odds are numeric before calculations
    public_open_odds = pd.to_numeric(public_open_odds, errors='coerce')
    public_consensus_odds = pd.to_numeric(public_consensus_odds, errors='coerce')
    fade_open_odds = pd.to_numeric(fade_open_odds, errors='coerce')
    fade_consensus_odds = pd.to_numeric(fade_consensus_odds, errors='coerce')


    public_move = public_consensus_odds - public_open_odds if not pd.isna(public_consensus_odds) and not pd.isna(public_open_odds) else None
    fade_move = fade_consensus_odds - fade_open_odds if not pd.isna(fade_consensus_odds) and not pd.isna(fade_open_odds) else None


    public_favor = get_favorability(public_open_odds, public_consensus_odds)
    fade_favor = get_favorability(fade_open_odds, fade_consensus_odds)

    # Check for reverse line movement
    rlm_detected, rlm_type, strength = detect_reverse_line_movement(
        fade_open_odds, fade_consensus_odds, public_tickets, 'fade'
    )

    # Get best sportsbook lines (most favorable to bettor)
    best_public_row = None
    best_fade_row = None

    if not side_data[public_side]['sportsbook'].empty:
        best_public_row = get_best_odds_row(side_data[public_side]['sportsbook'])

    if not side_data[fade_side]['sportsbook'].empty:
        best_fade_row = get_best_odds_row(side_data[fade_side]['sportsbook'])

    # Build report
    report_lines = []
    report_lines.append(f"PUBLIC SIDE: {public_label} ({public_tickets:.0f}% tickets, {public_money:.0f}% money)")
    report_lines.append(f"FADE SIDE: {fade_label} ({fade_tickets:.0f}% tickets, {fade_money:.0f}% money)")
    report_lines.append("Line Movements:")
    report_lines.append(f"    {public_label}: {format_odds(public_move) if public_move is not None else 'N/A'} ({public_favor})")
    report_lines.append(f"    {fade_label}: {format_odds(fade_move) if fade_move is not None else 'N/A'} ({fade_favor})")

    if rlm_detected:
        report_lines.append("🚨 REVERSE LINE MOVEMENT DETECTED!")
        report_lines.append(f"    Type: {rlm_type}")
        report_lines.append(f"    Strength: {strength:.1f}%")

    report_lines.append("Detailed Analysis:")
    for side_key in [fade_side, public_side]:
        current_open_odds = side_data[side_key]['open']['odds'].iloc[0]
        current_consensus_odds = side_data[side_key]['consensus']['odds'].iloc[0]
        
        current_open_odds = pd.to_numeric(current_open_odds, errors='coerce')
        current_consensus_odds = pd.to_numeric(current_consensus_odds, errors='coerce')

        current_move = current_consensus_odds - current_open_odds if not pd.isna(current_consensus_odds) and not pd.isna(current_open_odds) else None
        current_label = create_side_labels(market_type, side_key, side_teams, side_data, total_value)

        report_lines.append(f"    {current_label}: {format_odds(current_open_odds)} → {format_odds(current_consensus_odds)} (Move: {format_odds(current_move) if current_move is not None else 'N/A'})")

    report_lines.append("Best Sportsbook Lines:")
    add_best_line_with_ev(report_lines, best_fade_row, fade_side, side_teams,
                            side_data[fade_side]['fair_odds_value'], market_type)
    add_best_line_with_ev(report_lines, best_public_row, public_side, side_teams,
                            side_data[public_side]['fair_odds_value'], market_type)

    return report_lines

def ensure_numeric_columns(df):
    """Ensure relevant columns are numeric for calculations."""
    numeric_columns = ['odds', 'value', 'tickets_percent', 'money_percent', 'event_id']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def generate_formatted_report_internal(df, output_path):
    """Generate a formatted line movement report including spread, moneyline, and totals."""
    df = ensure_numeric_columns(df)

    report_lines = []
    # Filter out rows where event_id could not be converted to numeric (NaN)
    df_filtered = df[pd.to_numeric(df['event_id'], errors='coerce').notna()]
    
    # Sort by event_id for consistent report order
    df_filtered = df_filtered.sort_values(by='event_id')

    for event_id, event_data in df_filtered.groupby('event_id'):
        home_team = event_data['home_team'].iloc[0]
        away_team = event_data['away_team'].iloc[0]

        report_lines.append(f"*** {home_team} vs {away_team} (Event ID: {int(event_id)}) ***")
        report_lines.append("-" * 60)

        for market_type in ['spread', 'moneyline', 'totals']:
            report_lines.append(f"\n--- {market_type.capitalize()} Market ---")
            market_report = process_market(event_data, market_type, home_team, away_team)
            if market_report:
                report_lines.extend(market_report)
            else:
                report_lines.append("No sufficient data available for this market type.")
            report_lines.append("")

        report_lines.append("=" * 60)
        report_lines.append("\n")

    filepath = os.path.join(OUTPUT_DIR, output_path)
    with open(filepath, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {filepath}")
    return filepath

# --- Flask Routes ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_file('static/index.html')

@app.route('/fetch-and-process', methods=['GET'])
def fetch_and_process_data():
    """
    Endpoint to trigger fetching and processing of MLB betting data.
    Returns paths to generated files.
    """
    try:
        # Define the allowed book IDs
        ALLOWED_BOOK_IDS = [15, 13, 30, 281, 286, 1138, 1798, 1921, 2396, 3848, 4256]
        
        eastern_now = datetime.now(ZoneInfo("America/New_York"))
        date_str = eastern_now.strftime("%Y-%m-%d")

        # Step 1: Fetch raw data
        input_file_path, formatted_date = fetch_mlb_data_internal(date_str)
        
        # Step 2: Process with book ID filtering and calculate weighted averages
        filtered_raw_file_name = f"filtered_raw_data_{formatted_date}.csv"
        weighted_averages_file_name = f"weighted_averages_results_{formatted_date}.csv" # Add date to ensure uniqueness
        
        weighted_avg_filepath, filtered_raw_filepath = process_odds_file_internal(
            input_file_path, 
            weighted_averages_file_name,
            filtered_output_path=filtered_raw_file_name,
            allowed_book_ids=ALLOWED_BOOK_IDS
        )

        # Step 3: Generate the formatted report
        report_file_name = f"line_movement_report_mlb_{formatted_date}.txt"
        df_book_odds = load_and_clean_data(filtered_raw_filepath) # Load filtered data for report generation
        
        if df_book_odds is None:
            return jsonify({"status": "error", "message": "Failed to load data for report generation."}), 500

        report_filepath = generate_formatted_report_internal(df_book_odds, report_file_name)

        return jsonify({
            "status": "success",
            "message": "Data fetched, processed, and reports generated.",
            "original_data_path": os.path.basename(input_file_path),
            "filtered_raw_data_path": os.path.basename(filtered_raw_filepath) if filtered_raw_filepath else None,
            "weighted_averages_path": os.path.basename(weighted_avg_filepath) if weighted_avg_filepath else None,
            "report_path": os.path.basename(report_filepath)
        })

    except requests.exceptions.RequestException as req_err:
        print(f"Network or API error: {req_err}")
        return jsonify({"status": "error", "message": f"Failed to fetch data from API: {req_err}"}), 500
    except FileNotFoundError as fnf_err:
        print(f"File not found error: {fnf_err}")
        return jsonify({"status": "error", "message": f"Required file not found: {fnf_err}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500

@app.route('/reports/<filename>')
def serve_report(filename):
    """Endpoint to serve generated report files."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        # Determine mimetype based on file extension
        if filename.endswith('.txt'):
            mimetype = 'text/plain'
        elif filename.endswith('.csv'):
            mimetype = 'text/csv'
        else:
            mimetype = 'application/octet-stream' # Generic binary
        return send_file(filepath, mimetype=mimetype)
    else:
        return jsonify({"status": "error", "message": "File not found."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
