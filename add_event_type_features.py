"""
Add TD Garden event-type features (event_Basketball, event_Concert, ...) to the
cleaned monthly trip files. A trip gets event_<type> = 1 if it started on a day
with an event of that type.
"""

import os
import re
from collections import defaultdict
from datetime import date

import pandas as pd

EVENTS_FILE = 'tdgarden_events_with_type.csv'
DATA_DIR = 'modified data'
YEAR = 2024

MONTHS = {m: i for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}


def parse_event_date(date_str):
    """Parse strings like 'Jan 8th' or 'Feb 1st' as a date in YEAR; None if invalid."""
    match = re.match(r'([A-Za-z]+)\s+(\d+)(st|nd|rd|th)', str(date_str).strip())
    if not match or match.group(1) not in MONTHS:
        return None
    try:
        return date(YEAR, MONTHS[match.group(1)], int(match.group(2)))
    except ValueError:
        return None


def load_events(events_file):
    """Return {date: set of event types} and the sorted list of all event types."""
    df = pd.read_csv(events_file, header=None, names=['date', 'event_name', 'event_type'])

    event_dates = defaultdict(set)
    for date_str, event_type in zip(df['date'], df['event_type']):
        event_date = parse_event_date(date_str)
        if event_date is None:
            print(f"  Skipping unparseable date: {date_str!r}")
            continue
        event_dates[event_date].add(str(event_type).strip())

    event_types = sorted(set().union(*event_dates.values()))
    return dict(event_dates), event_types


def add_event_features(file_path, event_dates, event_types):
    """Recompute the event columns in one trip file, leaving all other columns as they are."""
    # Read as text so the other columns are written back unchanged
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    trip_dates = pd.to_datetime(df['started_at'], format='mixed').dt.date

    for event_type in event_types:
        days = {d for d, types in event_dates.items() if event_type in types}
        df[f'event_{event_type}'] = trip_dates.isin(days).astype(int)

    df.to_csv(file_path, index=False)
    print(f"  {os.path.basename(file_path)}: {len(df)} trips updated")


def main():
    event_dates, event_types = load_events(EVENTS_FILE)
    print(f"Found {len(event_dates)} event dates")
    print(f"Event types: {event_types}")

    files = [f'clean_data_{i}.csv' for i in range(1, 13)] + ['merged_all_months.csv']
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            add_event_features(file_path, event_dates, event_types)
        else:
            print(f"  File not found: {file_path}")


if __name__ == '__main__':
    main()
