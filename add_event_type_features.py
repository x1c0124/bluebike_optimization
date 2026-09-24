import pandas as pd
import re
from datetime import datetime
import os

def parse_event_date(date_str):
    """Parse date string like 'Jan 8th', 'Feb 1st' to datetime"""
    # Remove the header row if it exists
    if date_str == 'event_type' or 'event_type' in str(date_str):
        return None

    # Clean the date string
    date_str = str(date_str).strip()

    # Map month abbreviations to numbers
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # Extract month and day
    match = re.match(r'([A-Za-z]+)\s+(\d+)(st|nd|rd|th)', date_str)
    if match:
        month_str = match.group(1)
        day = int(match.group(2))
        month = month_map.get(month_str)
        if month:
            # Assume year 2024
            try:
                return datetime(2024, month, day)
            except ValueError:
                return None
    return None

def load_events(events_file):
    """Load events and create a mapping of dates to event types"""
    df = pd.read_csv(events_file, header=None, names=['date', 'event_name', 'event_type'])

    # Remove header row if exists
    df = df[df['event_type'] != 'event_type']

    # Parse dates
    event_dates = {}
    for _, row in df.iterrows():
        date_obj = parse_event_date(row['date'])
        if date_obj:
            date_key = date_obj.date()  # Use date only (no time)
            event_type = str(row['event_type']).strip()
            # If multiple events on same date, keep all types
            if date_key not in event_dates:
                event_dates[date_key] = []
            if event_type not in event_dates[date_key]:
                event_dates[date_key].append(event_type)

    # Get all unique event types
    all_event_types = set()
    for types in event_dates.values():
        all_event_types.update(types)

    return event_dates, sorted(list(all_event_types))

def add_event_features_to_monthly_file(file_path, event_dates, event_types):
    """Add event type features to a monthly CSV file"""
    print(f"Processing {file_path}...")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert started_at to datetime (handle different formats)
    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', errors='coerce')
    df['date'] = df['started_at'].dt.date

    # Initialize all event type columns with 0
    for event_type in event_types:
        df[f'event_{event_type}'] = 0

    # Set to 1 if date matches an event date
    for idx, row in df.iterrows():
        date = row['date']
        if date in event_dates:
            for event_type in event_dates[date]:
                df.at[idx, f'event_{event_type}'] = 1

    # Drop the temporary date column
    df = df.drop(columns=['date'])

    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"  Added {len(event_types)} event type features to {file_path}")

def main():
    # Paths
    events_file = 'tdgarden_events_with_type.csv'
    data_dir = 'modified data'

    # Load events
    print("Loading events from tdgarden_events_with_type.csv...")
    event_dates, event_types = load_events(events_file)

    print(f"Found {len(event_dates)} event dates")
    print(f"Event types: {event_types}")

    # Process monthly files
    monthly_files = [f'clean_data_{i}.csv' for i in range(1, 13)]
    monthly_files.append('merged_all_months.csv')

    for filename in monthly_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            add_event_features_to_monthly_file(file_path, event_dates, event_types)
        else:
            print(f"  File not found: {file_path}")

    print("\nDone! All monthly files have been updated with event type features.")

if __name__ == '__main__':
    main()
