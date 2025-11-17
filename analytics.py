# analytics.py
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

# --- Decryption Setup ---
def get_encryption_key(password, salt):
    """Derive a key from the password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def clean_key_names(key):
    """Replace key names with readable symbols."""
    key_mapping = {
        'Key.space': '␣',
        'Key.enter': '↵',
        'Key.backspace': '⌫',
        'Key.tab': '⇥',
        'Key.shift': '⇧',
        'Key.shift_r': '⇧',
        'Key.ctrl': '*',
        'Key.ctrl_l': '*L',
        'Key.ctrl_r': '*R',
        'Key.alt': '⎇',
        'Key.alt_l': '⎇L',
        'Key.alt_r': '⎇R',
        'Key.cmd': '⌘',
        'Key.esc': 'Esc',
        'Key.caps_lock': 'Caps',
        'Key.delete': 'Del',
        'Key.home': 'Home',
        'Key.end': 'End',
        'Key.page_up': 'PgUp',
        'Key.page_down': 'PgDn',
        'Key.up': '↑',
        'Key.down': '↓',
        'Key.left': '←',
        'Key.right': '→',
    }
    return key_mapping.get(str(key), str(key))

def _decrypt_column(df, column, fernet):
    """Decrypt a column if present; otherwise leave untouched."""
    if column not in df.columns:
        return df
    df[column] = df[column].apply(lambda x: fernet.decrypt(str(x).encode()).decode() if pd.notna(x) else "")
    return df


def _read_keylog_csv(file_path: str) -> pd.DataFrame:
    """Read keylog CSV tolerating legacy 3-column and new 5-column rows."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if parts[0].lower() == "timestamp":
                continue
            if len(parts) < 3:
                continue
            while len(parts) < 5:
                parts.append("")
            parts = parts[:5]
            rows.append(parts)

    df = pd.DataFrame(
        rows, columns=["timestamp", "key", "duration", "app", "window_title"]
    )
    return df


def _apply_filters(df, app_filter=None, window_filter=None):
    """Filter rows by app or window substring (case-insensitive)."""
    if app_filter and 'app' in df.columns:
        df = df[df['app'].fillna("").str.lower().str.contains(app_filter.lower())]
    if window_filter and 'window_title' in df.columns:
        df = df[df['window_title'].fillna("").str.lower().str.contains(window_filter.lower())]
    return df


def load_data(file_path, fernet, app_filter=None, window_filter=None):
    """Load, decrypt, and optionally filter data from a CSV file into a pandas DataFrame."""
    df = _read_keylog_csv(file_path)
    
    # Decrypt columns
    try:
        df = _decrypt_column(df, 'key', fernet)
        df = _decrypt_column(df, 'duration', fernet)
        df = _decrypt_column(df, 'app', fernet)
        df = _decrypt_column(df, 'window_title', fernet)
    except InvalidToken:
        print("Decryption failed. Please check your password.")
        exit()

    df = _apply_filters(df, app_filter, window_filter)

    # Clean key names for better visualization
    df['key'] = df['key'].apply(clean_key_names)
    return df

def analyze_key_press_distribution(df):
    """Analyze key press distribution and plot the results."""
    # Convert duration to numeric (in case it's not)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Plot the key press distribution
    plt.figure(figsize=(12, 6))
    df['key'].value_counts().plot(kind='bar', color='blue')
    plt.xlabel('Keys')
    plt.ylabel('Frequency')
    plt.title('Key Press Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('key_press_distribution.png')

def analyze_key_press_durations(df):
    """Analyze key press durations and plot the results."""
    # Convert duration to numeric (in case it's not)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Plot the key press durations
    plt.figure(figsize=(12, 6))
    df.boxplot(column='duration', by='key', grid=False)
    plt.xlabel('Keys')
    plt.ylabel('Duration (seconds)')
    plt.title('Key Press Durations')
    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('key_press_durations.png')


def summarize_by_app(df):
    """Print a quick summary of keystrokes and duration by app/window."""
    if 'app' not in df.columns or 'window_title' not in df.columns:
        return

    df['app'] = df['app'].fillna("").replace("", "<unknown>")
    df['window_title'] = df['window_title'].fillna("").replace("", "<unknown>")

    app_counts = df.groupby('app').size().sort_values(ascending=False)
    print("\nTop apps by keystrokes:")
    for app, count in app_counts.head(10).items():
        print(f"  {app}: {count}")

    window_counts = df.groupby('window_title').size().sort_values(ascending=False)
    print("\nTop windows by keystrokes:")
    for title, count in window_counts.head(10).items():
        print(f"  {title}: {count}")

    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        app_durations = df.groupby('app')['duration'].mean().sort_values(ascending=False)
        print("\nAvg keypress duration by app (s):")
        for app, dur in app_durations.head(10).items():
            print(f"  {app}: {dur:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze keylog data.')
    parser.add_argument('keylog_file', help='Path to the keylog CSV file.')
    parser.add_argument('--app', dest='app_filter', help='Only include keystrokes from apps matching this substring (case-insensitive).')
    parser.add_argument('--window', dest='window_filter', help='Only include keystrokes from windows matching this substring (case-insensitive).')
    args = parser.parse_args()

    # Get password from user
    password = getpass.getpass("Enter the password to decrypt the keylog: ")

    # Load the salt
    salt_file = 'key.salt'
    if not os.path.exists(salt_file):
        print("Salt file not found. Make sure 'key.salt' is in the same directory.")
        exit()
    with open(salt_file, 'rb') as f:
        salt = f.read()

    # Create a Fernet instance for decryption
    key = get_encryption_key(password, salt)
    fernet = Fernet(key)

    # Load the keylog data
    df = load_data(args.keylog_file, fernet, app_filter=args.app_filter, window_filter=args.window_filter)

    # App/window-level summary
    summarize_by_app(df)

    # Analyze and plot key press distribution
    analyze_key_press_distribution(df)

    # Analyze and plot key press durations
    analyze_key_press_durations(df)
