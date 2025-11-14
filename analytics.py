import pandas as pd
import matplotlib.pyplot as plt

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

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
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

if __name__ == "__main__":
    # Load the keylog data
    data_file = 'keylog.csv'
    df = load_data(data_file)

    # Analyze and plot key press distribution
    analyze_key_press_distribution(df)

    # Analyze and plot key press durations
    analyze_key_press_durations(df)