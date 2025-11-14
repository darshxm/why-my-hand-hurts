from pynput import keyboard
import csv
from datetime import datetime
import os

# Dictionary to store press start times for each key
press_times = {}

# Initialize CSV with headers if file doesn't exist
csv_file = 'keylog.csv'
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'key', 'duration'])

def on_press(key):
    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)
    
    # Record the start time of the press
    press_times[key_name] = datetime.now()

def on_release(key):
    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)
    
    # Calculate duration if we have a start time
    if key_name in press_times:
        start_time = press_times[key_name]
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save to CSV with timestamp and duration
        with open('keylog.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([start_time.strftime('%Y-%m-%d %H:%M:%S.%f'), key_name, duration])
        
        # Remove the key from dictionary
        del press_times[key_name]
    
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()