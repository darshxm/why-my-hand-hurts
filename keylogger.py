from pynput import keyboard
import csv
from datetime import datetime
import os
import getpass
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# --- Encryption Setup ---
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

# Get password from user
password = getpass.getpass("Enter a password to encrypt the keylog: ")

# Generate a new salt or load an existing one
salt_file = 'key.salt'
if os.path.exists(salt_file):
    with open(salt_file, 'rb') as f:
        salt = f.read()
else:
    salt = os.urandom(16)
    with open(salt_file, 'wb') as f:
        f.write(salt)

# Create a Fernet instance for encryption
key = get_encryption_key(password, salt)
fernet = Fernet(key)
# --- End of Encryption Setup ---

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

# Collect events until released
with open(csv_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
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
            
            # Encrypt the key and duration
            encrypted_key = fernet.encrypt(key_name.encode()).decode()
            encrypted_duration = fernet.encrypt(str(duration).encode()).decode()
            
            # Save to CSV with timestamp and encrypted data
            writer.writerow([start_time.strftime('%Y-%m-%d %H:%M:%S.%f'), encrypted_key, encrypted_duration])
            
            # Remove the key from dictionary
            del press_times[key_name]
        
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()