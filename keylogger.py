from pynput import keyboard
import csv
from datetime import datetime
import os
import platform
import subprocess
import getpass
import base64
from typing import Optional, Tuple
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pywinctl  # cross-platform active window introspection
    HAS_PYWINCTL = True
except ImportError:
    HAS_PYWINCTL = False

if platform.system() == "Windows":
    try:
        import win32gui
        import win32process
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
else:
    HAS_WIN32 = False

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

def get_active_app_window() -> Tuple[Optional[str], Optional[str]]:
    """
    Return (app_name, window_title) for the active window.
    Tries pywinctl first; falls back to platform tricks where possible.
    """
    # Best-effort: pywinctl handles Windows/macOS/Linux if installed
    if HAS_PYWINCTL:
        try:
            win = pywinctl.getActiveWindow()
            if win:
                title = win.title or None
                app = None
                try:
                    pid = win.getPid()
                    if HAS_PSUTIL:
                        app = psutil.Process(pid).name()
                except Exception:
                    pass
                return app, title
        except Exception:
            pass

    system = platform.system()
    # Windows fallback using pywin32 if available
    if system == "Windows" and HAS_WIN32:
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or None
            pid = win32process.GetWindowThreadProcessId(hwnd)[1]
            app = None
            if HAS_PSUTIL:
                try:
                    app = psutil.Process(pid).name()
                except Exception:
                    app = None
            else:
                app = str(pid)
            return app, title
        except Exception:
            return None, None

    # macOS fallback via AppleScript (no extra deps)
    if system == "Darwin":
        try:
            app = subprocess.check_output(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                text=True,
                timeout=1,
            ).strip() or None
            title = subprocess.check_output(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to tell (process 1 where frontmost is true) to if exists (window 1) then get name of window 1',
                ],
                text=True,
                timeout=1,
            ).strip() or None
            return app, title
        except Exception:
            return None, None

    # Linux/unknown fallback: nothing better without pywinctl
    return None, None


# Initialize CSV with headers if file doesn't exist
csv_file = 'keylog.csv'
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'key', 'duration', 'app', 'window_title'])

def on_press(key):
    try:
        key_name = key.char
    except AttributeError:
        key_name = str(key)
    
    # Record the start time of the press
    press_times[key_name] = datetime.now()

def ensure_header(writer):
    """Write header if file was empty and a new handle was opened."""
    if os.path.getsize(csv_file) == 0:
        writer.writerow(['timestamp', 'key', 'duration', 'app', 'window_title'])


# Collect events until released
with open(csv_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    ensure_header(writer)
    
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
            
            # Active app/window metadata (optional)
            app_name, window_title = get_active_app_window()
            
            # Encrypt the key and duration and metadata
            encrypted_key = fernet.encrypt(key_name.encode()).decode()
            encrypted_duration = fernet.encrypt(str(duration).encode()).decode()
            encrypted_app = fernet.encrypt((app_name or "").encode()).decode()
            encrypted_window = fernet.encrypt((window_title or "").encode()).decode()
            
            # Save to CSV with timestamp and encrypted data
            writer.writerow([
                start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                encrypted_key,
                encrypted_duration,
                encrypted_app,
                encrypted_window,
            ])
            
            # Remove the key from dictionary
            del press_times[key_name]
        
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
