# Why My Hand Hurts

This project is a set of tools to analyze your typing habits and generate an optimized keyboard layout to reduce strain. It consists of three main components:

1.  **Keylogger**: A simple keylogger that encrypts and records your keystrokes, including the key pressed, the timestamp, the duration of the press, and (when available) the active app and window title.
2.  **Analytics**: A script that analyzes the keylog data and generates plots to visualize your typing habits.
3.  **Keyboard Layout Optimizer**: A script that uses your typing data to generate a keyboard layout optimized for ergonomics and efficiency.

## Features

*   **Keystroke Logging**: Records key presses, timestamps, and durations to a CSV file.
*   **Typing Analysis**: Generates plots for key press distribution and durations.
*   **Keyboard Layout Optimization**:
    *   Generates an optimized keyboard layout based on your personal typing data.
    *   Uses a sophisticated scoring system that considers key frequency, finger effort, hand alternation, and same-finger bigrams.
    *   Visualizes the optimized layout with a frequency heatmap.
    *   Compares the optimized layout with the standard QWERTY layout.
    *   Provides detailed statistics about the optimized layout.
*   **Per-Finger Strain Scoring**: Estimates 0–100 strain scores for each finger using keystroke volume, positional effort, fatigue (keystroke duration changes), and natural rest time.

## Setup

1.  **Create a virtual environment and install the dependencies:**

    ```bash
    ./setup.sh
    ```
    **Note for Debian/Ubuntu users**: The `pynput` library may require the Python development headers to be installed. If the installation fails, you may need to install them with the following command:
    ```bash
    sudo apt-get install python3-dev
    ```

2.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

## Usage

### 1. Log Your Keystrokes

Run the keylogger to start recording your typing. The keylogger will run in the background and save the data to `keylog.csv`.

```bash
python keylogger.py
```

Press the `Esc` key to stop the keylogger.

**Note**: This keylogger is for personal use only. Be mindful of your privacy and do not run it on a machine that is not your own.

## Security and Privacy

This project includes features to protect your data, but it's important to understand the potential security and privacy risks involved.

### Data Encryption

*   **Encryption at Rest**: The `keylog.csv` file, which contains your raw keystroke data, is encrypted using the Fernet symmetric encryption scheme from the `cryptography` library.
*   **Password-Based Key**: The encryption key is derived from a password you provide when you start the keylogger. This password is not stored, so it's crucial that you remember it.
*   **Salted Key Derivation**: A random salt is used with the PBKDF2 key derivation function to protect against pre-computed key attacks. The salt is stored in the `key.salt` file.

### Risks and Mitigations

*   **Data in Memory**: When you run the `analytics.py` or `keyboard_layout.py` scripts, the encrypted data from `keylog.csv` is decrypted and loaded into memory. This means that for the duration of the script's execution, your raw keystroke data is present in your computer's RAM in plaintext. An attacker with access to your computer's memory could potentially steal this data.
*   **Information Leakage**: The analysis performed by this tool is designed to reveal patterns in your typing. The generated plots, statistics, and optimized keyboard layouts can inadvertently leak information about what you type. For example, high-frequency keys could give clues about the languages you speak or even commonly used passwords.
*   **Password Security**: The security of your logged data depends entirely on the strength of the password you use for encryption. A weak password could be cracked by an attacker who gains access to your `keylog.csv` and `key.salt` files.

### Recommendations

*   **Use a Strong Password**: Choose a long, complex, and unique password to encrypt your keylog data.
*   **Secure Your Machine**: The best way to protect your data is to ensure that your computer is secure from unauthorized access.
*   **Be Mindful of the Analysis**: Be aware that the output of the analysis scripts can reveal information about your typing habits. Do not share the generated plots or layouts if you are concerned about your privacy.

### 2. Analyze Your Typing Habits

Run the analytics script to generate plots from your keylog data.

```bash
python analytics.py keylog.csv
```

This will generate two files:

*   `key_press_distribution.png`: A bar chart showing the frequency of each key press.
*   `key_press_durations.png`: A box plot showing the duration of each key press.

### 3. Generate an Optimized Keyboard Layout

Run the keyboard layout optimizer to generate a layout based on your typing data.

```bash
python keyboard_layout.py keylog.csv
```

This will:

*   Print the optimized layout to the console.
*   Print detailed statistics about the layout.
*   Print a comparison with the QWERTY layout.
*   Print per-finger strain scores (0–100) and the contributing volume/effort/fatigue/rest terms.
*   Save the layout to a JSON file (e.g., `keyboard_layout_20251116_103000.json`).
*   Save a visualization of the layout to `keyboard_layout.png`.

### 4. Run Standalone Strain Analysis

You can also compute per-finger strain scores on the raw keylog (using the default QWERTY layout) without running the optimizer:

```bash
python finger_strain.py keylog.csv
```

## How It Works

The keyboard layout optimizer uses a two-step process:

1.  **Greedy Initialization**: It creates an initial layout by placing the most frequent keys in the positions with the lowest effort scores.
2.  **Hill-Climbing Optimization**: It iteratively swaps pairs of keys and keeps the swap if it improves the layout's score. The score is a measure of ergonomic comfort, penalizing same-finger bigrams and rewarding hand alternation and inward rolls.

The result is a keyboard layout that is tailored to your specific typing patterns, which can help reduce hand and finger strain.

## Literature Review: Estimating Finger Strain from Keylogs

Nobody today can directly measure “finger strain” from a plain keylogger, but there is a solid chain of research:

1. Software keyloggers to measure keystroke timing accurately.
2. Evidence that keystroke duration and timing change with muscle fatigue.
3. Models that treat keystroke volume, speed, and rest breaks as exposure to MSD risk.
4. Keyboard-layout “typing effort” models that convert which key a finger hits into a numeric effort score.

### 1. What You Can and Cannot Get from a Keylogger

Keyloggers give you key identity, event type, and timestamp. They do not capture force, posture, or EMG. Research shows you can still use keystroke timing and counts as a surrogate for muscle fatigue and as an exposure metric ([PubMed][1]). Modern approaches treat “finger strain from keylogs” as latent and use proxies: effort, fatigue, and exposure.

### 2. Keystroke Duration as a Fatigue / Strain Proxy

* **Chang et al. (2008)**: Keystroke duration increased when finger flexor muscles were fatigued; changes paralleled fatigue in muscle twitch duration ([PMC][2]).
* **Kim & Johnson (2011)**: Validated software-only millisecond-accurate keystroke duration as a non-invasive fatigue indicator ([PubMed][1]).
* **Kim & Johnson (2012)**: Showed digital keyboard signals can estimate typing forces; durations are sensitive to exposure changes ([Taylor & Francis Online][3]).

Takeaway: If your keylogger records key-down and key-up at millisecond resolution, per-key duration is the best validated software-only proxy for finger fatigue/strain.

### 3. Keystroke Volume and Intensity as Exposure to MSD Risk

* **Wellnomics**: Keystroke counts can beat “hours at computer” as predictors of symptoms; exposure metrics include total keystrokes, keystrokes per minute, and intensive period duration ([Wellnomics][4]).
* **Epidemiology (e.g., TNO)**: Software-logged keyboard/mouse use relates “% time typing”, “KPM”, and long bouts to neck/upper-extremity symptoms and peak exposure days ([TNO Publications][5]).
* **Lab/clinical**: Typing speed shifts force/activation patterns in forearm muscles, a driver for strain ([ScienceDirect][6]).

Takeaway: From key logs you can compute keystrokes per minute, continuous typing bouts, and per-finger workloads, and treat them as dose in a dose–response model of strain.

### 4. Typing Effort Models from Keyboard Layout Research

* **Carpalx**: Parametrized effort model using finger travel, row/finger penalties, and stroke path penalties (in-rolls vs out-rolls, same-finger bigrams) on letter/bigram frequencies ([mk.bcgsc.ca][7], [mk.bcgsc.ca][8]).
* **Newer layouts (Engram, Colemak-DH, 2024 thesis)**: Refinements with base effort grids, finger-strength weights, and comfort metrics ([Colemak Mods][9]).

Takeaway: With keylogger data plus geometry/finger mapping, you can compute per-finger effort scores that reflect distance and awkward motion as a structural component of strain.

### 5. Software “Strain Index” Models (RSIGuard, Etc.)

Commercial tools log keystroke/mouse use and compute “accumulated strain” and “natural rest,” triggering breaks based on strain instead of raw time. They integrate activity over time with decay, ramp strain faster during intense bursts, and reduce it during idle periods ([RSIGuard][10]).

### 6. Features You Can Derive from a QWERTY Keylogger (Per Finger)

1. **Keystroke duration–based fatigue**: Rolling means/variance; increasing durations signal fatigue ([PMC][2]).
2. **Volume and intensity**: KPM overall/per finger, burst length distributions, fraction of high-intensity time ([Wellnomics][4]).
3. **Geometry / effort**: Base key costs, travel distance from home, row/finger penalties, same-finger bigram counts (Carpalx-style) ([mk.bcgsc.ca][7]).
4. **Recovery / rest**: Inter-keystroke gaps; finger-specific idle windows as microbreaks.

### 7. Constructing a “Finger Strain Index” from Key Logs

A state-of-the-art inspired model for finger f over window t:

> Strain_f(t) = Exposure_f(t) × Effort_f(t) × FatigueFactor_f(t) − Recovery_f(t)

Where exposure comes from KPM/bursting, effort from geometry/penalties, fatigue from duration vs baseline, and recovery from rest decay (RSIGuard-inspired). Rescale to 0–100 and flag acute/overloaded fingers.

### 8. Gaps and Open Problems

* True strain depends on force, posture, anatomy—keystrokes do not capture these ([MedRxiv][11]).
* Keystroke-duration studies are often lab-based; thresholds for real-world coding remain open ([PMC][2]).
* No universal “finger strain index”; different models pick different weights/decay.

To do: collect a few EMG/motion sessions, fit a supervised model predicting EMG load from keystroke-derived features, then deploy the keylogger-only model. This mirrors current digital-phenotyping and HCI directions ([PMC][12]).

[1]: https://pubmed.ncbi.nlm.nih.gov/22256048/?utm_source=chatgpt.com "Validation of a software program for measuring fatigue- ..."
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3256245/?utm_source=chatgpt.com "Typing keystroke duration changed after submaximal ..."
[3]: https://www.tandfonline.com/doi/abs/10.1080/00140139.2012.709542?utm_source=chatgpt.com "Viability of using digital signals from the keyboard to ..."
[4]: https://wellnomics.com/wp-content/uploads/2020/01/Wellnomics-white-paper-Keystrokes-vs-Time-as-a-risk-factor-for-MSDs.pdf?utm_source=chatgpt.com "Keystrokes vs Time as a risk factor for musculoskeletal ..."
[5]: https://publications.tno.nl/publication/34626197/CoNRon/richter-2012-peak.pdf?utm_source=chatgpt.com "Original article"
[6]: https://www.sciencedirect.com/science/article/abs/pii/S016981410500048X?utm_source=chatgpt.com "The effects of typing speed and force on motor control in ..."
[7]: https://mk.bcgsc.ca/carpalx?utm_source=chatgpt.com "Carpalx - keyboard layout optimizer"
[8]: https://mk.bcgsc.ca/carpalx/?typing_effort=&utm_source=chatgpt.com "Carpalx typing effort model"
[9]: https://colemakmods.github.io/mod-dh/model.html?utm_source=chatgpt.com "Colemak Mod-DH - Keyboard effort grid - GitHub Pages"
[10]: https://www.rsiguard.com/documents/program/compare.htm?utm_source=chatgpt.com "RSIGuard Competitive Analysis"
[11]: https://www.medrxiv.org/content/10.1101/2021.06.09.21258367v1.full-text?utm_source=chatgpt.com "Relationship between the kinematics of wrist/finger joint ..."
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12563232/?utm_source=chatgpt.com "Mobile Typing as a Window into Sensorimotor and Cognitive ..."
