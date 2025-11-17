"""
Analyze per-finger strain from an encrypted keylog.

Strain_f = ExposureEffort_f × FatigueFactor_f × RecoveryFactor_f, then
normalized to 0–100 across fingers. Everything runs on the existing encrypted
`keylog.csv` plus layout/finger/effort maps from KeyboardLayoutOptimizer.
"""

from __future__ import annotations

import base64
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ----------------------------------------------------------------------
# Encryption Helper (aligned with analytics.py / keyboard_layout.py)
# ----------------------------------------------------------------------
def get_encryption_key(password: str, salt: bytes) -> bytes:
    """Derive a Fernet key from the password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


# ----------------------------------------------------------------------
# Default Layout / Ergonomics (QWERTY 3×10 Block)
# ----------------------------------------------------------------------
def default_qwerty_layout() -> Dict[str, Tuple[int, int]]:
    """Return a 3×10 QWERTY layout mapping: key -> (row, col)."""
    return {
        # Top row
        "q": (0, 0),
        "w": (0, 1),
        "e": (0, 2),
        "r": (0, 3),
        "t": (0, 4),
        "y": (0, 5),
        "u": (0, 6),
        "i": (0, 7),
        "o": (0, 8),
        "p": (0, 9),
        # Home row
        "a": (1, 0),
        "s": (1, 1),
        "d": (1, 2),
        "f": (1, 3),
        "g": (1, 4),
        "h": (1, 5),
        "j": (1, 6),
        "k": (1, 7),
        "l": (1, 8),
        ";": (1, 9),
        # Bottom row
        "z": (2, 0),
        "x": (2, 1),
        "c": (2, 2),
        "v": (2, 3),
        "b": (2, 4),
        "n": (2, 5),
        "m": (2, 6),
        ",": (2, 7),
        ".": (2, 8),
        "/": (2, 9),
    }


def default_position_effort() -> Dict[Tuple[int, int], float]:
    """Return the 3×10 effort grid used in KeyboardLayoutOptimizer."""
    return {
        # Top row
        (0, 0): 4.0,
        (0, 1): 2.5,
        (0, 2): 2.0,
        (0, 3): 2.0,
        (0, 4): 2.3,
        (0, 5): 2.3,
        (0, 6): 2.0,
        (0, 7): 2.0,
        (0, 8): 2.5,
        (0, 9): 4.0,
        # Home row
        (1, 0): 2.5,
        (1, 1): 1.5,
        (1, 2): 1.0,
        (1, 3): 1.0,
        (1, 4): 1.3,
        (1, 5): 1.3,
        (1, 6): 1.0,
        (1, 7): 1.0,
        (1, 8): 1.5,
        (1, 9): 2.5,
        # Bottom row
        (2, 0): 5.0,
        (2, 1): 3.5,
        (2, 2): 3.0,
        (2, 3): 2.5,
        (2, 4): 2.7,
        (2, 5): 2.7,
        (2, 6): 2.5,
        (2, 7): 3.0,
        (2, 8): 3.5,
        (2, 9): 5.0,
    }


def default_position_finger() -> Dict[Tuple[int, int], str]:
    """Return (row, col) -> finger mapping for a 3×10 layout."""
    mapping: Dict[Tuple[int, int], str] = {}
    for row in range(3):
        for col in range(10):
            if col == 0:
                finger = "L_pinky"
            elif col == 1:
                finger = "L_ring"
            elif col == 2:
                finger = "L_middle"
            elif col in (3, 4):
                finger = "L_index"
            elif col in (5, 6):
                finger = "R_index"
            elif col == 7:
                finger = "R_middle"
            elif col == 8:
                finger = "R_ring"
            else:
                finger = "R_pinky"
            mapping[(row, col)] = finger
    return mapping


def default_finger_strength() -> Dict[str, int]:
    """Return default finger strength weights (1=weakest, 5=strongest)."""
    return {
        "L_pinky": 1,
        "L_ring": 2,
        "L_middle": 3,
        "L_index": 4,
        "R_index": 4,
        "R_middle": 3,
        "R_ring": 2,
        "R_pinky": 1,
    }


@dataclass
class FingerFeatures:
    """Container for per-finger features and scores."""

    n_keystrokes: int
    volume_term: float
    mean_effort_weighted: float
    effort_term: float
    exposure_effort: float
    baseline_duration: float
    late_duration: float
    fatigue_ratio: float
    fatigue_factor: float
    rest_ratio: float
    recovery_factor: float
    strain_raw: float
    strain_score: float


# ----------------------------------------------------------------------
# Finger Strain Analyzer
# ----------------------------------------------------------------------
class FingerStrainAnalyzer:
    """
    Compute per-finger strain scores from an encrypted keylog.

    Strain_f = ExposureEffort_f × FatigueFactor_f × RecoveryFactor_f,
    then normalized to 0–100 across fingers.
    """

    def __init__(
        self,
        keylog_file: str,
        layout: Optional[Dict[str, Tuple[int, int]]] = None,
        position_effort: Optional[Dict[Tuple[int, int], float]] = None,
        position_finger: Optional[Dict[Tuple[int, int], str]] = None,
        finger_strength: Optional[Dict[str, int]] = None,
        volume_weight: float = 0.6,
        effort_weight: float = 0.4,
        recovery_weight: float = 0.6,
        rest_threshold_sec: float = 3.0,
    ):
        self.keylog_file = keylog_file

        self.layout = layout or default_qwerty_layout()
        self.position_effort = position_effort or default_position_effort()
        self.position_finger = position_finger or default_position_finger()
        self.finger_strength = finger_strength or default_finger_strength()

        self.volume_weight = volume_weight
        self.effort_weight = effort_weight
        self.recovery_weight = recovery_weight
        self.rest_threshold_sec = rest_threshold_sec

        self.df: Optional[pd.DataFrame] = None
        self.finger_features: Optional[Dict[str, FingerFeatures]] = None
        self.finger_scores: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Data Loading / Cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_key(key_str: str) -> Optional[str]:
        """Normalize key string and strip out non-character keys."""
        s = str(key_str)
        if s.startswith("Key."):
            return None
        if len(s) > 2:
            return None
        return s.lower()

    def load_keylog(
        self,
        fernet: Fernet,
        app_filter: Optional[str] = None,
        window_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Decrypt and load keylog.csv into a cleaned DataFrame."""
        df = pd.read_csv(self.keylog_file)

        try:
            df["key"] = df["key"].apply(lambda x: fernet.decrypt(str(x).encode()).decode())
            df["duration"] = df["duration"].apply(
                lambda x: float(fernet.decrypt(str(x).encode()).decode())
            )
            if "app" in df.columns:
                df["app"] = df["app"].apply(lambda x: fernet.decrypt(str(x).encode()).decode())
            if "window_title" in df.columns:
                df["window_title"] = df["window_title"].apply(
                    lambda x: fernet.decrypt(str(x).encode()).decode()
                )
        except InvalidToken:
            print("Decryption failed. Check your password/salt.")
            raise

        if app_filter and "app" in df.columns:
            df = df[df["app"].fillna("").str.lower().str.contains(app_filter.lower())]
        if window_filter and "window_title" in df.columns:
            df = df[df["window_title"].fillna("").str.lower().str.contains(window_filter.lower())]

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        df["key"] = df["key"].apply(self._canonical_key)
        df = df.dropna(subset=["key"])

        df = df[df["key"].isin(self.layout.keys())].copy()
        if df.empty:
            print("No usable key events found for the given layout.")
            self.df = df
            return df

        df["pos"] = df["key"].apply(self.layout.get)
        df = df.dropna(subset=["pos"])

        df["row"] = df["pos"].apply(lambda rc: rc[0])
        df["col"] = df["pos"].apply(lambda rc: rc[1])

        df["finger"] = df["pos"].apply(self.position_finger.get)
        df = df.dropna(subset=["finger"])

        df["effort"] = df["pos"].apply(lambda rc: self.position_effort.get(rc, 5.0))

        df["end_time"] = df["timestamp"] + pd.to_timedelta(df["duration"], unit="s")

        df = df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        return df

    # ------------------------------------------------------------------
    # Feature And Strain Computation
    # ------------------------------------------------------------------
    def compute_finger_features(self) -> Dict[str, FingerFeatures]:
        """Compute intermediate features (volume, effort, fatigue, rest) per finger."""
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Call load_keylog() first.")

        df = self.df
        features: Dict[str, FingerFeatures] = {}
        finger_groups = {f: g.copy() for f, g in df.groupby("finger")}

        max_strength = max(self.finger_strength.values())
        strength_scale = {
            f: max_strength / self.finger_strength.get(f, max_strength)
            for f in finger_groups.keys()
        }

        counts = {f: len(g) for f, g in finger_groups.items()}
        max_N = max(counts.values()) if counts else 0

        mean_effort_weighted: Dict[str, float] = {}
        for f, g in finger_groups.items():
            if g.empty:
                mean_effort_weighted[f] = 0.0
                continue
            scale = strength_scale[f]
            mean_effort_weighted[f] = float((g["effort"] * scale).mean())
        max_mean_effort = max(mean_effort_weighted.values()) if mean_effort_weighted else 0

        for f, g in finger_groups.items():
            g = g.sort_values("timestamp").reset_index(drop=True)
            n = len(g)

            volume_term = counts[f] / max_N if max_N > 0 else 0.0

            if max_mean_effort > 0:
                effort_term = mean_effort_weighted[f] / max_mean_effort
            else:
                effort_term = 0.0

            if n < 20:
                baseline_duration = float(g["duration"].median()) if n > 0 else 0.0
                late_duration = baseline_duration
                fatigue_ratio = 0.0
                fatigue_factor = 1.0
            else:
                q = max(5, n // 4)
                baseline_duration = float(g["duration"].iloc[:q].median())
                late_duration = float(g["duration"].iloc[-q:].median())
                if baseline_duration > 0:
                    raw_ratio = (late_duration - baseline_duration) / baseline_duration
                else:
                    raw_ratio = 0.0
                fatigue_ratio = max(0.0, min(raw_ratio, 1.0))
                fatigue_factor = 1.0 + fatigue_ratio

            if n <= 1:
                rest_ratio = 1.0
            else:
                t_start = g["timestamp"].iloc[0]
                t_end = g["timestamp"].iloc[-1]
                span = (t_end - t_start).total_seconds()
                if span <= 0:
                    rest_ratio = 1.0
                else:
                    gaps = g["timestamp"].iloc[1:].to_numpy() - g["end_time"].iloc[:-1].to_numpy()
                    gaps_sec = gaps / np.timedelta64(1, "s")
                    extra_rest = np.maximum(gaps_sec - self.rest_threshold_sec, 0.0)
                    rest_time = float(extra_rest.sum())
                    rest_ratio = max(0.0, min(rest_time / span, 1.0))

            recovery_factor = 1.0 - self.recovery_weight * rest_ratio
            recovery_factor = max(0.4, min(recovery_factor, 1.0))

            exposure_effort = self.volume_weight * volume_term + self.effort_weight * effort_term

            features[f] = FingerFeatures(
                n_keystrokes=counts[f],
                volume_term=volume_term,
                mean_effort_weighted=mean_effort_weighted[f],
                effort_term=effort_term,
                exposure_effort=exposure_effort,
                baseline_duration=baseline_duration,
                late_duration=late_duration,
                fatigue_ratio=fatigue_ratio,
                fatigue_factor=fatigue_factor,
                rest_ratio=rest_ratio,
                recovery_factor=recovery_factor,
                strain_raw=0.0,
                strain_score=0.0,
            )

        for f, feat in features.items():
            strain_raw = feat.exposure_effort * feat.fatigue_factor * feat.recovery_factor
            features[f].strain_raw = float(strain_raw)

        max_strain_raw = max((feat.strain_raw for feat in features.values()), default=0.0)

        for f, feat in features.items():
            if max_strain_raw > 0:
                features[f].strain_score = 100.0 * feat.strain_raw / max_strain_raw
            else:
                features[f].strain_score = 0.0

        self.finger_features = features
        self.finger_scores = {finger: feat.strain_score for finger, feat in features.items()}
        return features

    def get_strain_scores(self) -> Dict[str, float]:
        """Return {finger: strain_score} (0–100)."""
        if self.finger_scores is None:
            raise ValueError("No scores computed yet. Call compute_finger_features().")
        return self.finger_scores

    def print_report(self) -> None:
        """Pretty-print a small per-finger report."""
        if self.finger_features is None:
            raise ValueError("No features computed yet. Call compute_finger_features().")

        print("\n" + "=" * 60)
        print("PER-FINGER STRAIN REPORT (0–100, higher = more strain)")
        print("=" * 60)
        for finger, feat in sorted(self.finger_features.items()):
            print(
                f"{finger:8s} | "
                f"Strain: {feat.strain_score:6.1f}  "
                f"Vol: {feat.volume_term:.2f}  "
                f"Eff: {feat.effort_term:.2f}  "
                f"Fatigue: {feat.fatigue_ratio:.2f}  "
                f"Rest: {feat.rest_ratio:.2f}"
            )
        print("=" * 60)


# ----------------------------------------------------------------------
# Optional CLI Entry Point
# ----------------------------------------------------------------------
def main() -> None:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        description="Compute per-finger strain scores from an encrypted keylog."
    )
    parser.add_argument("keylog_file", help="Path to keylog CSV file.")
    parser.add_argument(
        "--use-qwerty",
        action="store_true",
        help="Use default QWERTY 3×10 layout (default if no custom layout passed).",
    )
    parser.add_argument(
        "--app",
        dest="app_filter",
        help="Only include keystrokes from apps matching this substring (case-insensitive).",
    )
    parser.add_argument(
        "--window",
        dest="window_filter",
        help="Only include keystrokes from windows matching this substring (case-insensitive).",
    )
    args = parser.parse_args()

    password = getpass.getpass("Enter the password to decrypt the keylog: ")
    salt_file = "key.salt"
    if not os.path.exists(salt_file):
        print("Salt file not found. Make sure 'key.salt' is in the same directory.")
        return
    with open(salt_file, "rb") as f:
        salt = f.read()

    key = get_encryption_key(password, salt)
    fernet = Fernet(key)

    layout = default_qwerty_layout() if args.use_qwerty else None
    analyzer = FingerStrainAnalyzer(keylog_file=args.keylog_file, layout=layout)
    analyzer.load_keylog(fernet, app_filter=args.app_filter, window_filter=args.window_filter)
    if analyzer.df is None or analyzer.df.empty:
        print("No data to analyze.")
        return
    analyzer.compute_finger_features()
    analyzer.print_report()


if __name__ == "__main__":
    main()
