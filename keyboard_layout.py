import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class KeyboardLayoutOptimizer:
    """Optimize keyboard layout based on typing data and ergonomic principles."""
    
    def __init__(self, keylog_file='keylog.csv'):
        self.keylog_file = keylog_file
        self.df = None
        self.key_frequencies = {}
        self.bigram_frequencies = {}
        
        # Define finger strength (1=weakest, 5=strongest) - currently not used in scoring,
        # but kept for future extensions.
        self.finger_strength = {
            'L_pinky': 1, 'L_ring': 2, 'L_middle': 3, 'L_index': 4,
            'R_index': 4, 'R_middle': 3, 'R_ring': 2, 'R_pinky': 1
        }
        
        # Effort scores for each position (row, column) - lower is better.
        # Now defined for the full 3x10 alpha block (including center columns 4 and 5).
        self.position_effort = {
            # Top row
            (0, 0): 4.0, (0, 1): 2.5, (0, 2): 2.0, (0, 3): 2.0,
            (0, 4): 2.3, (0, 5): 2.3, (0, 6): 2.0, (0, 7): 2.0,
            (0, 8): 2.5, (0, 9): 4.0,
            # Home row
            (1, 0): 2.5, (1, 1): 1.5, (1, 2): 1.0, (1, 3): 1.0,
            (1, 4): 1.3, (1, 5): 1.3, (1, 6): 1.0, (1, 7): 1.0,
            (1, 8): 1.5, (1, 9): 2.5,
            # Bottom row
            (2, 0): 5.0, (2, 1): 3.5, (2, 2): 3.0, (2, 3): 2.5,
            (2, 4): 2.7, (2, 5): 2.7, (2, 6): 2.5, (2, 7): 3.0,
            (2, 8): 3.5, (2, 9): 5.0,
        }
        
        # Finger assignment per column for all rows.
        # 0: L_pinky, 1: L_ring, 2: L_middle, 3-4: L_index,
        # 5-6: R_index, 7: R_middle, 8: R_ring, 9: R_pinky
        self.position_finger = {}
        for row in range(3):
            for col in range(10):
                if col == 0:
                    finger = 'L_pinky'
                elif col == 1:
                    finger = 'L_ring'
                elif col == 2:
                    finger = 'L_middle'
                elif col in (3, 4):
                    finger = 'L_index'
                elif col in (5, 6):
                    finger = 'R_index'
                elif col == 7:
                    finger = 'R_middle'
                elif col == 8:
                    finger = 'R_ring'
                else:  # col == 9
                    finger = 'R_pinky'
                self.position_finger[(row, col)] = finger
        
        # Current layout: mapping key -> (row, col)
        self.current_layout = None
        self.layout_score = None
    
    # ----------------------------------------------------------------------
    # Data Loading And Analysis
    # ----------------------------------------------------------------------
    def load_and_analyze_data(self):
        """Load keystroke data and compute key and bigram frequencies."""
        print("Loading keystroke data...")
        self.df = pd.read_csv(self.keylog_file)
        
        # Filter out special keys for layout optimization
        special_keys = [
            'Key.', 'shift', 'ctrl', 'alt', 'cmd', 'tab', 'enter',
            'backspace', 'space', 'esc', 'caps', 'delete'
        ]
        
        # Keep only "character" keys (letters, symbols) for layout
        letter_keys = []
        for key in self.df['key']:
            key_str = str(key)
            if not any(special in key_str for special in special_keys) and len(key_str) <= 2:
                letter_keys.append(key_str.lower())
        
        # Calculate frequencies
        self.key_frequencies = Counter(letter_keys)
        print(f"Analyzed {len(letter_keys)} keystrokes")
        print(f"Unique keys: {len(self.key_frequencies)}")
        
        # Calculate bigram frequencies
        self._calculate_bigrams(letter_keys)
        
        return self.key_frequencies
    
    def _calculate_bigrams(self, keys):
        """Calculate bigram (two-key sequence) frequencies."""
        bigrams = []
        for i in range(len(keys) - 1):
            bigram = keys[i] + keys[i + 1]
            bigrams.append(bigram)
        
        self.bigram_frequencies = Counter(bigrams)
        print(f"Unique bigrams: {len(self.bigram_frequencies)}")
    
    # ----------------------------------------------------------------------
    # Layout Generation And Optimization
    # ----------------------------------------------------------------------
    def generate_optimal_layout(self, max_keys=None):
        """
        Generate an initial keyboard layout using a greedy algorithm.
        
        - Most frequent keys go to positions with lowest effort.
        - Only the top `max_keys` are used if provided; otherwise, we use as many
          keys as there are positions available.
        """
        print("\nGenerating initial layout...")
        
        if not self.key_frequencies:
            raise ValueError("No key frequency data. Run load_and_analyze_data() first.")
        
        # Sort keys by descending frequency
        sorted_keys = sorted(
            self.key_frequencies.items(), key=lambda x: x[1], reverse=True
        )
        
        # Sort positions by effort (best positions first)
        all_positions = sorted(
            self.position_effort.items(), key=lambda x: x[1]
        )  # list of (pos, effort)
        
        if max_keys is None:
            max_keys = len(all_positions)
        
        # Limit to max_keys and available positions
        sorted_keys = sorted_keys[:max_keys]
        
        layout = {}
        for i, (key, freq) in enumerate(sorted_keys):
            if i >= len(all_positions):
                break
            pos, effort = all_positions[i]
            layout[key] = pos  # store only position; finger/effort derived later
        
        self.current_layout = layout
        self.layout_score = self._score_layout(layout)
        print(f"Initial layout score: {self.layout_score:.2f}")
        return layout
    
    def optimize_for_bigrams(self, iterations=1000):
        """
        Optimize layout to minimize same-finger bigrams and maximize hand alternation.
        
        Uses a simple hill-climbing approach:
        - Start from current layout (or generate one).
        - Repeatedly swap positions of two random keys.
        - Keep the swap if it improves the score.
        
        NOTE: Layout representation is key -> (row, col) with immutable tuples,
        so dict.copy() is safe (no shared inner mutable objects).
        """
        print(f"\nOptimizing for bigrams ({iterations} iterations)...")
        
        if not self.current_layout:
            self.generate_optimal_layout()
        
        best_layout = self.current_layout.copy()
        best_score = self._score_layout(best_layout)
        
        keys = list(best_layout.keys())
        if len(keys) < 2:
            print("Not enough keys to optimize.")
            self.current_layout = best_layout
            self.layout_score = best_score
            return best_layout
        
        for i in range(iterations):
            # Create a candidate layout by swapping two random keys
            test_layout = best_layout.copy()
            k1, k2 = np.random.choice(keys, 2, replace=False)
            pos1, pos2 = test_layout[k1], test_layout[k2]
            test_layout[k1], test_layout[k2] = pos2, pos1
            
            score = self._score_layout(test_layout)
            
            if score < best_score:
                best_layout = test_layout
                best_score = score
                if i % 50 == 0:
                    print(f"  Iteration {i}: Score improved to {best_score:.2f}")
        
        self.current_layout = best_layout
        self.layout_score = best_score
        print(f"\nFinal layout score: {best_score:.2f}")
        return best_layout
    
    def _score_layout(self, layout):
        """
        Score a layout (lower is better).
        
        Factors:
        - Frequency × Effort (frequent keys in easy positions)
        - Same-finger bigram penalty
        - Hand alternation bonus
        - Inward roll bonus (same-hand, inward movement)
        """
        score = 0.0
        
        # 1. Effort score (frequency × effort)
        for key, pos in layout.items():
            freq = self.key_frequencies.get(key, 0)
            if freq == 0:
                continue
            effort = self.position_effort.get(pos, 5.0)
            score += freq * effort
        
        # 2. Bigram penalties/bonuses
        for bigram, freq in self.bigram_frequencies.items():
            if len(bigram) != 2:
                continue
            key1, key2 = bigram[0], bigram[1]
            
            if key1 not in layout or key2 not in layout:
                continue
            
            pos1 = layout[key1]
            pos2 = layout[key2]
            finger1 = self.position_finger[pos1]
            finger2 = self.position_finger[pos2]
            
            # Same-finger penalty (avoid)
            if finger1 == finger2:
                score += freq * 10.0
            
            # Hand alternation bonus
            hand1 = 'L' if finger1.startswith('L') else 'R'
            hand2 = 'L' if finger2.startswith('L') else 'R'
            if hand1 != hand2:
                score -= freq * 0.5
            
            # Inward roll bonus (same hand, but comfortable direction)
            if hand1 == hand2:
                col1, col2 = pos1[1], pos2[1]
                # Inward roll: right hand goes right->left, left hand goes left->right
                if (hand1 == 'R' and col1 > col2) or (hand1 == 'L' and col1 < col2):
                    score -= freq * 0.3
        
        return score
        
        
        def compute_qwerty_finger_strain(self,
                                     same_finger_penalty=10.0,
                                     use_duration=False):
        """
        Compute a relative 'strain index' per finger for a standard QWERTY layout
        based on your actual keystroke data.

        Parameters
        ----------
        same_finger_penalty : float
            Penalty added per same-finger bigram occurrence (scaled by bigram frequency).
        use_duration : bool
            If True, weight keystrokes by key 'duration' from self.df instead of just counts.

        Returns
        -------
        strain_raw : dict
            Raw strain index per finger (arbitrary units).
        strain_percent : dict
            Strain per finger as percentage of total.
        details : dict
            Contains 'base_load' and 'dynamic_load' per finger.
        """

        if self.df is None or self.key_frequencies is None:
            raise ValueError("Call load_and_analyze_data() before computing strain.")

        # --- 1) Define QWERTY positions ---
        qwerty_positions = {
            # Top row
            'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
            'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
            # Home row
            'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
            'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8), ';': (1, 9),
            # Bottom row
            'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
            'n': (2, 5), 'm': (2, 6), ',': (2, 7), '.': (2, 8), '/': (2, 9),
        }

        # Finger mapping by column (same as in current class)
        def column_to_finger(col: int) -> str:
            if col == 0:
                return 'L_pinky'
            elif col == 1:
                return 'L_ring'
            elif col == 2:
                return 'L_middle'
            elif col in (3, 4):
                return 'L_index'
            elif col in (5, 6):
                return 'R_index'
            elif col == 7:
                return 'R_middle'
            elif col == 8:
                return 'R_ring'
            else:
                return 'R_pinky'

        # --- 2) Build per-finger base load from monograms ---
        fingers = ['L_pinky', 'L_ring', 'L_middle', 'L_index',
                   'R_index', 'R_middle', 'R_ring', 'R_pinky']
        base_load = {f: 0.0 for f in fingers}
        dynamic_load = {f: 0.0 for f in fingers}

        # If use_duration=True, we need an alternative frequency based on summed duration
        if use_duration:
            # Build per-key duration sum from self.df
            key_durations = {}
            for _, row in self.df.iterrows():
                key = str(row['key']).lower()
                if key in qwerty_positions:
                    key_durations[key] = key_durations.get(key, 0.0) + float(row['duration'])
        else:
            key_durations = None  # we will use self.key_frequencies

        # Base load: frequency (or total duration) × position effort
        for key in qwerty_positions.keys():
            pos = qwerty_positions[key]
            effort = self.position_effort.get(pos, 5.0)

            if use_duration:
                freq_like = key_durations.get(key, 0.0)
            else:
                freq_like = self.key_frequencies.get(key, 0)

            if freq_like <= 0:
                continue

            _, col = pos
            finger = column_to_finger(col)
            base_load[finger] += freq_like * effort

        # --- 3) Dynamic load from same-finger bigrams on QWERTY ---
        for bigram, freq in self.bigram_frequencies.items():
            if len(bigram) != 2:
                continue
            k1, k2 = bigram[0].lower(), bigram[1].lower()

            if k1 not in qwerty_positions or k2 not in qwerty_positions:
                continue

            pos1 = qwerty_positions[k1]
            pos2 = qwerty_positions[k2]
            f1 = column_to_finger(pos1[1])
            f2 = column_to_finger(pos2[1])

            if use_duration:
                # crude approximation: scale by avg duration of the two keys
                dur1 = key_durations.get(k1, 0.0)
                dur2 = key_durations.get(k2, 0.0)
                freq_like = (dur1 + dur2) / 2.0  # or just dur1+dur2
            else:
                freq_like = freq

            if f1 == f2:
                dynamic_load[f1] += freq_like * same_finger_penalty

        # --- 4) Combine and normalize ---
        strain_raw = {}
        for f in fingers:
            strain_raw[f] = base_load[f] + dynamic_load[f]

        total_strain = sum(strain_raw.values()) or 1.0
        strain_percent = {f: (strain_raw[f] / total_strain) * 100.0 for f in fingers}

        details = {
            'base_load': base_load,
            'dynamic_load': dynamic_load,
        }

        # Pretty-print summary
        print("\n" + "=" * 60)
        print("QWERTY FINGER STRAIN INDEX")
        print("=" * 60)
        print("(Relative units; higher = more modeled load)")
        for f in fingers:
            print(f"{f:8s}  Base: {base_load[f]:10.1f}  Dyn: {dynamic_load[f]:10.1f}  "
                  f"Total: {strain_raw[f]:10.1f}  ({strain_percent[f]:5.1f}%)")
        print("=" * 60)

        return strain_raw, strain_percent, details

    
    # ----------------------------------------------------------------------
    # Visualization And Reporting
    # ----------------------------------------------------------------------
    def visualize_layout(self, save_file='keyboard_layout.png'):
        """Visualize the keyboard layout with a frequency heatmap."""
        if not self.current_layout:
            print("No layout generated yet!")
            return
        
        # Create matrices for keys and frequencies
        keyboard = [['' for _ in range(10)] for _ in range(3)]
        frequency_map = [[0 for _ in range(10)] for _ in range(3)]
        
        for key, pos in self.current_layout.items():
            row, col = pos
            keyboard[row][col] = key
            frequency_map[row][col] = self.key_frequencies.get(key, 0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Layout visualization with frequency heatmap
        sns.heatmap(
            frequency_map,
            annot=np.array(keyboard),
            fmt='',
            cmap='YlOrRd',
            ax=ax1,
            cbar_kws={'label': 'Frequency'}
        )
        ax1.set_title(
            'Optimized Keyboard Layout (With Frequency Heatmap)',
            fontsize=16,
            fontweight='bold'
        )
        ax1.set_xlabel('Column', fontsize=12)
        ax1.set_ylabel('Row', fontsize=12)
        ax1.set_yticklabels(['Top', 'Home', 'Bottom'])
        
        # Top 20 most frequent keys in the layout
        top_keys = sorted(
            self.current_layout.keys(),
            key=lambda k: self.key_frequencies.get(k, 0),
            reverse=True
        )[:20]
        keys = top_keys
        freqs = [self.key_frequencies.get(k, 0) for k in keys]
        
        ax2.barh(keys, freqs, color='steelblue')
        ax2.set_xlabel('Frequency', fontsize=12)
        ax2.set_ylabel('Key', fontsize=12)
        ax2.set_title(
            'Top 20 Most Frequent Keys In The Layout',
            fontsize=14,
            fontweight='bold'
        )
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nLayout visualization saved to {save_file}")
        
        return fig
    
    def print_layout(self):
        """Print the keyboard layout in a readable 3×10 grid format."""
        if not self.current_layout:
            print("No layout generated yet!")
            return
        
        keyboard = [['' for _ in range(10)] for _ in range(3)]
        for key, pos in self.current_layout.items():
            row, col = pos
            keyboard[row][col] = key.upper()
        
        print("\n" + "=" * 60)
        print("YOUR OPTIMIZED KEYBOARD LAYOUT")
        print("=" * 60)
        print("\nTop Row:    ", " ".join(f"[{k:^3}]" if k else "[   ]" for k in keyboard[0]))
        print("Home Row:   ", " ".join(f"[{k:^3}]" if k else "[   ]" for k in keyboard[1]))
        print("Bottom Row: ", " ".join(f"[{k:^3}]" if k else "[   ]" for k in keyboard[2]))
        print("\n" + "=" * 60)
        
        if self.layout_score is not None:
            print(f"Layout Score: {self.layout_score:.2f} (lower is better)")
        print()
    
    def save_layout(self, filename=None):
        """Save the layout to a JSON file with timestamp."""
        if not self.current_layout:
            print("No layout generated yet!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keyboard_layout_{timestamp}.json"
        
        layout_data = {
            'timestamp': datetime.now().isoformat(),
            'score': self.layout_score,
            'layout': {}
        }
        
        for key, pos in self.current_layout.items():
            row, col = pos
            finger = self.position_finger[pos]
            effort = self.position_effort.get(pos, None)
            freq = self.key_frequencies.get(key, 0)
            layout_data['layout'][key] = {
                'position': [row, col],
                'finger': finger,
                'frequency': freq,
                'effort': effort,
            }
        
        with open(filename, 'w') as f:
            json.dump(layout_data, f, indent=2)
        
        print(f"Layout saved to {filename}")
        return filename
    
    def compare_with_qwerty(self):
        """Compare the optimized layout with a QWERTY layout using the same scoring function."""
        if not self.current_layout:
            print("No layout generated yet!")
            return
        
        # Define QWERTY layout (3x10 grid)
        qwerty = {
            # Top row
            'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
            'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
            # Home row
            'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
            'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8), ';': (1, 9),
            # Bottom row
            'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
            'n': (2, 5), 'm': (2, 6), ',': (2, 7), '.': (2, 8), '/': (2, 9),
        }
        
        # Ensure we compare the same set of keys as in the optimized layout
        keys_in_layout = set(self.current_layout.keys())
        qwerty_layout = {}
        for key in keys_in_layout:
            if key in qwerty and qwerty[key] in self.position_effort:
                qwerty_layout[key] = qwerty[key]
        
        if not qwerty_layout:
            print("No overlapping keys between QWERTY and optimized layout to compare.")
            return
        
        qwerty_score = self._score_layout(qwerty_layout)
        optimized_score = self.layout_score
        if optimized_score is None:
            optimized_score = self._score_layout(self.current_layout)
        
        print("\n" + "=" * 60)
        print("LAYOUT COMPARISON")
        print("=" * 60)
        print(f"QWERTY Score (Same Keys):    {qwerty_score:.2f}")
        print(f"Optimized Score:             {optimized_score:.2f}")
        improvement = ((qwerty_score - optimized_score) / qwerty_score) * 100
        print(f"Improvement:                 {improvement:.1f}%")
        print("=" * 60)
    
    def generate_statistics(self):
        """Generate detailed statistics about the current layout."""
        if not self.current_layout:
            print("No layout generated yet!")
            return
        
        stats = {
            'total_keys': len(self.current_layout),
            'home_row_keys': 0,
            'home_row_freq_percent': 0.0,
            'hand_balance': {'left': 0.0, 'right': 0.0},
            'finger_usage': defaultdict(int),
            'same_finger_bigrams': 0,
            'hand_alternation_bigrams': 0,
        }
        
        total_freq = sum(self.key_frequencies.get(k, 0) for k in self.current_layout.keys())
        if total_freq == 0:
            total_freq = 1  # avoid division by zero
        
        # Monogram-based stats
        for key, pos in self.current_layout.items():
            freq = self.key_frequencies.get(key, 0)
            row, col = pos
            
            # Home row usage
            if row == 1:
                stats['home_row_keys'] += 1
                stats['home_row_freq_percent'] += freq
            
            finger = self.position_finger[pos]
            hand = 'left' if finger.startswith('L') else 'right'
            
            stats['hand_balance'][hand] += freq
            stats['finger_usage'][finger] += freq
        
        stats['home_row_freq_percent'] = (stats['home_row_freq_percent'] / total_freq) * 100.0
        stats['hand_balance']['left'] = (stats['hand_balance']['left'] / total_freq) * 100.0
        stats['hand_balance']['right'] = (stats['hand_balance']['right'] / total_freq) * 100.0
        
        # Bigram-based stats
        for bigram, freq in self.bigram_frequencies.items():
            if len(bigram) != 2:
                continue
            key1, key2 = bigram[0], bigram[1]
            if key1 not in self.current_layout or key2 not in self.current_layout:
                continue
            
            pos1 = self.current_layout[key1]
            pos2 = self.current_layout[key2]
            finger1 = self.position_finger[pos1]
            finger2 = self.position_finger[pos2]
            
            if finger1 == finger2:
                stats['same_finger_bigrams'] += freq
            
            hand1 = 'L' if finger1.startswith('L') else 'R'
            hand2 = 'L' if finger2.startswith('L') else 'R'
            if hand1 != hand2:
                stats['hand_alternation_bigrams'] += freq
        
        print("\n" + "=" * 60)
        print("LAYOUT STATISTICS")
        print("=" * 60)
        print(f"Total keys placed: {stats['total_keys']}")
        print(f"Keys on home row:  {stats['home_row_keys']}")
        print(f"Home row usage:    {stats['home_row_freq_percent']:.1f}%")
        print(f"\nHand balance (by frequency):")
        print(f"  Left hand:  {stats['hand_balance']['left']:.1f}%")
        print(f"  Right hand: {stats['hand_balance']['right']:.1f}%")
        print(f"\nBigram analysis:")
        print(f"  Hand alternation: {stats['hand_alternation_bigrams']} occurrences")
        print(f"  Same finger:      {stats['same_finger_bigrams']} occurrences")
        print("=" * 60)
        
        return stats


def main():
    """Main function to run the keyboard layout optimizer."""
    print("=" * 60)
    print("KEYBOARD LAYOUT OPTIMIZER")
    print("=" * 60)
    
    optimizer = KeyboardLayoutOptimizer('keylog.csv')
    
    # Load and analyze data
    optimizer.load_and_analyze_data()
    
    # Generate initial layout
    optimizer.generate_optimal_layout()
    
    # Optimize for bigrams
    optimizer.optimize_for_bigrams(iterations=1000)
    
    # Display results
    optimizer.print_layout()
    optimizer.generate_statistics()
    optimizer.compare_with_qwerty()
    
    # Save and visualize
    optimizer.save_layout()
    optimizer.visualize_layout()
    
    print("\n✓ Keyboard layout optimization complete!")
    print("  - Layout saved to JSON file")
    print("  - Visualization saved to keyboard_layout.png")
    print("\nTo regenerate with new data, just run this script again!")


if __name__ == "__main__":
    main()
