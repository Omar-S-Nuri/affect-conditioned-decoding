# hippocampus_memory.py
import numpy as np

class HippocampusMemory:
    def __init__(self, memory_size=5, decay_rate=0.2):
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        # Speicher für die letzten (Text, 7D-Vektor)-Paare
        self.episodic_buffer = []

    def push_experience(self, text, vector):
        """Speichert die aktuelle Erfahrung im Kurzzeitgedächtnis."""
        if len(self.episodic_buffer) >= self.memory_size:
            self.episodic_buffer.pop(0)  # Älteste Erinnerung löschen
        self.episodic_buffer.append({"text": text.lower(), "vector": np.array(vector)})

    def get_associative_bias(self, current_words):
        """Berechnet das emotionale Echo basierend auf vergangenen Reizen."""
        if not self.episodic_buffer:
            return np.zeros(7)

        # Wenn der aktuelle Satz Wörter enthält, die semantisch/kontextuell zu alten passen
        # Simuliere neuronale Assoziation: Ältere emotionale Zustände klingen ab
        echo_vector = np.zeros(7)
        weight_sum = 0.0

        for idx, exp in enumerate(reversed(self.episodic_buffer)):
            # Zeitliche Abschwächung (je älter die Erinnerung, desto schwächer das Echo)
            time_factor = 1.0 / (idx + 1) * (1.0 - self.decay_rate)
            echo_vector += exp["vector"] * time_factor
            weight_sum += time_factor

        if weight_sum > 0:
            return echo_vector / weight_sum
        return np.zeros(7)
