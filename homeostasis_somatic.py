# homeostasis_somatic.py
# =====================================================================
# TECHNICAL DESIGN NOTE (MANUSCRIPT ALIGNMENT):
# The terminology used in this module (e.g., 'BiomechanicalHomeostasis')
# is utilized metaphorically to maintain structural alignment with the
# architectural concept described in the manuscript. Mechanistically,
# this class acts as a stateful, token-dependent sequence constraint
# tracker that modulates sampling hyper-parameters dynamically based 
# on rolling execution costs and contextual affect metrics.
# =====================================================================

import numpy as np

class BiomechanicalHomeostasis:
    def __init__(self, decay_rate=0.3):
        # Interne homöostatische Vitalwerte
        self.energy = 1.0
        self.metabolic_rate = 0.03  # Sanfterer Basisverbrauch
        self.decay_rate = decay_rate
        
        # Somatische Schlüsselwörter
        self.energy_restorers = ["sleep", "rest", "eat", "charge", "schlafen", "ausruhen", "essen", "warten", "calm", "safe"]
        self.stress_indicators = ["run", "fight", "panic", "angriff", "flucht", "schmerz", "danger", "bear", "lion"]

    def update_internal_state(self, num_tokens_generated):
        """Berechnet den passiven Energieverlust basierend auf der kognitiven Last."""
        # Sanfte Skalierung des Verbrauchs
        token_cost = num_tokens_generated * 0.002
        self.energy = max(0.05, self.energy - (self.metabolic_rate + token_cost))
        return self.energy

    def process_two_way_coupling(self, ai_output_text, stimmungs_vektor):
        """Das Denken des LLMs reguliert den Körper und dämpft aktiv die Panik-Stimmung!"""
        text_lower = ai_output_text.lower()
        
        rest_hits = sum(1 for word in self.energy_restorers if word in text_lower)
        stress_hits = sum(1 for word in self.stress_indicators if word in text_lower)
        
        if rest_hits > 0:
            self.energy = min(1.0, self.energy + (rest_hits * 0.20))
            # 🔴 WICHTIGER FIX: Entspannung senkt die Panik im Stimmungsvektor (Gefahr auf Index 3 dämpfen)
            stimmungs_vektor[3] = max(1.0, stimmungs_vektor[3] - (rest_hits * 0.40))
            print(f"🔄 SOMATISCHE RÜCKKOPPLUNG: KI-Verhalten regeneriert den Körper (+{rest_hits * 0.20:.2f} Energie, Panik gedämpft).")
        
        if stress_hits > 0:
            self.energy = max(0.05, self.energy - (stress_hits * 0.08))
            print(f"🔄 SOMATISCHE RÜCKKOPPLUNG: Physische Stressreaktion verbraucht Energie (-{stress_hits * 0.08:.2f} Energie).")
            
        return stimmungs_vektor

    def modulate_pal_vector(self, pal_raw_values, stimmungs_vektor):
        """Modifiziert den PAL-Vektor anhand des biologischen Energiemangels."""
        energy_deficit = 1.0 - self.energy
        
        # Energiemangel erhöht die Verwundbarkeit leicht (Index 3), blockiert aber nicht das gesamte System
        pal_raw_values[3] += energy_deficit * 0.15 
        
        final_values = pal_raw_values * stimmungs_vektor
        return np.clip(final_values, 0.0, 1.0)
