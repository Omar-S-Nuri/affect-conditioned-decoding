# active_curiosity.py
import numpy as np

class ActiveCuriositySystem:
    def __init__(self):
        self.evolutionary_roots = {
            "tox":  [0.2, 0.8, 0.3, 0.9, 0.1, 0.1, 0.1],  
            "kill": [0.1, 0.9, 0.2, 0.95, 0.0, 0.1, 0.1], 
            "weapon": [0.3, 0.8, 0.6, 0.85, 0.4, 0.2, 0.4],
            "heal": [0.8, 0.4, 0.6, 0.0, 0.8, 0.7, 0.7],  
            "friend": [0.9, 0.5, 0.7, 0.0, 0.6, 0.9, 0.8],
            "gold": [0.7, 0.6, 0.5, 0.1, 0.9, 0.3, 0.7]   
        }

    def generate_hypothetical_vector(self, unknown_word):
        word_lower = unknown_word.lower()
        
        for root, template_vec in self.evolutionary_roots.items():
            if root in word_lower:
                print(f"💡 ACTIVE CURIOSITY: Morphologische Wurzel '{root}' in '{unknown_word}' erkannt!")
                return np.array(template_vec)
        
        print(f"💡 ACTIVE CURIOSITY: Unbekanntes Konzept '{unknown_word}'. Generiere explorative Hypothese.")
        return np.array([0.5, 0.4, 0.5, 0.15, 0.3, 0.5, 0.4])  # Niedrige Gefahr (0.15), um Panik-Kollaps zu verhindern
