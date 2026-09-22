# Affect-Conditioned Decoding: Coupling a Lightweight Emotion Estimator to Sampling Parameters in Large Language Models

This repository contains the official framework for **Affect-Conditioned Decoding**, a method that dynamic-adaptively modulates Large Language Model (LLM) sampling trajectories based on low-cost prompt affect vectors. 

Rather than relying on fixed inference parameters, this framework maps raw text inputs into a 7-dimensional affect space (Valence, Arousal, Dominance, Danger, Resource Value, Social Bond, and Goal Proximity) using a lightweight ridge regression matrix. The resulting vectors alter downstream generation constraints (τ, top_k) in real-time.

---

## ⚠️ Important Scientific Limitation & Scaling Disclosure

**Localized Mode Collapse in Sub-Billion Regimes:**  
Empirical evaluation benchmarks on parameter-constrained architectures (specifically **Qwen1.5-0.5B-Chat**) reveal a critical operational threshold. Under acute contextual threat or resource exhaustion, forcing the model into strict sampling restrictions (τ=0.25, top_k=15) triggers a localized **Mode Collapse**. 

The model pseudo-defensively falls into degenerative autoregressive repetitions of standard phrases. Consequently, downstream evaluation shows an artificially deflated cross-entropy perplexity (PPL ~3.0). This framework exposes this phenomenon not as a sign of clean cognitive alignment, but as a **sampling artifact** driven by a compressed token search space. 

To mitigate this, our framework injects the high-density 7D affect vector prefix, which acts as a semantic context anchor, significantly stabilizing vocabulary diversity and steering trajectories toward contextually appropriate behaviors compared to pure hyperparameter throttling.

---

## 📊 Scientific Ablation Architecture

```text
       [Input Text] ➔ [Active Curiosity / 7D Vectorizer]
                             │
                             ▼
     [Rolling Cache] ➔ [Asynchronous Affective Latency] ➔ [+]
                                                           │
                                                           ▼
     [Resource Tracking] ➔ [Metabolic State] ➔ [*] ➔ [Affect Cascades]
                                                          │
                                                          ▼
                     ┌────────────────────────────────────┴────────────────────────────────────┐
                     ▼ (Danger > 0.60 OR Energy < 0.25)                                        ▼ (Optimal State)
         [Restricted Sampling Path]                                                [Creative Base Path]
         [Temp: 0.25 | Top_K: 15]                                                  [Temp: 0.75 | Top_K: 50]
                     │                                                                         │
                     └────────────────────────────────────┬────────────────────────────────────┘
                                                          ▼
                                            [Sub-Billion Qwen Neocortex]
                                             (Subject to localized loop risks)
                                                          │
                                                          ▼
                                              [Affect-Conditioned Output]
                                                          │
                                                          ▼
                                              [Two-Way Somatic Feedback]
```

---

## 🛠️ Repository & Installation

### Project File Structure
```text
📁 affect-conditioned-decoding/
├── Paper/
│   ├── main.tex                      # Complete LaTeX Draft (English/German versions available)
│   └── Affect_Conditioned_Decoding.pdf # Fully compiled scientific draft
├── .env                              # Private API configuration (Optional Cloud Migration)
├── .gitignore                        # Prevents tracking of model weights, cache, and keys
├── LICENSE                           # MIT License
├── README.md                         # Documentation
├── active_curiosity.py               # Handles unseen morphology via character fallback
├── chat_with_pal_final.py            # Live interactive loop with dynamic metabolic costs
├── evaluate_pal.py                   # Rigorous 4-Way Semantic Specificity Ablation Matrix
├── expand_lexicon.py                 # Local vector-expansion pipeline via Ollama (Llama 3.1)
├── finetune_llm_verhaltensvielfalt.py# Fine-tunes the Qwen cortical weight distributions
├── hippocampus_memory.py             # Decay-based affective latency buffering module
├── homeostasis_somatic.py            # Computational resource depletion matrix
├── merge_lexicons.py                 # Merges raw psycholinguistic source datasets
├── requirements.txt                  # System dependencies
├── test_model.py                     # Static single-stimulus diagnostic validator
└── train_pal.py                      # Multi-output Ridge model matrix fitter
```

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd affect-conditioned-decoding
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ./venv/Scripts/activate   # On Windows
   pip install -r requirements.txt
   ```

3. **Initialize Local Ollama Node:**
   Ensure [Ollama](https://ollama.com) is installed and active on your local machine, then pull the target dataset expansion model:
   ```bash
   ollama pull llama3.1:latest
   ```

---

## 🗺️ Execution Pipeline

To reproduce the experiments and initialize the full decoding framework fully locally, execute the modules in this exact sequence:

```bash
# 1. Standardize and merge raw psycholinguistic lexicon sheets
python merge_lexicons.py

# 2. Expand multi-dimensional vector coverage 100% locally via Ollama Node
python expand_lexicon.py

# 3. Train the lightweight 7D estimator matrix (PAL) using nrc_final_evolutionary.txt
python train_pal.py

# 4. Fine-tune the cortical language weights of the Neocortex (Optional/Reproducibility)
python finetune_llm_verhaltensvielfalt.py

# 5. Run single-stimulus diagnostic tests (Optional verification)
python test_model.py

# 6. Launch the live interactive system with metabolic feedback loops
python chat_with_pal_final.py

# 7. Run the 4-way ablation evaluation suite to extract statistical metrics
python evaluate_pal.py
```

---

## 🧪 4-Way Semantic Specificity Ablation Protocol

To eliminate mathematical sampling artifacts and isolate the true causal effect of the 7D affect vector, `evaluate_pal.py` automatically benchmarks **four distinct experimental paths** using **Target-Only Perplexity** (masking out prompt tokens via `ignore_index=-100`) and automated **Semantic Embedding Similarity** via `all-MiniLM-L6-v2`:

1. **Condition 1 (Absolute Control):** Baseline model with default sampling parameters (τ=0.75, k=50) without prefix strings.
2. **Condition 2 (Noise Padding Control):** Baseline model with default parameters prepended by a meaningless, affect-free placeholder prefix matching the exact character length of the 7D vector (generated utilizing high-frequency neutral filler tokens).
3. **Condition 3 (Hyperparameter Baseline):** Baseline model running on tight sampling restrictions (τ=0.25, k=15) without vector context.
4. **Condition 4 (Full Framework):** Full system combining both the structural 7D affect vector prefix and dynamic sampling constraints concurrently.

Statistical validation is automatically backed up via non-parametric **Mann-Whitney-U ranking tests** to verify directional p-value significance (\(p < 0.05\)).

---

## 📝 License
This project is open-source and licensed under the Open-Source MIT License - see the [LICENSE](LICENSE) file for details.
