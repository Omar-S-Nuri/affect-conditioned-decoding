## evaluate_pal.py

def calculate_perplexity_and_metrics(prompt, context_vector="", temp=0.75, top_k=50):
    full_prompt = f"{context_vector} REIZ: {prompt}. REAKTION:" if context_vector else f"REIZ: {prompt}. REAKTION:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 1. PPL berechnen (Überraschung des Modells)
        logits = model(**inputs).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(loss.item())
        
        # 2. Text generieren zur Messung der Repetition (Distinct-n)
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=temp, top_k=top_k)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reaction = generated_text.split("REAKTION:")[-1].strip()
        words = reaction.split()
        
        # Errechnung von Distinct-2 (Einzigartige Wort-Paare / Gesamte Wort-Paare)
        # Wenn dieser Wert gegen 0 sinkt, hast du einen repetitiven Kollaps!
        if len(words) > 1:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            distinct_2 = len(set(bigrams)) / len(bigrams)
        else:
            distinct_2 = 1.0
            
    return perplexity, len(words), distinct_2
