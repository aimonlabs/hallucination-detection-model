import numpy as np
from IPython.display import display, HTML
from spacy import displacy

def render_predictions_with_scheme(tokenizer_or_text, text_or_spans, predictions_or_spans, show_scores=True, color_scheme="white-red", use_spans=False):
    """
    Render text with highlighted tokens or spans.
    
    Args:
        tokenizer_or_text: If use_spans=False, this is the tokenizer. If use_spans=True, this is the text to display.
        text_or_spans: If use_spans=False, this is the text. If use_spans=True, this is the spans with scores.
        predictions_or_spans: If use_spans=False, this is token predictions. If use_spans=True, this is ignored.
        show_scores: Whether to display scores
        color_scheme: Color scheme to use ("white-red" or "green-red")
        use_spans: Whether to use spans or tokens
    """
    
    template_with_labels = """
        <mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; text-shadow: 0.05em 0 white, 0 0.05em white, -0.05em 0 white, 0 -0.05em white, -0.05em -0.05em white, -0.05em 0.05em white, 0.05em -0.05em white, 0.05em 0.05em white;">
            {text}
            <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{label}{kb_link}</span>
        </mark>
        """

    template_without_labels = """
        <mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; text-shadow: 0.05em 0 white, 0 0.05em white, -0.05em 0 white, 0 -0.05em white, -0.05em -0.05em white, -0.05em 0.05em white, 0.05em -0.05em white, 0.05em 0.05em white;">
            {text}
        </mark>
        """
    
    # Convert to ents format
    ents = []
    
    if use_spans:
        # When using spans mode
        text = tokenizer_or_text
        spans_with_scores = text_or_spans
        
        # Convert spans_with_scores to ents
        for item in spans_with_scores:
            span, score = item[0], item[1]
            start, end = span
            
            # If the word itself is included, use it for verification
            word = item[2] if len(item) > 2 else text[start:end]
            
            ents.append({
                "start": start,
                "end": end,
                "label": f"{score:.2f}",
                "score": score
            })
    else:
        # Original token-based mode
        tokenizer = tokenizer_or_text
        text = text_or_spans
        predictions = predictions_or_spans
        
        # Tokenize input text
        tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = tokens["offset_mapping"]

        # Convert predictions to ents format
        for (start, end), score in zip(offsets, predictions):
            if start != end:  # Skip padding or special tokens
                ents.append({
                    "start": start,
                    "end": end,
                    "label": f"{score:.2f}",
                    "score": score
                })
    
    # Define colors based on scores and selected color scheme
    colors = {}
    for ent in ents:
        score = ent['score']
        label = f"{score:.2f}"
        
        if color_scheme == "white-red":
            # From white (255,255,255) to red (255,0,0)
            colors[label] = f"rgba(255, {255 - int(score * 255)}, {255 - int(score * 255)}, {0.2 + score * 0.8})"
        elif color_scheme == "green-red":
            # From green (0,255,0) to red (255,0,0)
            red = int(score * 255)
            green = int(255 - score * 255)
            colors[label] = f"rgba({red}, {green}, 0, {0.2 + score * 0.8})"
    
    if show_scores:
        options = {
            "ents": list(colors.keys()), 
            "colors": colors,
            "template": template_with_labels
        }
    else:
        options = {
            "ents": list(colors.keys()), 
            "colors": colors,
            "template": template_without_labels
        }
        
    # Create displacy-compatible dictionary
    displacy_input = {"text": text, "ents": ents}

    # Render with displacy
    displacy.render(displacy_input, style="ent", manual=True, jupyter=True, options=options)


def display_hallucination_results_words(result, show_scores=True, color_scheme="white-red"):
    """
    Display hallucination results using word-level spans from high_scoring_words.
    """
    # Get response text
    response_text = result['text']
    
    # Get high scoring words
    high_scoring_words = result['high_scoring_words']
    
    # Display title
    display(HTML("<h3>Hallucination Detection Results</h3>"))
    
    # Display word-level highlights
    display(HTML("<h4>High Scoring Words</h4>"))
    render_predictions_with_scheme(
        response_text, high_scoring_words, None,
        show_scores=show_scores, color_scheme=color_scheme, use_spans=True
    )
    
    # Display candidate sentences
    if result['candidate_sentences']:
        display(HTML("<h4>Candidate Sentences</h4>"))
        for i, sentence in enumerate(result['candidate_sentences']):
            ck_result = next((r for r in result['ck_results'] if r['text'] == sentence), None)
            confidence = ck_result['hallucination_probability'] if ck_result else 'N/A'
            
            # Format confidence properly
            if isinstance(confidence, float):
                confidence_display = f"{confidence:.4f}"
                color_intensity = min(int(confidence * 255), 255)
                prediction = "Hallucination" if ck_result['prediction'] == 1 else "Not Hallucination"
            else:
                confidence_display = str(confidence)
                color_intensity = 128
                prediction = "Unknown"
            
            display(HTML(
                f"<div style='background-color: rgba(255, {255-color_intensity}, {255-color_intensity}, 0.3); "
                f"padding: 10px; margin: 5px; border-radius: 5px;'>"
                f"<p style='margin: 0;'><b>Sentence {i+1}:</b> {sentence}</p>"
                f"<p style='margin: 0; font-size: 0.8em;'><b>Classification:</b> {prediction} (Confidence: {confidence_display})</p>"
                f"</div>"
            ))
    else:
        display(HTML("<p><b>No candidate sentences detected</b></p>"))
    
    # Display overall metrics
    display(HTML(
        f"<div style='margin-top: 15px;'>"
        f"<p><b>Hallucination Severity:</b> {result['hallucination_severity']:.4f}</p>"
        f"<p><b>Adjusted Hallucination Severity:</b> {result['adjusted_hallucination_severity']:.4f}</p>"
        f"</div>"
    ))

def display_hallucination_results(result, tokenizer, show_scores=True, color_scheme="white-red"):
    """
    Display the original and adjusted token probabilities for hallucination detection.
    """
    
    # Get response text
    response_text = result['text']
    
    # Get original and adjusted token probabilities
    original_probs = np.array(result['token_probabilities']['original'])
    adjusted_probs = np.array(result['token_probabilities']['adjusted'])
    
    # Display title
    display(HTML("<h3>Hallucination Detection Results</h3>"))
    
    # Display original token probabilities
    display(HTML("<h4>Original Token Probabilities</h4>"))
    render_predictions_with_scheme(
        tokenizer, response_text, original_probs, 
        show_scores=show_scores, color_scheme=color_scheme, use_spans=False
    )
    
    # Display adjusted token probabilities
    display(HTML("<h4>Adjusted Token Probabilities (influenced by sentence classifier)</h4>"))
    render_predictions_with_scheme(
        tokenizer, response_text, adjusted_probs, 
        show_scores=show_scores, color_scheme=color_scheme, use_spans=False
    )
    
    # Display hallucinated sentences
    if result['hallucinated_sentences']:
        display(HTML("<h4>Hallucinated Sentences</h4>"))
        for i, sentence in enumerate(result['hallucinated_sentences']):
            ck_result = result['ck_results'][i] if i < len(result['ck_results']) else None
            confidence = ck_result['hallucination_probability'] if ck_result else 'N/A'
            
            # Format confidence properly
            if isinstance(confidence, float):
                confidence_display = f"{confidence:.4f}"
                color_intensity = min(int(confidence * 255), 255)
            else:
                confidence_display = str(confidence)
                color_intensity = 128
            
            display(HTML(
                f"<div style='background-color: rgba(255, {255-color_intensity}, {255-color_intensity}, 0.3); "
                f"padding: 10px; margin: 5px; border-radius: 5px;'>"
                f"<p style='margin: 0;'><b>Sentence {i+1}:</b> {sentence}</p>"
                f"<p style='margin: 0; font-size: 0.8em;'><b>Confidence:</b> {confidence_display}</p>"
                f"</div>"
            ))
    else:
        display(HTML("<p><b>No hallucinated sentences detected</b></p>"))
    
    # Display overall metrics - Use seq_logits directly
    seq_result = result.get('seq_result', {})
    seq_logits = seq_result.get('logits', [0, 0])  # Default if not present
    seq_probability = seq_result.get('probabilities', [0, 0])[1]  # Get hallucination probability
    
    display(HTML(
        f"<div style='margin-top: 15px;'>"
        f"<p><b>Overall Hallucination (seq_logits):</b> {seq_logits[1]:.4f}</p>"
        f"<p><b>Hallucination Severity:</b> {result['hallucination_severity']:.4f}</p>"
        f"</div>"
    ))