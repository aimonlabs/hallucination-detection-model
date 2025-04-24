import numpy as np
from IPython.display import display, HTML
from spacy import displacy

def render_predictions_with_scheme(tokenizer_or_text, text_or_spans, predictions_or_spans, show_scores=True, color_scheme="white-red", use_spans=True, class_info=None):
    """
    Render text with highlighted tokens or spans.
    
    Args:
        tokenizer_or_text: If use_spans=False, this is the tokenizer. If use_spans=True, this is the text to display.
        text_or_spans: If use_spans=False, this is the text. If use_spans=True, this is the spans with scores.
        predictions_or_spans: If use_spans=False, this is token predictions. If use_spans=True, this is ignored.
        show_scores: Whether to display scores
        color_scheme: Color scheme to use ("white-red", "green-red", "blue-red")
        use_spans: Whether to use spans or tokens
        class_info: Optional dict mapping span indices to class (0 or 1)
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
        for i, item in enumerate(spans_with_scores):
            span, score = item[0], item[1]
            start, end = span
            
            # If the word itself is included, use it for verification
            word = item[2] if len(item) > 2 else text[start:end]
            
            # Determine class if class_info is provided
            entity_class = class_info.get(i, 1) if class_info else 1
            
            ents.append({
                "start": start,
                "end": end,
                "label": f"{score:.2f}",
                "score": score,
                "class": entity_class
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
        for i, ((start, end), score) in enumerate(zip(offsets, predictions)):
            if start != end:  # Skip padding or special tokens
                # Determine class if class_info is provided
                entity_class = class_info.get(i, 1) if class_info else 1
                
                ents.append({
                    "start": start,
                    "end": end,
                    "label": f"{score:.2f}",
                    "score": score,
                    "class": entity_class
                })
    
    # Define colors based on scores and selected color scheme
    colors = {}
    for ent in ents:
        score = ent['score']
        label = f"{score:.2f}"
        
        # Choose color based on class and color scheme
        entity_class = ent.get('class', 1)  # Default to class 1 if not specified
        
        if color_scheme == "white-red":
            # Class 0: White to blue
            if entity_class == 0:
                colors[label] = f"rgba({255 - int(score * 100)}, {255 - int(score * 100)}, 255, {0.2 + score * 0.6})"
            # Class 1: White to red
            else:
                colors[label] = f"rgba(255, {255 - int(score * 150)}, {255 - int(score * 150)}, {0.2 + score * 0.6})"
        
        elif color_scheme == "blue-red":
            # Class 0: Blue 
            if entity_class == 0:
                colors[label] = f"rgba(100, 150, 255, {0.3 + score * 0.5})"
            # Class 1: Red
            else:
                colors[label] = f"rgba(255, 100, 100, {0.3 + score * 0.5})"
        
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

def display_hallucination_results_words(result, show_scores=True, color_scheme="white-red", separate_classes=False):
    """
    Display hallucination results using word-level spans from high_scoring_words.
    
    Args:
        result: Result from hallucination detection
        show_scores: Whether to show scores
        color_scheme: Color scheme for highlighting
        separate_classes: Whether to use different colors for class 0 vs class 1
    """
    # Get response text
    response_text = result['text']
    
    # Get high scoring words
    high_scoring_words = result['high_scoring_words']
    
    # Create a mapping from words to sentence classes if separate_classes is True
# Create a mapping from words to sentence classes if separate_classes is True
    word_to_class = {}
    if separate_classes and 'candidate_sentences' in result and 'ck_results' in result:
        # Create direct mapping from sentence text to classification
        sentence_class = {}
        for idx, (sentence, ck_result) in enumerate(zip(result['candidate_sentences'], result['ck_results'])):
            sentence_class[sentence] = ck_result['prediction']
        
        # Find sentence for each word
        for i, item in enumerate(high_scoring_words):
            span = item[0]
            word_pos = span[0]
            
            # Find which sentence contains this word
            for sentence, cls in sentence_class.items():
                sent_start = response_text.find(sentence)
                if sent_start <= word_pos < sent_start + len(sentence):
                    word_to_class[i] = cls
                    break
    
    # Display title
    display(HTML("<h3>Hallucination Detection Results</h3>"))
    
    # Display word-level highlights
    display(HTML("<h4>High Scoring Words</h4>"))
    render_predictions_with_scheme(
        response_text, high_scoring_words, None,
        show_scores=show_scores, 
        color_scheme=color_scheme, 
        use_spans=True,
        class_info=word_to_class if separate_classes else None
    )
    
    # Display candidate sentences with updated colors
    if result['candidate_sentences']:
        display(HTML("<h4>Candidate Sentences</h4>"))
        for i, sentence in enumerate(result['candidate_sentences']):
            ck_result = next((r for r in result['ck_results'] if r['text'] == sentence), None)
            
            if ck_result:
                confidence = ck_result['hallucination_probability']
                prediction_class = ck_result['prediction']
                
                # Determine color based on class and confidence
                if separate_classes and prediction_class == 0:
                    # Blue gradient for class 0
                    bg_color = f"rgba(100, 150, 255, {0.3 + confidence * 0.4})"
                    prediction_label = "Common Knowledge"
                else:
                    # Red gradient for class 1
                    bg_color = f"rgba(255, 100, 100, {0.3 + confidence * 0.4})"
                    prediction_label = "Hallucination"
                    
                confidence_display = f"{confidence:.4f}"
            else:
                confidence_display = "N/A"
                bg_color = "rgba(200, 200, 200, 0.3)"
                prediction_label = "Unknown"
            
            # Display with updated styling
            display(HTML(
                f"<div style='background-color: {bg_color}; "
                f"padding: 10px; margin: 5px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.1);'>"
                f"<p style='margin: 0;'><b>Sentence {i+1}:</b> {sentence}</p>"
                f"<p style='margin: 0; font-size: 0.8em;'><b>Classification:</b> {prediction_label} (Confidence: {confidence_display})</p>"
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