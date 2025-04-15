import torch
import numpy as np
from hdm2.data.process_text import extract_sentences, prepare_single_input, add_period_if_missing, get_sentence_token_boundaries, prepare_sentences, map_tokens_to_sentences
from hdm2.utils.debug_utils import debug_token_mapping, debug_candidate_sentences
from hdm2.utils.process_output import get_candidate_sentences_with_stats, get_high_scoring_word_boundaries, adjust_token_scores

def get_last_token_embeddings(hidden_states, attention_mask, padding='right'):
    """
    Get embeddings for the last token in each sequence.
    
    Args:
        hidden_states: Hidden states from the model
        attention_mask: Attention mask tensor
        padding: Padding side ('right' or 'left')
        
    Returns:
        Tensor with last token embeddings
    """
    batch_size = hidden_states.size(0)
    
    if padding == 'right':
        # For right padding, the last token is the one before padding starts
        sequence_lengths = attention_mask.sum(dim=1) - 1
    else:
        # For left padding, the last token is always the last position
        sequence_lengths = torch.tensor([hidden_states.size(1) - 1] * batch_size, device=hidden_states.device)
    
    # Gather the last hidden states
    last_hidden = torch.stack([hidden_states[i, sequence_lengths[i], :] for i in range(batch_size)])
    
    return last_hidden

def get_token_predictions(token_logits, threshold=0.9):
    probs = torch.softmax(token_logits, dim=-1)
    class_1_probs = probs[:, :, 1]
    predictions = (class_1_probs >= threshold).long()
    return predictions

def check_hdm2(prompt, context, response, token_model, ck_classifier, tokenizer):
    combined_context = prompt + " " + context
    ip_tokens = tokenizer(combined_context, response, return_tensors="pt").to(token_model.backbone.device)
    
    with torch.no_grad():
        op = token_model(**ip_tokens)
    
    prompt_len = ip_tokens.sequence_ids(0).index(1)
    token_scores = torch.softmax(op['token_logits'][0], axis=1)[prompt_len:, 1].cpu().numpy()
    #seq_scores = torch.softmax(op['seq_logits'][0], axis=1)[prompt_len:, 1].cpu().numpy()
    
    # Get token predictions for the response part only
    response_logits = op['token_logits'][:, prompt_len:]
    token_preds = get_token_predictions(response_logits, threshold=0.9)
    
    batch, model_inputs = prepare_single_input(combined_context, response, tokenizer)

    # Create full predictions tensor with zeros for context tokens
    full_preds = []
    for i in range(len(token_preds)):
        # Create a list of zeros for the full sequence
        full_pred = [0] * len(ip_tokens['input_ids'][i])
        # Place the response predictions at the correct positions
        for j, pred in enumerate(token_preds[i]):
            full_pred[prompt_len + j] = pred.item()
        full_preds.append(full_pred)

    # Extract hallucinated sentences
    hallucinated_sentences = extract_sentences(batch, full_preds, is_single_data=(len(token_preds)==1))
    hallucinated_sentences = [add_period_if_missing(x) for x in hallucinated_sentences]
    
    #print(f"Hallucinated sentences: {hallucinated_sentences}")
    ck_results = classify_sentences(token_model, ck_classifier, tokenizer, hallucinated_sentences, layer_id=25)
        
    return token_scores, hallucinated_sentences, ck_results

def classify_sentences(token_model, classifier, tokenizer, texts_or_indices, 
                      use_last_tokens=False, use_truncated_context=False,
                      tokenized_inputs=None, sentences=None, context="",
                      response="", candidate_indices=None, 
                      layer_id=25, device='cuda'):
    """
    Classify sentences with three possible approaches:
    1. Individual sentences (use_last_tokens=False, use_truncated_context=False)
    2. Last token in context+response (use_last_tokens=True)
    3. Context + truncated response (use_truncated_context=True)
    
    Args:
        token_model: Base model
        classifier: Classifier head
        tokenizer: Tokenizer
        texts_or_indices: List of texts or token indices
        use_last_tokens: Whether to use last token embeddings from full context+response
        use_truncated_context: Whether to use context+truncated_response for each sentence
        tokenized_inputs: Pre-tokenized inputs (required if use_last_tokens=True)
        sentences: Original sentences (for better output)
        context: Context text (required if use_truncated_context=True)
        response: Response text (required if use_truncated_context=True)
        candidate_indices: Indices of candidate sentences (required if use_truncated_context=True)
        layer_id: Layer to extract embeddings from
        device: Device to run on
    """
    if not texts_or_indices and not use_truncated_context:
        return []
    
    # Check incompatible options
    if use_last_tokens and use_truncated_context:
        raise ValueError("Cannot use both use_last_tokens and use_truncated_context")
    
    base_model = token_model.backbone.base_model.model
    padding_side = getattr(tokenizer, 'padding_side', 'right')
    
    with torch.no_grad():
        if not use_last_tokens and not use_truncated_context:
            # Approach 1: Original - process each sentence separately
            inputs = tokenizer(
                texts_or_indices,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            outputs = base_model(**inputs, output_hidden_states=True)
            
            last_hidden = get_last_token_embeddings(
                outputs.hidden_states[layer_id], 
                inputs['attention_mask'],
                padding=padding_side
            )
            
        elif use_last_tokens:
            # Approach 2: Use token positions from full context+response
            if tokenized_inputs is None:
                raise ValueError("tokenized_inputs must be provided when use_last_tokens=True")
            
            outputs = base_model(**tokenized_inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[layer_id]
            token_positions = texts_or_indices
            
            last_hidden = torch.stack([
                hidden_states[0, position, :] 
                for position in token_positions
            ])
            
        else:
            # Approach 3: Use context + truncated response for each sentence
            if context is None or response is None or sentences is None or candidate_indices is None:
                raise ValueError("context, response, sentences, and candidate_indices must be provided when use_truncated_context=True")
            
            # Prepare truncated versions of the response
            truncated_responses = []
            for idx in candidate_indices:
                # Get the current sentence
                current_sentence = sentences[idx]

                # Find where this sentence ends in the original response
                sentence_pos = response.find(current_sentence)
                if sentence_pos == -1:
                    # Fallback if exact match not found
                    included_sentences = sentences[:idx+1]
                    last_included = included_sentences[-1]
                    last_pos = response.find(last_included)
                    sentence_end_pos = last_pos + len(last_included)
                else:
                    sentence_end_pos = sentence_pos + len(current_sentence)

                # Slice the response up to this position
                truncated_response = response[:sentence_end_pos]
                truncated_responses.append(truncated_response)

            print(truncated_responses)
            
            # Tokenize in batch
            context_list = [context] * len(truncated_responses)
            inputs = tokenizer(
                context_list, 
                truncated_responses,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            outputs = base_model(**inputs, output_hidden_states=True)

            # Extract the last token embeddings for each sequence
            last_hidden = get_last_token_embeddings(
                outputs.hidden_states[layer_id],
                inputs['attention_mask'],
                padding=padding_side
            )
        
        # Classify using extracted embeddings
        classifier_outputs = classifier(last_hidden)
        logits = classifier_outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        # Prepare results
        results = []
        for i, pred in enumerate(preds):
            if not use_last_tokens and not use_truncated_context:
                # Original approach - use provided texts
                text = texts_or_indices[i]
            elif use_truncated_context:
                # Truncated approach - use original sentence
                idx = candidate_indices[i]
                text = sentences[idx] if sentences is not None else f"Sentence at index {idx}"
            else:
                # Last token approach - use original sentence if available
                text = sentences[i] if sentences is not None else f"Token at position {texts_or_indices[i]}"
            
            results.append({
                'text': text,
                'prediction': pred.item(),
                'hallucination_probability': probs[i, 1].item(),
                'non_hallucination_probability': probs[i, 0].item()
            })
        
        return results
    
def find_candidate_sentences(sentences_data, token_scores, threshold):
    """
    Find sentences with token scores above threshold.
    
    Args:
        sentences_data: Dictionary with sentence data
        token_scores: Array of token scores
        threshold: Score threshold
        
    Returns:
        List of sentence indices
    """
    candidate_indices = []
    
    for i, token_indices in enumerate(sentences_data["token_indices"]):
        # Check if any token in this sentence has a score above threshold
        if any(token_scores[idx] > threshold for idx in token_indices if idx < len(token_scores)):
            candidate_indices.append(i)
    
    return candidate_indices

def _detect_hallucinations(prompt, context, response, 
                         token_model, ck_classifier, 
                         tokenizer,
                         ck_layer_ix = 25,
                         token_threshold=0.5, 
                         ck_threshold=0.9,
                         use_last_tokens=False, 
                         use_truncated_context=False, 
                         debug=False,
                         is_include_spans = False,
                         device='cuda',
                         ):
    """
    Detect hallucinations using token-level and sentence-level classifiers.
    
    Args:
        prompt: Prompt text
        context: Context text
        response: Response text
        token_model: Token-level model
        ck_classifier: Sentence-level classifier
        tokenizer: Tokenizer
        ck_layer_ix: Layer index for extracting embeddings
        token_threshold: Threshold for token scores
        use_last_tokens: Whether to use last token embeddings
        use_truncated_context: Whether to use context+truncated_response for each sentence
        debug: Enable debug output
        is_include_spans: Whether to include word spans in results
        return_json: Whether to return result as JSON string

    Returns:
        Result dictionary with hallucination detection information
    """
    # tokenization step
    # Important to get offsets (for getting word boundaries)
    # and not move tokens to GPU at this point
    combined_context = prompt + " " + context
    tokens = tokenizer(combined_context, response, return_tensors="pt",
                      return_offsets_mapping=True,)
    
    # Get the boundary between context and response
    prompt_len = tokens.sequence_ids(0).index(1)
    
    # Separate model inputs from mapping data
    model_inputs = {k: v for k, v in tokens.items() if k != 'offset_mapping'}

    # determine the model's dtype
    model_dtype = next(token_model.parameters()).dtype

    # Only convert attention_mask to float dtype, keep input_ids as LongTensor
    # Also, don't use offset_mapping in this step
    model_inputs = {}
    for k, v in tokens.items():
        if k != 'offset_mapping':
            if k == 'input_ids':
                model_inputs[k] = v.to(token_model.backbone.device)
            else:
                model_inputs[k] = v.to(token_model.backbone.device, dtype=model_dtype)

    # Get token scores using original approach
    with torch.no_grad():
        op = token_model(**model_inputs)
    
    # Use original logic for token scores and sequence probabilities
    token_scores = torch.softmax(op['token_logits'][0], axis=1)[prompt_len:, 1].float().cpu().numpy()
    seq_probs = torch.softmax(op['seq_logits'], dim=-1)[0].float().cpu().numpy()
    
    # Extract sentences
    sentences_data = prepare_sentences(response)
    
    # Map tokens to sentences
    sentences_data["token_indices"] = map_tokens_to_sentences(
        sentences_data["sentences"], 
        response, 
        tokenizer,
        tokens, 
        prompt_len
    )
    
    # Debug token mapping if requested
    if debug:
        debug_token_mapping(sentences_data, tokenizer, tokens["input_ids"][0][prompt_len:])
    
    # Get sentence boundaries
    sentence_boundaries = get_sentence_token_boundaries(sentences_data)
    
    # Find candidate sentences
    candidate_indices = find_candidate_sentences(sentences_data, token_scores, token_threshold)
    candidate_sentences = [sentences_data["sentences"][idx] for idx in candidate_indices]
    
    # Get candidate sentence stats using the new function
    candidate_sentence_stats = get_candidate_sentences_with_stats(
        sentences_data, 
        candidate_indices, 
        token_scores, 
        token_threshold
    )
    
    # Debug candidate sentences if requested
    if debug:
        debug_candidate_sentences(
            sentences_data, 
            candidate_indices, 
            token_scores, 
            tokenizer, 
            tokens["input_ids"][0][prompt_len:]
        )
    
    adjusted_hallucination_severity = float(seq_probs[1])

    if candidate_indices:
        if use_truncated_context:
            # Approach 3: Use context + truncated response for each sentence
            ck_results = classify_sentences(
                token_model,
                ck_classifier,
                tokenizer,
                None,  # Not used in this approach
                use_last_tokens=False,
                use_truncated_context=True,
                context=combined_context,
                response=response,
                sentences=sentences_data["sentences"],
                candidate_indices=candidate_indices,
                layer_id=ck_layer_ix,
                device=device,
            )
        elif use_last_tokens:
            # Approach 2: Use last token embeddings from full context+response
            candidate_boundaries = [sentence_boundaries[idx] for idx in candidate_indices]
            last_token_positions = [prompt_len + boundary[1] for boundary in candidate_boundaries]
            
            ck_results = classify_sentences(
                token_model,
                ck_classifier,
                tokenizer,
                last_token_positions,
                use_last_tokens=True,
                tokenized_inputs=tokens,
                sentences=candidate_sentences,
                layer_id=ck_layer_ix,
                device=device,
            )
        else:
            # Approach 1: Original - classify individual sentences
            ck_results = classify_sentences(
                token_model,
                ck_classifier,
                tokenizer,
                candidate_sentences,
                use_last_tokens=False,
                layer_id=ck_layer_ix,
                device=device,
            )
        
        # Apply CK threshold to override predictions
        for result in ck_results:
            # Only consider it a hallucination if the probability exceeds the threshold
            if result['hallucination_probability'] < ck_threshold:
                result['prediction'] = 0  # Force non-hallucination if below threshold
        
        # Adjust token scores
        adjusted_scores = adjust_token_scores(
            token_scores,
            sentences_data,
            candidate_indices,
            ck_results
        )

        # Calculate adjusted hallucination severity
        all_non_hallucinated = all(result['prediction'] == 0 for result in ck_results)
        if all_non_hallucinated and ck_results:
            # Calculate average hallucination probability from CK results
            avg_prob = sum(result['hallucination_probability'] for result in ck_results) / len(ck_results)
            adjusted_hallucination_severity = float(avg_prob)
    else:
        adjusted_scores = token_scores
        ck_results = []
        

    
    # Create result with all requested information
    result = {
        "text": response,
        "token_scores": adjusted_scores.tolist(),
        "hallucination_detected": bool(np.any(adjusted_scores > token_threshold)),
        "ck_results": ck_results,  # Include classification results
        "candidate_sentences": candidate_sentences,  # The actual candidate sentences
        "candidate_sentence_stats": candidate_sentence_stats,  # New: sentence stats with IDs and scores
        "seq_logits": op['seq_logits'][0].cpu().tolist(),  # Sequence logits
        "seq_probs": seq_probs.tolist(),  # Softmaxed sequence probabilities
        "hallucination_severity": float(seq_probs[1]),  # Hallucination severity from seq score
        "adjusted_hallucination_severity": adjusted_hallucination_severity  # New field
    }
    
    # Add high-scoring words with boundaries
    high_scoring_words = get_high_scoring_word_boundaries(
        adjusted_scores,
        tokens,  # Pass the existing tokenized input
        prompt_len,
        response,
        threshold=token_threshold,
        include_word=is_include_spans
    )
    result["high_scoring_words"] = high_scoring_words
    
    return result

