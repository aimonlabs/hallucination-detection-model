def get_candidate_sentence_stats(sentences_data, candidate_indices, token_scores, threshold=0.5):
    """
    Get candidate sentences with their sentence IDs and high-scoring token metrics.
    
    Args:
        sentences_data: Dictionary with sentence data
        candidate_indices: Indices of candidate sentences
        token_scores: Array of token scores
        threshold: Score threshold for high-scoring tokens
        
    Returns:
        List of tuples: [(sentence_id, sentence_text, high_score_fraction), ...]
    """
    results = []
    
    for idx in candidate_indices:
        sentence = sentences_data["sentences"][idx]
        token_indices = sentences_data["token_indices"][idx]
        
        # Skip sentences with no tokens
        if not token_indices:
            continue
            
        # Get scores for tokens
        scores = [token_scores[i] if i < len(token_scores) else 0 for i in token_indices]
        high_scores = [score for score in scores if score > threshold]
        
        # Calculate fraction of high-scoring tokens
        high_score_fraction = len(high_scores) / len(scores) if scores else 0
        
        # Add to results (sentence_id is the index in the original list)
        results.append((idx, sentence, high_score_fraction))
    
    return results

def analyze_token_scores(sentences_data, candidate_indices, token_scores, threshold=0.5):
    """
    Utility function to analyze token scores for candidate sentences.
    Can be used by both debug and regular functions.
    
    Args:
        sentences_data: Dictionary with sentence data
        candidate_indices: Indices of candidate sentences
        token_scores: Array of token scores
        threshold: Score threshold
        
    Returns:
        Dictionary with analysis for each candidate sentence
    """
    results = {}
    
    for idx in candidate_indices:
        sentence = sentences_data["sentences"][idx]
        token_indices = sentences_data["token_indices"][idx]
        
        # Skip sentences with no tokens
        if not token_indices:
            results[idx] = {
                "sentence": sentence,
                "token_indices": [],
                "scores": [],
                "high_scores": [],
                "high_score_fraction": 0
            }
            continue
            
        # Get scores for tokens
        scores = [token_scores[i] if i < len(token_scores) else 0 for i in token_indices]
        high_scores = [score for score in scores if score > threshold]
        
        # Calculate fraction of high-scoring tokens
        high_score_fraction = len(high_scores) / len(scores) if scores else 0
        
        # Store analysis
        results[idx] = {
            "sentence": sentence,
            "token_indices": token_indices,
            "scores": scores,
            "high_scores": high_scores,
            "high_score_fraction": high_score_fraction
        }
    
    return results

def get_candidate_sentences_with_stats(sentences_data, candidate_indices, token_scores, threshold=0.5):
    """
    Get candidate sentences with their IDs and high-scoring token metrics.
    Uses the common analysis utility function.
    
    Args:
        sentences_data: Dictionary with sentence data
        candidate_indices: Indices of candidate sentences
        token_scores: Array of token scores
        threshold: Score threshold
        
    Returns:
        List of tuples: [(sentence_id, sentence_text, high_score_fraction), ...]
    """
    # Get analysis data
    analysis = analyze_token_scores(sentences_data, candidate_indices, token_scores, threshold)
    
    # Format for return
    results = []
    for idx in candidate_indices:
        data = analysis[idx]
        results.append((idx, data["sentence"], data["high_score_fraction"]))
    
    return results

def adjust_token_scores(token_scores, sentences_data, candidate_indices, ck_results):
    """
    Adjust token scores based on classification results.
    
    Args:
        token_scores: Token scores array
        sentences_data: Dictionary with sentence data
        candidate_indices: Indices of candidate sentences
        ck_results: Classification results
        
    Returns:
        Adjusted token scores
    """
    adjusted_scores = token_scores.copy()
    
    for i, (candidate_idx, ck_result) in enumerate(zip(candidate_indices, ck_results)):
        # Only adjust if prediction is non-hallucinated (class 0)
        if ck_result["prediction"] == 0:
            non_hall_prob = ck_result["non_hallucination_probability"]
            token_indices = sentences_data["token_indices"][candidate_idx]
            
            # Vectorize
            adjusted_scores[token_indices] = token_scores[token_indices] * non_hall_prob
    
    return adjusted_scores

def get_high_scoring_word_boundaries(token_scores, tokenized_input, 
                                     prompt_len, response, 
                                     threshold=0.5, include_word=True):
    """
    Get word boundaries for high-scoring tokens using ONLY tokenizer word_ids.
    
    Args:
        token_scores: Array of token scores (response tokens only)
        tokenized_input: Already tokenized input (context + response)
        prompt_len: Length of prompt in tokens
        response: Response text
        threshold: Score threshold
        include_word: Include word text in output
    """
    # Get word IDs and offsets
    word_ids = tokenized_input.word_ids()
    offset_mapping = tokenized_input['offset_mapping'][0].cpu().numpy()
    sequence_ids = tokenized_input.sequence_ids(0)
    
    # Map response tokens to their word IDs and original score indices
    response_words = {}  # word_id -> (char_start, char_end, [token indices])
    
    for i, (seq_id, word_id) in enumerate(zip(sequence_ids, word_ids)):
        # Only process response tokens
        if seq_id != 1 or word_id is None:
            continue
            
        # Get token index in token_scores array
        score_idx = i - prompt_len
        
        # Get character offsets for this token
        char_start, char_end = offset_mapping[i]
        
        # Add to or update word entry
        if word_id not in response_words:
            response_words[word_id] = [char_start, char_end, [score_idx]]
        else:
            # Update character range if needed
            response_words[word_id][0] = min(response_words[word_id][0], char_start)
            response_words[word_id][1] = max(response_words[word_id][1], char_end)
            response_words[word_id][2].append(score_idx)
    
    # Find high-scoring words
    high_scoring_words = []
    
    for word_id, (start, end, token_indices) in response_words.items():
        # Get scores for all tokens in this word
        scores = [token_scores[idx] for idx in token_indices if idx < len(token_scores)]
        
        # Check if any score is above threshold
        high_scores = [s for s in scores if s > threshold]
        
        if high_scores:
            # Get highest score
            max_score = max(high_scores)
            
            # Get the word text
            word_text = response[start:end]
            
            if include_word:
                high_scoring_words.append(([start, end], max_score, word_text))
            else:
                high_scoring_words.append(([start, end], max_score))
    
    # Sort by position
    high_scoring_words.sort(key=lambda x: x[0][0])
    
    return high_scoring_words
