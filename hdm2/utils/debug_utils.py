# Debug functions
from hdm2.utils.process_output import analyze_token_scores

def debug_token_mapping(sentences_data, tokenizer, response_tokens):
    """
    Debug function to check if tokens are correctly mapped to sentences.
    
    Args:
        sentences_data: Dictionary with sentence data
        tokenizer: Tokenizer for decoding tokens
        response_tokens: Token IDs for the response
    """
    print("\n=== TOKEN MAPPING DEBUG ===")
    
    # Check if token_indices exists
    if 'token_indices' not in sentences_data:
        print("No token indices found in sentences_data")
        return
    
    # For each sentence
    for i, (sentence, token_indices) in enumerate(zip(sentences_data['sentences'], sentences_data['token_indices'])):
        print(f"\nSentence {i}: \"{sentence}\"")
        print(f"Token indices: {token_indices}")
        
        # Extract and decode tokens
        if token_indices:
            # Get the tokens for this sentence
            sentence_token_ids = [response_tokens[idx].item() if idx < len(response_tokens) else 0 
                                 for idx in token_indices]
            
            # Decode them
            decoded_tokens = tokenizer.decode(sentence_token_ids)
            print(f"Decoded tokens: \"{decoded_tokens}\"")
            
            # Check if the decoded tokens match the sentence
            if decoded_tokens.strip() == sentence.strip():
                print("✓ Tokens match sentence")
            else:
                print("⚠ Warning: Tokens don't match sentence exactly")
                
                # Show individual tokens
                individual_tokens = [tokenizer.decode([response_tokens[idx].item()]) for idx in token_indices if idx < len(response_tokens)]
                print(f"Individual tokens: {individual_tokens}")
        else:
            print("No tokens mapped to this sentence!")

def debug_candidate_sentences(sentences_data, candidate_indices, token_scores, tokenizer, response_tokens, threshold=0.5):
    """
    Debug function to check candidate sentences and their token scores.
    Uses the common analysis utility function.
    
    Args:
        sentences_data: Dictionary with sentence data
        candidate_indices: Indices of candidate sentences
        token_scores: Array of token scores
        tokenizer: Tokenizer for decoding tokens
        response_tokens: Token IDs for the response
        threshold: Score threshold
    """
    print("\n=== CANDIDATE SENTENCES DEBUG ===")
    print(f"Number of candidate sentences: {len(candidate_indices)}")
    
    # Get analysis data
    analysis = analyze_token_scores(sentences_data, candidate_indices, token_scores, threshold)
    
    # Display debug information
    for i, idx in enumerate(candidate_indices):
        data = analysis[idx]
        sentence = data["sentence"]
        token_indices = data["token_indices"]
        
        print(f"\nCandidate {i} (Sentence {idx}): \"{sentence}\"")
        print(f"Token indices: {token_indices}")
        
        # Get tokens and scores
        if token_indices:
            # Get tokens
            sentence_token_ids = [response_tokens[idx].item() if idx < len(response_tokens) else 0 
                                 for idx in token_indices]
            decoded_tokens = tokenizer.decode(sentence_token_ids)
            print(f"Decoded tokens: \"{decoded_tokens}\"")
            
            # Print score information
            print(f"Token scores: {data['scores']}")
            print(f"Number of high-scoring tokens: {len(data['high_scores'])} out of {len(data['scores'])}")
            print(f"Fraction of high-scoring tokens: {data['high_score_fraction']:.2f}")
        else:
            print("No tokens mapped to this sentence!")

