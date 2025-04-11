
import re
import nltk
import numpy as np

def extract_sentences(batch, token_labels, is_single_data=True):
    """
    Extract sentences containing tokens marked as 1.
    
    Args:
        batch: Dict with 'sentences' and 'offset_mapping'
        token_labels: Token-level binary labels (0/1), can be 1D or 2D tensor/list
        is_single_data: If True, process single instance, else process batch
    
    Returns:
        List of sentences (single instance) or list of lists (batch)
    """
    # Standardize input to batch format
    if is_single_data:
        batch = {key: [value] for key, value in batch.items()}
        # Ensure token_labels is a list of lists
        if isinstance(token_labels, list) and not isinstance(token_labels[0], list):
            token_labels = [token_labels]
        elif hasattr(token_labels, 'ndim') and token_labels.ndim == 1:
            token_labels = [token_labels]

    extracted_sentences = []
    
    for batch_idx, labels in enumerate(token_labels):
        sentences = batch['sentences'][batch_idx]
        offset_mapping = batch['offset_mapping'][batch_idx]
        
        # Flatten labels if needed
        if hasattr(labels, 'ndim') and labels.ndim > 1:
            labels = labels[0]  # Take first row if 2D tensor
        elif isinstance(labels, list) and isinstance(labels[0], list):
            labels = labels[0]  # Take first row if 2D list
            
        labels_array = np.array(labels)
        positive_token_indices = np.where(labels_array == 1)[0]
        
        if len(positive_token_indices) == 0:
            extracted_sentences.append([])
            continue
            
        # Get sentence indices for positive tokens
        sentence_indices = []
        cumulative_length = 0
        
        for sent_idx, sent in enumerate(sentences):
            next_cumulative_length = cumulative_length + len(sent)
            
            # Check if any positive tokens fall within this sentence's range
            sentence_has_hallucination = False
            for idx in positive_token_indices:
                if idx < len(offset_mapping):
                    tok_start_ix = offset_mapping[idx][0]
                    if cumulative_length <= tok_start_ix < next_cumulative_length:
                        sentence_has_hallucination = True
                        break  # Found a hallucination in this sentence, no need to check other tokens
            
            if sentence_has_hallucination:
                sentence_indices.append(sent_idx)
                
            cumulative_length = next_cumulative_length
        
        # Get sentences in order of appearance
        batch_sentences = [sent for idx, sent in enumerate(sentences) 
                         if idx in sentence_indices]
        
        extracted_sentences.append(batch_sentences)

    return extracted_sentences[0] if is_single_data else extracted_sentences

def prepare_single_input(context, response, tokenizer):
    """
    Prepare single text input for inference.
    
    Args:
        context: Context text string
        response: Response text string
        tokenizer: HuggingFace tokenizer
    """
    # Tokenize
    inputs = tokenizer(
        context,
        response,
        truncation=True,
        max_length=2048,
        padding=True,
        return_offsets_mapping=True
    )
    
    # Get sentences
    import nltk
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(response)
    
    batch = {
        'sentences': sentences,
        'offset_mapping': inputs.offset_mapping,
    }
    
    return batch, inputs

def add_period_if_missing(text):
    text = text.strip()
    if text and not re.search(r"[.?!]$", text):
        text += "."
    else:
        text += ""
    return text

def get_sentence_token_boundaries(sentences_data):
    """
    Get first and last token indices for each sentence.
    
    Args:
        sentences_data: Dictionary with sentence information
        
    Returns:
        List of tuples (first_idx, last_idx) for each sentence
    """
    boundaries = []
    for token_indices in sentences_data["token_indices"]:
        if token_indices:
            first_idx = min(token_indices)
            last_idx = max(token_indices)
            boundaries.append((first_idx, last_idx))
        else:
            boundaries.append((None, None))
    return boundaries

def prepare_sentences(response):
    """
    Prepare sentences from the response text - NO tokenization.
    
    Args:
        response: The response text
        
    Returns:
        Dictionary with sentence information
    """
    # Split the response into sentences
    try:
        sentences = nltk.sent_tokenize(response)
    except:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(response)
    
    # Return only the sentences
    sentences_data = {
        "sentences": sentences
    }
    
    return sentences_data

def map_tokens_to_sentences(sentences, response, tokenizer, tokenized_input, prompt_len):
    """
    Map tokens to sentences without additional tokenization.
    """
    # Extract response tokens
    response_tokens = tokenized_input["input_ids"][0][prompt_len:]
    decoded_response = tokenizer.decode(response_tokens)
    
    # Map sentences to character positions in decoded response
    token_indices = []
    for sentence in sentences:
        sentence_indices = []
        sentence_pos = decoded_response.find(sentence)
        if sentence_pos >= 0:
            # Find which tokens fall within this sentence
            char_pos = 0
            for i in range(len(response_tokens)):
                # Get this token's text
                token_text = tokenizer.decode(response_tokens[i])
                token_len = len(token_text)
                
                # Check if token is in this sentence
                token_start = char_pos
                token_end = char_pos + token_len
                if token_end > sentence_pos and token_start < sentence_pos + len(sentence):
                    sentence_indices.append(i)
                
                char_pos += token_len
        
        token_indices.append(sentence_indices)
    
    return token_indices

