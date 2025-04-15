import torch
import nltk
import logging
from hdm2.utils.load_models import load_hallucination_detection_model
from hdm2.utils.model_utils import _detect_hallucinations

class HallucinationDetectionModel:
    """
    A wrapper class for hallucination detection models that provides a simplified interface.
    """
    
    def __init__(self, 
                repo_id='AimonLabs/hallucination-detection-model', 
                model_components_path=None, 
                ck_classifier_path=None,
                device=None,
                load_in_8bit=False,
                quantization_config=None):
        """
        Initialize the hallucination detection model.
        
        Args:
            repo_id: HuggingFace repository ID (required if loading from HF)
            model_components_path: Path to local model components (if specified, load locally)
            ck_classifier_path: Path to local CK classifier (required if loading locally)
            device: Device to load model on ('cuda' or 'cpu'), auto-detected if None
            load_in_8bit: Whether to load model in 8-bit precision
            quantization_config: Optional custom quantization configuration
        """
        # Download NLTK data
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
        except Exception as e:
            logging.warning(f"Failed to download NLTK punkt: {e}")
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load models
        self.token_model, self.ck_classifier, self.tokenizer = load_hallucination_detection_model(
            repo_id=repo_id,
            model_components_path=model_components_path,
            ck_classifier_path=ck_classifier_path,
            device=self.device,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config
        )
        
        logging.info(f"Hallucination Detection Model loaded successfully on {self.device}")
    
    def apply(self, prompt, context, response, 
             token_threshold=0.5,
             ck_threshold=0.7,
             ck_layer_ix=25,
             use_last_tokens=False,
             use_truncated_context=True,
             debug=False,
             is_include_spans=True):
        """
        Apply hallucination detection to a prompt, context and response.
        
        Args:
            prompt: Prompt text
            context: Context text
            response: Response text
            token_threshold: Threshold for token probability to be considered hallucination
            ck_layer_ix: Layer index for feature extraction
            use_last_tokens: Whether to use last token embeddings
            use_truncated_context: Whether to use truncated context for each sentence
            debug: Enable debug output
            is_include_spans: Include word spans in results
            
        Returns:
            Dictionary containing hallucination detection results
        """
        
        return _detect_hallucinations(
            prompt=prompt,
            context=context,
            response=response,
            token_model=self.token_model,
            ck_classifier=self.ck_classifier,
            tokenizer=self.tokenizer,
            ck_layer_ix=ck_layer_ix,
            token_threshold=token_threshold,
            ck_threshold=ck_threshold,
            use_last_tokens=use_last_tokens,
            use_truncated_context=use_truncated_context,
            debug=debug,
            is_include_spans=is_include_spans,
            device=self.device
        ) 