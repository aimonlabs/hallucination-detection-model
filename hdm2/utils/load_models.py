import json
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from hdm2.models.context_knowledge import TokenLogitsToSequenceModel
from huggingface_hub import hf_hub_download
import tempfile
from hdm2.models.common_knowledge import CKClassifier
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig

def load_model_components(
    repo_id=None,
    model_components_path=None,
    load_in_8bit=False,
    quantization_config=None,
    dtype=torch.bfloat16,
    device='cuda',
):
    """
    Load the saved model components from Hugging Face or local path.
    
    Args:
        repo_id: HuggingFace repository ID (required if model_components_path is None)
        model_components_path: Path to local model components (if specified, load locally)
        load_in_8bit: Whether to load model in 8-bit precision
        quantization_config: Optional custom quantization configuration
    """
    
    if model_components_path is None:
        if repo_id is None:
            raise ValueError("Either repo_id or model_components_path must be provided")
            
        # Load from HuggingFace
        # Create a temporary directory to store downloaded files
        temp_dir = tempfile.mkdtemp()
        
        # Helper to download files from HF
        def get_file(filename):
            return hf_hub_download(
                repo_id=repo_id,
                filename=f"{filename}",
                local_dir=temp_dir
            )
        
        # 1. Load model configuration
        model_config_path = get_file("model_config.json")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        
        # For adapter, download all necessary files to a single directory
        adapter_dir = os.path.join(temp_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Download adapter config
        adapter_config_file = hf_hub_download(
            repo_id=repo_id,
            filename="cx/adapter_config.json",
            local_dir=adapter_dir
        )
        
        # Download adapter model
        adapter_model_file = hf_hub_download(
            repo_id=repo_id,
            filename="cx/adapter_model.safetensors",
            local_dir=adapter_dir
        )
        
        # Other paths
        adapter_path = os.path.dirname(adapter_config_file)
        tok_score_path = get_file("tok_score.pt")
        seq_score_path = get_file("seq_score.pt")
    else:
        # Load from local path
        logging.info(f"Loading model components from local path: {model_components_path}")
        
        # Use local paths
        with open(os.path.join(model_components_path, "model_config.json"), "r") as f:
            model_config = json.load(f)
        
        adapter_path = os.path.join(model_components_path, "lora_adapter")
        tok_score_path = os.path.join(model_components_path, "tok_score.pt")
        seq_score_path = os.path.join(model_components_path, "seq_score.pt")
    
    # 2. Get model parameters
    base_model_name = model_config["base_model_name"]
    num_token_labels = model_config["num_token_labels"]
    num_seq_labels = model_config["num_seq_labels"]
    
    # 3. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if dtype is None:
        dtype = torch.bfloat16

    # 4. Initialize model with or without quantization
    if load_in_8bit:
        if quantization_config is None:
            # Setup default quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            dtype = None
       
    model = TokenLogitsToSequenceModel(
        model_name=base_model_name,
        num_token_labels=num_token_labels,
        num_seq_labels=num_seq_labels,
        is_apply_peft=False,
        quantization_config=quantization_config,
        dtype=dtype,
        device=device,
    )

    # 5. Load LoRA adapter
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
        adapter_name = adapter_config.get("adapter_name", "default")
    
    # Load the adapter
    model.backbone = PeftModel.from_pretrained(
        model.backbone,
        adapter_path,
        adapter_name=adapter_name
    )
    
    # 6. Load classifier components
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model_dtype = next(model.parameters()).dtype

    # Load classifier weights and convert to right dtype
    tok_score_state = torch.load(tok_score_path, map_location=device)
    seq_score_state = torch.load(seq_score_path, map_location=device)

    # Convert state_dict dtype if needed
    if not load_in_8bit:  # No need to convert for 8-bit quantized models
        tok_score_state = {k: v.to(model_dtype) for k, v in tok_score_state.items()}
        seq_score_state = {k: v.to(model_dtype) for k, v in seq_score_state.items()}
    
    model.tok_score.load_state_dict(tok_score_state)
    model.seq_score.load_state_dict(seq_score_state)
    
    # 7. Move model to device
    model = model.to(device)
    
    return model, tokenizer

def load_ck_checkpoint(model, checkpoint_path):
    # If path is directory, look for model file
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.safetensors")
    else:
        model_path = checkpoint_path
    
    state_dict = load_file(model_path)
    
    model.load_state_dict(state_dict)
    return model

def load_hallucination_detection_model(
    repo_id=None,
    model_components_path=None,
    ck_classifier_path=None,
    device='cuda',
    load_in_8bit=False,
    quantization_config=None,
    dtype=torch.bfloat16,
):
    """
    Load all components of the hallucination detection system.
    
    Args:
        repo_id: HuggingFace repository ID (required if loading from HF)
        model_components_path: Path to local model components (if specified, load locally)
        ck_classifier_path: Path to local CK classifier (required if loading locally)
        device: Device to load model on ('cuda' or 'cpu')
        load_in_8bit: Whether to load model in 8-bit precision
        quantization_config: Optional custom quantization configuration
    """
    # Determine loading method and validate parameters
    loading_from_local = model_components_path is not None
    
    if loading_from_local:
        if ck_classifier_path is None:
            # Set default path for CK classifier when loading locally
            ck_classifier_path = 'ck_classifier_op_2/checkpoint-4802/'
            logging.info(f"Using default CK classifier path: {ck_classifier_path}")
    else:
        # Loading from HF
        if repo_id is None:
            raise ValueError("repo_id must be provided when loading from HuggingFace")
    
    # Load token model and tokenizer
    token_model, tokenizer = load_model_components(
        repo_id=repo_id,
        model_components_path=model_components_path,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
        dtype=dtype,
        device=device,
    )
    
    # Load classifier
    ck_classifier = CKClassifier(hidden_size=2048, num_labels=2).to(device)
    
    # Load CK classifier weights based on source
    if not loading_from_local:
        # Load from HuggingFace
        import tempfile
        from huggingface_hub import hf_hub_download
        
        # Create temp dir for CK
        temp_dir = tempfile.mkdtemp()
        
        # Get CK weights from HF
        ck_path = hf_hub_download(
            repo_id=repo_id,
            filename="ck/ck.safetensors",
            local_dir=temp_dir
        )
        
        # Load the weights
        try:
            # Try loading with safetensors
            ck_weights = load_file(ck_path, device=device)
        except Exception as e:
            # Fallback to regular torch loading
            ck_weights = torch.load(ck_path, map_location=device)
        
        # Load weights
        ck_classifier.load_state_dict(ck_weights)
    else:
        # Load from local path
        logging.info(f"Loading CK classifier from local path: {ck_classifier_path}")
        ck_classifier = load_ck_checkpoint(ck_classifier, ck_classifier_path)
    
    # Match CK classifier dtype to token model or specified dtype
    if load_in_8bit:
        model_dtype = next(token_model.parameters()).dtype
        ck_classifier = ck_classifier.to(model_dtype)
    elif dtype is not None:
        ck_classifier = ck_classifier.to(dtype)
    
    return token_model, ck_classifier, tokenizer