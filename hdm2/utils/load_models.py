import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from hdm2.models.context_knowledge import TokenLogitsToSequenceModel
from huggingface_hub import hf_hub_download
import tempfile
from hdm2.models.common_knowledge import CKClassifier
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig

def load_model_components(model_components_path=None, 
                         use_hf=True, repo_id=None,
                         is_load_in_8bit=False,
                         quantization_config=None,
                         ):
    """
    Load the saved model components from local path or Hugging Face.
    """
    
    if use_hf:
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
    
   # Add before initializing the model
    if is_load_in_8bit:
        if quantization_config is None:
            # Setup quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                #llm_int8_skip_modules=[ 'tok_score', 
                #                       'seq_score']
            )
       
        # Initialize model with quantization
        model = TokenLogitsToSequenceModel(
            model_name=base_model_name,
            num_token_labels=num_token_labels,
            num_seq_labels=num_seq_labels,
            is_apply_peft=False,
            quantization_config=quantization_config
        )
    else:
       # Original model initialization
        model = TokenLogitsToSequenceModel(
            model_name=base_model_name,
            num_token_labels=num_token_labels,
            num_seq_labels=num_seq_labels,
            is_apply_peft=False
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.tok_score.load_state_dict(
        torch.load(tok_score_path, map_location=device)
    )
    model.seq_score.load_state_dict(
        torch.load(seq_score_path, map_location=device)
    )
    
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
    model_components_path='../models/token_seq_model/model_components/',
    ck_classifier_path='ck_classifier_op_2/checkpoint-4802/',
    use_hf=False,
    repo_id=None,
    device='cuda',
    is_load_in_8bit=False,
    quantization_config=None,
):
    """Load all components of the hallucination detection system."""
    # Load token model and tokenizer
    token_model, tokenizer = load_model_components(
        model_components_path=model_components_path,
        use_hf=use_hf,
        repo_id=repo_id,
        is_load_in_8bit=is_load_in_8bit,
        quantization_config=quantization_config,
    )
    
    # Load CK classifier
    if use_hf:
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
        
        # Load classifier with safetensors
        try:
            from safetensors.torch import load_file
            ck_classifier = CKClassifier(hidden_size=2048, num_labels=2).to(device)
            ck_weights = load_file(ck_path, device=device)
            ck_classifier.load_state_dict(ck_weights)
        except ImportError:
            # Fallback if safetensors isn't available
            import torch
            ck_classifier = CKClassifier(hidden_size=2048, num_labels=2).to(device)
            ck_classifier.load_state_dict(torch.load(ck_path, map_location=device))
    else:
        # Use local path
        ck_classifier = CKClassifier(hidden_size=2048, num_labels=2).to(device)
        ck_classifier = load_ck_checkpoint(ck_classifier, ck_classifier_path)
    
    return token_model, ck_classifier, tokenizer