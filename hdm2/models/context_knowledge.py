from dataclasses import dataclass
import torch
from transformers import AutoConfig, AutoModel, Trainer
from peft import get_peft_model, LoraConfig
from typing import Dict, Type
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
import bitsandbytes as bnb

@dataclass
class ModelOutputs:
    sequence_hidden: torch.Tensor
    token_hidden: torch.Tensor

class ModelHandler:
    def __init__(self, config, 
                 pad_token_id=None
                 ):
        
        self.config = config
        self.pad_token_id = pad_token_id
    def get_decoder_layer_str(self):
        if self.config.model_type == 't5':
            return 'num_decoder_layers'
        elif self.config.model_type in ['phi3', 'qwen2']:
            return 'num_hidden_layers'
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
    def preprocess_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def preprocess_seq_output(self, seq_output, input_ids):
        return seq_output

    def process_seq_logits(self, seq_logits, input_ids):

        batch_size = input_ids.shape[0]

        if self.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.pad_token_id).to(seq_logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=seq_logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            raise ValueError("Input ids are None")
        
            # TODO: Add logger warning
            """
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
            """

        pooled_logits = seq_logits[torch.arange(batch_size, device=seq_logits.device), last_non_pad_token]

        return pooled_logits

MODEL_HANDLERS: Dict[str, Type[ModelHandler]] = {
    "qwen2": ModelHandler,
    "phi": ModelHandler,
}

def get_model_handler(model_name: str) -> ModelHandler:
    """Get appropriate handler based on model name"""
    for key, handler_cls in MODEL_HANDLERS.items():
        if key in model_name.lower():
            return handler_cls
    return ModelHandler  # Default handler

class TokenLogitsToSequenceModel(nn.Module):
    def __init__(self, model_name, 
                 num_token_labels, 
                 num_seq_labels, 
                 num_decoder_layers=None,
                 is_apply_peft=True,
                 peft_config=None,
                 quantization_config=None,
                 dtype=torch.bfloat16,
                 device='cuda',
                 ):
        super(TokenLogitsToSequenceModel, self).__init__()

        base_config = AutoConfig.from_pretrained(model_name)
        
        pad_token_id = base_config.pad_token_id
        if pad_token_id is None:
            base_tokenizer = AutoTokenizer.from_pretrained(model_name)
            pad_token_id = base_tokenizer.pad_token_id

        base_model_type = base_config.model_type
        # TODO: Add T5 support
        assert base_model_type in ['qwen2', 'phi3']

        self.num_seq_labels = num_seq_labels
        self.num_tok_labels = num_token_labels

        model_handler_cls = get_model_handler(model_name)
        self.model_handler = model_handler_cls(base_config, pad_token_id)
    
        if num_decoder_layers is not None:
            base_num_decoders = base_config.num_hidden_layers
            assert 1 < num_decoder_layers <= base_num_decoders
        
        num_decoder_key_str = self.model_handler.get_decoder_layer_str()

        # Build model kwargs
        model_kwargs = {}
        if num_decoder_layers is not None:
            model_kwargs[num_decoder_key_str] = num_decoder_layers

        # Set dtype
        if quantization_config is None and dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        # Add quantization config if provided
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs["device_map"] = "auto" if device == 'cuda' else None
                    
        # Load the model with all appropriate parameters
        self.backbone = AutoModel.from_pretrained(model_name, 
                                                  **model_kwargs)
        

        if is_apply_peft:
            if peft_config is None:
                peft_config = LoraConfig(
                    #task_type=TaskType.TOKEN_CLS,
                    inference_mode=False,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules='all-linear',
                    bias="none",
                    #target_modules=[
                    #    "q", "k", "v", "o", # For T5
                    #    #"self_attn.qkv_proj",  # attention weights
                    #    #"mlp.gate_up_proj",   # MLP
                    #    #"mlp.down_proj"       # MLP output
                    #],
                )
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.config = AutoConfig.from_pretrained(model_name)
        #super().__init__(self.config)

        model_dtype = next(self.backbone.parameters()).dtype
        
        if self.num_seq_labels is not None:
            self.seq_score = nn.Linear(
                self.backbone.config.hidden_size, 
                self.num_seq_labels, 
                bias=False,
                ).to(model_dtype)

        if self.num_tok_labels is not None:
            if getattr(self.backbone.config, "classifier_dropout", None) is not None:
                classifier_dropout = self.backbone.config.classifier_dropout
            elif getattr(self.backbone.config, "hidden_dropout", None) is not None:
                classifier_dropout = self.backbone.config.hidden_dropout
            else:
                classifier_dropout = 0.1
            self.tok_dropout = nn.Dropout(classifier_dropout)

            self.tok_score = nn.Linear(
                self.backbone.config.hidden_size, 
                self.num_tok_labels
            ).to(model_dtype)
        
    def get_logits(self, input_ids: torch.Tensor, 
                   attention_mask: torch.Tensor,
                   ) -> torch.Tensor:
        preprocessed_inputs = self.model_handler.preprocess_inputs(input_ids, attention_mask)
        
        model_outputs = self.backbone(
            **preprocessed_inputs
            )
        
        hidden_state = model_outputs[0]

        seq_output = self.model_handler.preprocess_seq_output(hidden_state, input_ids)
        seq_logits = self.seq_score(seq_output)
        seq_logits = self.model_handler.process_seq_logits(seq_logits, input_ids)

        tok_output = self.tok_dropout(hidden_state)
        tok_logits = self.tok_score(tok_output)

        return tok_logits, seq_logits
        
    def forward(self, input_ids, attention_mask,
                ):
        
        token_logits, seq_logits = self.get_logits(input_ids, attention_mask)

        #print(input_ids.shape, outputs.logits.shape)
        
        # Attention mask: Shape (batch_size, seq_length)
        # Token logits: Shape (batch_size, seq_length, num_token_classes)
        # if mean-pooling over valid tokens, then:
        """
        masked_logits = token_logits * attention_mask.unsqueeze(-1)  # Mask padded logits (broadcast attention_mask)
        sum_logits = masked_logits.sum(dim=1)  # Sum over sequence length
        valid_token_counts = attention_mask.sum(dim=1).unsqueeze(-1)  # Count valid tokens per sequence
        mean_logits = sum_logits / valid_token_counts  # Divide by valid token counts
        """
        
        return {'token_logits': token_logits, 
                'seq_logits': seq_logits}
