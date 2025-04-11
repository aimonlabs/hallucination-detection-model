import torch
import torch.nn as nn

class CKClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, 
                 intermediate_dim: int = 1024, hidden_dim: int = 1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, hidden_states, labels=None):
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}
