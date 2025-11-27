import torch
import torch.nn as nn
from transformers import RobertaModel

class HarmScoringModel(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.4)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # A more robust way to perform mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask

        dropped = self.dropout(mean_pooled_output)
        score = self.regressor(dropped)
        return score.squeeze(-1)