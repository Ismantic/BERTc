"""BertCRF: AutoModel encoder + Linear + CRF head for sequence labeling."""
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BertCRF(nn.Module):
    def __init__(self, model_path, num_tags=4, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.num_tags = num_tags

    def _emissions(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))

    def forward(self, input_ids, attention_mask, labels=None):
        emissions = self._emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            # CRF returns log-likelihood; loss = -log P
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss, emissions
        return emissions

    @torch.no_grad()
    def decode(self, input_ids, attention_mask):
        # CRF expects fp32 emissions for Viterbi numerics — cast out of bf16
        emissions = self._emissions(input_ids, attention_mask).float()
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)
