"""Multi-task BertCRF: shared encoder + 3 heads (CWS-CRF + POS-Linear + NER-CRF)。

forward 输入 input_ids/attention_mask + 任意子集 labels,返回 losses dict。
decode_cws / decode_ner: 用 CRF viterbi 解码 BIES/BIO。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF


class BertMT(nn.Module):
    def __init__(self, model_path, num_cws=4, num_pos=27, num_ner=13, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # CWS head with CRF
        self.cws_classifier = nn.Linear(hidden, num_cws)
        self.cws_crf = CRF(num_cws, batch_first=True)
        # POS head: 直接 cross-entropy(class 多,用 CRF 慢且不必要)
        self.pos_classifier = nn.Linear(hidden, num_pos)
        # NER head with CRF
        self.ner_classifier = nn.Linear(hidden, num_ner)
        self.ner_crf = CRF(num_ner, batch_first=True)

    def forward(self, input_ids, attention_mask,
                cws_labels=None, pos_labels=None, ner_labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.dropout(out.last_hidden_state)
        cws_emi = self.cws_classifier(hs)
        pos_logits = self.pos_classifier(hs)
        ner_emi = self.ner_classifier(hs)
        mask = attention_mask.bool()
        losses = {}
        if cws_labels is not None:
            losses["cws"] = -self.cws_crf(cws_emi, cws_labels, mask=mask, reduction="mean")
        if pos_labels is not None:
            losses["pos"] = F.cross_entropy(
                pos_logits.view(-1, pos_logits.size(-1)),
                pos_labels.view(-1), ignore_index=-100)
        if ner_labels is not None:
            losses["ner"] = -self.ner_crf(ner_emi, ner_labels, mask=mask, reduction="mean")
        return losses, (cws_emi, pos_logits, ner_emi)

    @torch.no_grad()
    def decode_cws(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emi = self.cws_classifier(self.dropout(out.last_hidden_state)).float()
        return self.cws_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def decode_ner(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emi = self.ner_classifier(self.dropout(out.last_hidden_state)).float()
        return self.ner_crf.decode(emi, mask=attention_mask.bool())

    @torch.no_grad()
    def predict_pos(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.pos_classifier(self.dropout(out.last_hidden_state))
        return logits.argmax(-1)
