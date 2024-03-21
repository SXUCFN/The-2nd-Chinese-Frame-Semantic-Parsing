import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = 64
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(768, 768 // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.num_labels * self.inner_dim * 2)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, target=None, labels=None, device=None, for_test=False):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # hidden_token = bert_out.last_hidden_state
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        outputs = self.dense(hidden_token)
        # lstm_out, (hidden, _) = self.lstm(hidden_token)
        # outputs = self.dense(lstm_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(hidden_token.shape[0], hidden_token.shape[1], 64, device)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.inner_dim ** 0.5
        token_logits = torch.concat([logits[i][:, target[i][0], target[i][1]].unsqueeze(0) for i in range(len(target))], dim=0)

        if for_test:
            loss = None
        else:
            loss_fc = torch.nn.CrossEntropyLoss()
            loss = loss_fc(token_logits, labels)
        return {
            "logits": token_logits,
            "loss": loss
        }

