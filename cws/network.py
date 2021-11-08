from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertOnlyMLMHead
import torch.nn as nn

class Network(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(Network, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None and masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='sum')
            loss_tag = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return loss_tag, masked_lm_loss
        else:
            return logits
