from transformers import DistilBertModel, DistilBertPreTrainedModel, AutoModelForMaskedLM, AutoModel
from utils import GEGLU, ReGLU
from torch import nn
import torch



class DistilBertForPretrain(DistilBertPreTrainedModel):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__(config)

        # DistilBert base model
        self.albert = AutoModelForMaskedLM.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
#         self.albert = AutoModel.from_pretrained(config=config)

        # MLM Head
        self.loss_fct = nn.CrossEntropyLoss()
        self.sop_classifier = nn.Linear(768, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        sop_labels=None,
    ):
        # Extracting features from DistilBert
        albert_output = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = albert_output[0]  # (batch_size, sequence_length, dim)

        # MLM prediction
#         mlm_logits = self.mlm_classifier(hidden_states)  # (batch_size, sequence_length, vocab_size)
        output = self.distillbert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
        
        hidden_states = output.hidden_states[-1]
        # SOP prediction (only using the first token's representation)
        mlm_logits = output.logits
        sop_logits = self.sop_classifier(hidden_states[:, 0, :])  # (batch_size, 2)

        outputs = (mlm_logits, sop_logits)

        # Loss calculation
        if labels is not None:
            mlm_loss = output.loss
            sop_loss = self.loss_fct(sop_logits.view(-1, 2), sop_labels.view(-1))
            total_loss = (mlm_loss + sop_loss)/2
            outputs = (total_loss,) + outputs

        return outputs

class DistillBertMult(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.classifier = torch.nn.Linear(768, 500)

    def forward(self, input_ids, attention_mask):
        output_1 = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.hidden_act(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class DistillBertBin(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.hidden_act(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class DistilBertForToken(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        self.classifier = torch.nn.Linear(self.albert.config.dim, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output[zero_msk])
        outputs = self.classifier(sequence_output)

        return outputs  # (loss), scores

class DistilBertForBinMult(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        output_1 = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        return bin_output, mult_output
    
class DistilBertForBinMultToken(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )
        self.token_classifier = torch.nn.Sequential(
            self.dropout,
            torch.nn.Linear(self.albert.config.dim, 2)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        token_output = self.token_classifier(hidden_state[zero_msk])
        return bin_output, mult_output, token_output
    
class DistilBertForBinMultTokenFreeze(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        for param in self.albert.parameters():
            param.requires_grad = False
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )
        self.token_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(self.albert.config.dim, 2)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        token_output = self.token_classifier(hidden_state[zero_msk])
        return bin_output, mult_output, token_output
    
class DistilBertForBinMultTokenKNN(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )
        self.token_classifier = torch.nn.Sequential(
            self.dropout,
            torch.nn.Linear(self.albert.config.dim, 2)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        token_output = self.token_classifier(hidden_state[zero_msk])
        return bin_output, mult_output, token_output, pooler
    
class DistilBertForBinMultTokenLearnLoss(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )
        self.token_classifier = torch.nn.Sequential(
            self.dropout,
            torch.nn.Linear(self.albert.config.dim, 2)
        )
        
class DistilBertForBinMultTokenLoss(torch.nn.Module):
    def __init__(self, config, custom_hidden_act=False, loss = "avr", pretrained_path="albert-base-uncased"):
        super().__init__()
        self.albert = DistilBertModel.from_pretrained(pretrained_path, config=config)
        self.hidden_act = nn.ReLU()
        if custom_hidden_act == "geglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = GEGLU()
                self.hidden_act = GEGLU()
        elif custom_hidden_act == "reglu":
            for i in range(len(self.albert.encoder.layer)):
                self.albert.encoder.layer[i].intermediate.act_fn = ReGLU()
                self.hidden_act = ReGLU()
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.bin_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 2)
        )
        self.mult_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            self.hidden_act,
            self.dropout,
            torch.nn.Linear(768, 500)
        )
        self.token_classifier = torch.nn.Sequential(
            self.dropout,
            torch.nn.Linear(self.albert.config.dim, 2)
        )
        if loss == "avr": self.loss_weight = torch.tensor([1/3, 1/3, 1/3])
        elif loss == "naive": self.loss_weight = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
#         elif loss == "uncertainty": self.loss_weight = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        token_output = self.token_classifier(hidden_state[zero_msk])
        return bin_output, mult_output, token_output
    
    def forward(self, input_ids, attention_mask=None, labels=None, zero_msk=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooler = hidden_state[:, 0]
        
        bin_output = self.bin_classifier(pooler)
        mult_output = self.mult_classifier(pooler)
        token_output = self.token_classifier(hidden_state[zero_msk])
        return bin_output, mult_output, token_output, self.loss_weight
    
# class DistilBertForBinMultToken(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.albert = DistilBertModel.from_pretrained('albert-base-uncased', config=config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.bin_classifier = torch.nn.Linear(self.albert.config.dim, 2)
#         self.mult_classifier = torch.nn.Linear(768, 500)
#         self.bin_pre_classifier = torch.nn.Linear(768, 768)
#         self.mult_pre_classifier = torch.nn.Linear(768, 768)

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         output_1 = self.albert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_state = output_1[0]
#         pooler = hidden_state[:, 0]
#         pooler = self.bin_pre_classifier(pooler)
#         pooler = torch.nn.ReLU()(pooler)
#         pooler = self.dropout(pooler)
#         bin_output = self.bin_classifier(pooler)
        
#         pooler = self.mult_pre_classifier(pooler)
#         pooler = torch.nn.ReLU()(pooler)
#         pooler = self.dropout(pooler)
#         mult_output = self.mult_classifier(pooler)
        
#         return bin_output, mult_output
    
    