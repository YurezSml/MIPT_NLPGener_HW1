from transformers import AutoTokenizer, AutoModel
import torch

class CrossEncoderBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer.add_tokens(["[U_token]"], special_tokens=True)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.linear(pooled_output)