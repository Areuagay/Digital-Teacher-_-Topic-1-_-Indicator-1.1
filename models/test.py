import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Add a classification layer on top of BERT
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Run input through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the [CLS] token
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        # Pass the [CLS] token through the classification layer
        logits = self.fc(cls_hidden_state)
        return logits

# Example usage
text = "I love this product, it's amazing!"
num_classes = 2  # positive and negative classes

# Tokenize input text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Create the model
model = SentimentClassifier(num_classes=num_classes)

# Get the model's prediction
outputs = model(inputs['input_ids'], inputs['attention_mask'])
predicted_class = torch.argmax(outputs, dim=1)

print(predicted_class.item())
