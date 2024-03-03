from torch import nn
from torch.functional import F

class LSTMClassification(nn.Module):

    def __init__(self, config, target_size=1):
        super(LSTMClassification, self).__init__()
        self.lstm = nn.LSTM(input_size=config["input_dim"], 
                            hidden_size=config["hidden_dim"],
                            num_layers=config["num_layers"],
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(500 * config["hidden_dim"], config["hidden_dim"]),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(config["hidden_dim"], target_size)
            )
        
    def forward(self, input_):
        lstm_out, (h, c) = self.lstm(input_)
        logits = self.fc(lstm_out)
        scores = F.sigmoid(logits)
        return scores.double()
    
class TransformerClassification(nn.Module):

    def __init__(self, config):
        super(TransformerClassification, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(**config['encoder'])
        self.transformer_encoder  = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'], enable_nested_tensor=False)
        self.fc_dim = config['fc']['dim']
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(500, self.fc_dim),
            nn.Dropout(config['fc']['dropout']),
            nn.ReLU(),
            nn.Linear(self.fc_dim, 1)
            )
        
    def forward(self, input_):
        encoder_out = self.transformer_encoder(input_)
        logits = self.fc(encoder_out)
        scores = F.sigmoid(logits)
        return scores