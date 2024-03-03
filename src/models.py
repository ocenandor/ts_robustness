from torch import nn
from torch.functional import F
import numpy as np

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
        config_m = config['model']
        config_d = config['data']
        
        n_features = config_m['embedding']['out_channels']
        nhead = int(np.sqrt(n_features))
        self.embedding_layer = nn.Conv1d(padding=1, kernel_size=3, **config_m['embedding'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=nhead, **config_m['encoder'])
        self.transformer_encoder  = nn.TransformerEncoder(self.encoder_layer, num_layers=config_m['num_layers'], enable_nested_tensor=False)
        self.fc_dim = config_m['fc']['inner_dim']
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config_d['seq_length'] * n_features, self.fc_dim),
            nn.Dropout(config_m['fc']['dropout']),
            nn.ReLU(),
            nn.Linear(self.fc_dim, 1)
            )
        
    def forward(self, input_):
        if len(input_.shape) == 2:
            input_ = input_.unsqueeze(2)
        input_ = input_.permute(0, 2, 1)
        embedding_out = self.embedding_layer(input_).permute(0, 2, 1)
        encoder_out = self.transformer_encoder(embedding_out)
        logits = self.fc(encoder_out)
        scores = F.sigmoid(logits)
        return scores