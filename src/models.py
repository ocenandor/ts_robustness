from torch import nn
from torch.functional import F
import numpy as np

class LSTMClassification(nn.Module):

    def __init__(self, config):
        super(LSTMClassification, self).__init__()
        config_m = config['model']
        config_d = config['data']
        self.lstm = nn.LSTM(input_size=config_d["seq_length"], 
                            hidden_size=config_m["lstm"]["hidden_dim"],
                            num_layers=config_m["lstm"]["num_layers"],
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config_m["lstm"]["hidden_dim"], config_m["fc"]["fc_dim"]),
            nn.Dropout(config_m["fc"]["dropout"]),
            nn.ReLU(),
            nn.Linear(config_m["fc"]["fc_dim"], 2)
            )
        
    def forward(self, input_):
        if len(input_.shape) == 3:
            input_ = input_.squeeze(2)
        lstm_out, (h, c) = self.lstm(input_)
        logits = self.fc(lstm_out)
        scores = F.sigmoid(logits)
        return scores
    
    def eval(self):
        self.lstm.train()
        self.fc.eval()

class CNNClassification(nn.Module):

    def __init__(self, config):
        
        super(CNNClassification, self).__init__()
        config_m = config['model']
        config_d = config['data']
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=config_m['backbone']["input_dim"],
                      out_channels=config_m['backbone']["hidden_dim"], 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=config_m['backbone']["hidden_dim"],
                      out_channels=config_m['backbone']["hidden_dim"], 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config_d['seq_length'] // 2 // 2 * config_m['backbone']["hidden_dim"], 256),
            nn.ReLU(),  
            nn.Linear(256, 2), 
        )

    def forward(self, input_):
        if len(input_.shape) == 2:
            input_ = input_.unsqueeze(2)
        input_ = input_.permute(0, 2, 1)
        input_ = self.backbone(input_)
        logits = self.fc(input_)
        return logits

    
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
            nn.Linear(self.fc_dim, 2)
            )
        
    def forward(self, input_):
        if len(input_.shape) == 2:
            input_ = input_.unsqueeze(2)
        input_ = input_.permute(0, 2, 1)
        embedding_out = self.embedding_layer(input_).permute(0, 2, 1)
        encoder_out = self.transformer_encoder(embedding_out)
        logits = self.fc(encoder_out)
        return logits
