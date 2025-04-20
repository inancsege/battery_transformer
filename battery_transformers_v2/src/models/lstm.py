import torch
import torch.nn as nn
from .base_model import BaseModel

class SOHLSTM(BaseModel): # Inherit from BaseModel if defined
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1, output_dim=100):
        super(SOHLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # Adjust the final layer to match the desired output dimension (e.g., number of future steps to predict)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim) # Output matches target length
        )
        self.output_dim = output_dim

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        output = self.fc(last_time_step_out)
        # Ensure output shape matches target [batch_size, output_dim]
        # If output_dim is 1, maybe squeeze is needed depending on loss fn
        if self.output_dim == 1:
             return output.squeeze(-1) # Or keep as [batch_size, 1]
        return output
