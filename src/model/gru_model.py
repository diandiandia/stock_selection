import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

class GRUTrainer:
    def __init__(self, input_size, seq_len=10, hidden_size=64, num_layers=2, lr=0.001, batch_size=64, epochs=20, device=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GRUModel(input_size, hidden_size, num_layers).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, X_tensor, y_tensor):
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.view(-1), y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    def predict(self, X_seq):
        self.model.eval()
        with torch.no_grad():
            X_seq = X_seq.to(self.device)
            prob = self.model(X_seq).cpu().numpy()
        return prob
    
    def save(self, path="models/gru_model"):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path="models/gru_model"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
