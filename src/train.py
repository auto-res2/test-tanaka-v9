import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BackdoorModule(nn.Module):
    def __init__(self):
        super(BackdoorModule, self).__init__()
        self.fc = nn.Linear(512, 512)
        
    def forward(self, x, guidance):
        return self.fc(x)

class CamouflageModule(nn.Module):
    def __init__(self):
        super(CamouflageModule, self).__init__()
        self.fc = nn.Linear(512, 512)
        
    def forward(self, latent):
        return latent + 0.1 * torch.tanh(self.fc(latent))

class CamouflageMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(CamouflageMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        return x + torch.tanh(self.fc2(out))

class CamouflageResNet(nn.Module):
    def __init__(self, input_dim=512, num_blocks=3):
        super(CamouflageResNet, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            ))
        self.blocks = nn.ModuleList(layers)
        
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class CamouflageTransformer(nn.Module):
    def __init__(self, input_dim=512, num_heads=8):
        super(CamouflageTransformer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)
        return x + self.fc(attn_out)

class SurrogateDefense(nn.Module):
    def __init__(self, input_dim=512):
        super(SurrogateDefense, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        
    def forward(self, latent):
        return latent - 0.1 * torch.tanh(self.fc(latent))

def backdoor_activation_loss(outputs, target):
    return nn.MSELoss()(outputs, target)

def defense_evasion_loss(poisoned_latent, benign_latent):
    cos = nn.CosineSimilarity(dim=1)
    similarity = cos(poisoned_latent, benign_latent)
    return torch.mean(1 - similarity)
