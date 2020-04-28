import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_dim, embed_dim, sparse):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim, sparse=sparse)
        self.linear = nn.Linear(embed_dim, vocab_dim)
        
    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=1)
        outs = self.linear(embeds)
        log_probs = F.log_softmax(outs)
        return log_probs

# Skip-Gram Model
class SkipGram(nn.Module):
    def __init__(self, vocab_dim, embed_dim, sparse):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim, sparse=sparse)
        self.linear = nn.Linear(embed_dim, vocab_dim)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        outs = self.linear(embeds)
        log_probs = F.log_softmax(outs)
        return log_probs