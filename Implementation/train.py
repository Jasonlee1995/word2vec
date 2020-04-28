import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.notebook import tqdm
from model import CBOW, SkipGram

torch.manual_seed(0)


class word2vec():
    def __init__(self, mode, vocab_dim, embed_dim, sparse):
        self.mode = mode
        if self.mode == 'cbow':
            self.model = CBOW(vocab_dim, embed_dim, sparse)
        elif self.mode == 'skip-gram':
            self.model = SkipGram(vocab_dim, embed_dim, sparse)
        
    def train(self, training_data, num_epochs=3, learning_rate=0.025):
        # Upload Model to GPU
        device = torch.device('cuda:0')
        self.model.to(device)
        
        # Set Optimizer and Linear Scheduler
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler_1 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma = 2/3)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma = 1/2)
        
        # Set Loss Function
        loss_function = nn.NLLLoss()
        
        # Train
        for epoch in range(num_epochs):
            print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in tqdm(enumerate(training_data)):
                if X.nelement() != 0:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = loss_function(self.model.forward(X), y)
                    loss.backward()
                    optimizer.step()
                    if i%50000 == 0:
                        print('Iteration : {}, Loss : {:.6f}'.format(i, loss.item()))
            if epoch == 0:
                scheduler_1.step()
            elif epoch == 1:
                scheduler_2.step()