import torch
from torch import nn
from torchinfo import summary as torch_summary



class Tester(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.start = nn.Sequential(
            nn.Linear(in_features = 16, out_features = 64),
            nn.LeakyReLU())
        
        self.lstm = nn.LSTM(
            input_size = 64, 
            hidden_size = 128,
            batch_first=True)
        
        self.finish = nn.Sequential(
            nn.Linear(in_features = 128, out_features = 3),
            nn.Sigmoid())
        
    def forward(self, x, hidden = None):
        if(len(x.shape) == 2): sequence = False
        else:                sequence = True
        x = self.start(x)
        if(not sequence):
            x = x.view(x.shape[0], 1, x.shape[1])
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(x)
        else:               x, hidden = self.lstm(x, (hidden[0], hidden[1]))
        if(not sequence):
            x = x.view(x.shape[0], x.shape[-1])
        x = self.finish(x)
        return(x, hidden)
    

tester = Tester()
print("\n\n")
print(tester)
print()
print(torch_summary(tester, (1,16)))


easy_in = torch.rand((4,16))
easy_out, hidden = tester(easy_in)
print(easy_in.shape)
print(easy_out.shape, hidden[0].shape, hidden[1].shape)


hard_in = torch.rand((4,8,16))
hard_out, hidden = tester(hard_in)
print(hard_in.shape)
print(hard_out.shape, hidden[0].shape, hidden[1].shape)
    