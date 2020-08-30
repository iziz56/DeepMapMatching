#%%
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
#%%
link6_GPS = np.load('/Users/gimjiung/Desktop/new_map_matching_algorithm/training_data/number_of_link_3/GPS_0.npy') 
link6_Label = np.load('/Users/gimjiung/Desktop/new_map_matching_algorithm/training_data/number_of_link_3/Label_0.npy') 

#%%
train_input = link6_GPS[0]
train_target = link6_Label[0]
train_input = train_input[train_input!=-1]
train_target = train_target[train_target!= -1]
train_target = np.unique(train_target)
train_input = np.append(train_input,[0,0,229,229])
train_target = np.append(train_target,[[229]])
train_input = torch.Tensor(train_input).reshape(-1,2).unsqueeze(0)
train_target = torch.LongTensor(train_target)
#%%
class EncoderRNN(nn.Module):
    def __init__(self,in_feature, hidden,num_layer):
        super(EncoderRNN, self).__init__()
        self.in_feature = in_feature
        self.hidden = hidden
        self.embed_fc = torch.nn.Linear(in_feature, hidden)
        self.rnncell = torch.nn.GRU(hidden, hidden,num_layer,batch_first=True)

    def forward(self, x,hidden):
        x = self.embed_fc(x)
        x, hidden = self.rnncell(x,hidden)
        return x, hidden
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Encoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)

#     def forward(self, x, hidden):
#         x = self.embedding(x).view(1, 1, -1)
#         x, hidden = self.gru(x, hidden)
#         return x, hidden
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x = self.embedding(x).view(1, 1, -1)
        x, hidden = self.gru(x, hidden)
        x = self.softmax(self.out(x[0]))
        return x, hidden
encoder = EncoderRNN(2,128,1)
decoder = Decoder(128,230)
# %%
SOS_token = 0
EOS_toekn = 229
def train( print_every=1, learning_rate=0.01):
    loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    # training_batch = [random.choice(pairs) for _ in range(n_iter)]
    # training_source = [tensorize(source_vocab, pair[0]) for pair in training_batch]
    # training_target = [tensorize(target_vocab, pair[1]) for pair in training_batch]
    for i in range(100):
        encoder_hidden = torch.zeros([1, 1, encoder.hidden])
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        _, encoder_hidden = encoder(train_input,encoder_hidden)
        decoder_input = torch.Tensor([[SOS_token]]).long()
        decoder_hidden = encoder_hidden

        target_length = train_target.size()[0]
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, train_target[di].unsqueeze(0))
            # decoder_input = target_tensor[di]  # teacher forcing
            decoder_input = decoder_output.topk(1)[1]
            
        

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_iter = loss.item() / target_length
        loss_total += loss_iter

        if i % print_every == 0:
            loss_avg = loss_total / print_every
            loss_total = 0
            print("[{} - {}%] loss = {:05.4f}".format(i, i / 1 * 100, loss_avg))

# %%
train()


# %%
