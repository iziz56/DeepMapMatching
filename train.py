
#%%
import numpy as np
import torch
from models.rnn import RNNMODEL
#%%
# TODO :: add more data
# TODO :: load --> variable store --> torch dataloader
link6_GPS = np.load('D:/deepmapmatching/GPS_0.npy') 
link6_Label = np.load('D:/deepmapmatching/Label_0.npy') 

n_data = link6_GPS.shape[0]

# TODO :: argparse
learning_rate = 0.001
train_ratio = 0.7
test_ratio = 0.3
num_iterations = 100

n_train = int(n_data * train_ratio)
n_test  = n_data - n_train

randidx=  np.random.permutation(n_data)
train_idx = randidx[:n_train]
test_idx = randidx[n_train:(n_train+n_test)]

train_input = link6_GPS[train_idx]
train_label = link6_Label[train_idx]
train_len = train_input[:,:,0] != -1
train_len = train_len.sum(axis = 1)

test_input = link6_GPS[test_idx]
test_label = link6_Label[test_idx]
test_len = test_input[:,:,0] != -1
test_len = test_len.sum(axis = 1)

# unq_labels= np.unique(train_label)
# {i:unq_labels[i] for i in range(len(unq_labels))}


model = RNNMODEL(in_feature=2,
                out_feature=229,
                hidden = 128,
                num_layer=3)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss(ignore_index = -1)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

cuda = torch.device('cuda')
train_input = torch.Tensor(train_input)
train_label = torch.LongTensor(train_label)
train_len = torch.LongTensor(train_len)
train_len , idx = torch.sort(train_len,descending=True)
train_input = train_input[idx]
train_label = train_label[idx]
train_label.squeeze_()
train_input = train_input.cuda()
train_label = train_label.cuda()

#%%
for i in range(1):
    train_pred = model.forward(train_input, train_len)
    loss = criterion(   train_pred.view(train_label.size(0) * train_label.size(1), train_pred.size(2)) , 
                        train_label.view(train_label.size(0) * train_label.size(1)))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)

    # TODO :: tensorboard 



# %%
