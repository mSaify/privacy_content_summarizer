from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

import torch

def get_float_tensor(data):
    val=[]
    for e in data:
        val.append(e.astype(float))
    return torch.tensor(val).float()


def data_loader(train_inputs, val_inputs, train_labels, val_labels,
                batch_size=50):

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(get_float_tensor(data) for data in
          [train_inputs, val_inputs, train_labels, val_labels])

    # Specify batch_size
    batch_size = 50

    # train_inputs = train_inputs.unsqueeze(0)
    # val_inputs = val_inputs.unsqueeze(0)
    # train_labels= train_labels.unsqueeze(0)
    # val_labels = val_labels.unsqueeze(0)

    print("original training X tensor shape")
    print(train_inputs.shape)

    print("original training y tensor shape")
    print(train_labels.shape)

    print("original test X tensor shape")
    print(val_inputs.shape)

    print("original test y tensor shape")
    print(val_labels.shape)



    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader