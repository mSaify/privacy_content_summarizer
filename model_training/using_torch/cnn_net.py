import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_NLP(nn.Module):

    def __init__(self,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):

        super(CNN_NLP, self).__init__()

        self.num_filters=num_filters
        self.num_classes = num_classes
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=(filter_sizes[i]))

            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):

        x_reshaped = input_ids.permute(0,2,1)
        #print(x_reshaped.shape)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        #print("convlution filter shape and size")
        #print(x_conv_list[0].shape)

        x_pool_list = [F.max_pool1d(x_conv, kernel_size=(x_conv.shape[2]))
            for x_conv in x_conv_list]


        x_fc = torch.cat([x_pool.squeeze(2) for x_pool in x_pool_list],
                         dim=1)


        # print('======= x_Fc shape 1======')
        # print(x_fc.shape)
        # print(x_fc[0].shape)

        # x_fc = torch.cat([x_pool for x_pool in x_fc],
        #                  dim=1)

        fc_shape = (self.num_classes,np.sum(self.num_filters))


        logits = self.fc(self.dropout(x_fc))

        return logits