import torch.optim as optim
from model_training.using_torch.cnn_net import CNN_NLP


def initialize_model(
        input_size,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        num_classes=2,
        dropout=0.5,
        learning_rate=0.01,
        device=-1):
    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)


    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer
