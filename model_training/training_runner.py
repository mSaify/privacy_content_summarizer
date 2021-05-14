import random
import time
from torch import nn
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

loss_fn = nn.CrossEntropyLoss()


from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, optimizer, train_dataloader, val_dataloader=None, epochs=10):
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(
        f"'Epoch' |'Train Loss' | 'Val Loss' |  'Val Acc' | 'Val Precision' | 'Val Recall' | 'Val F1 Score' | 'Elapsed' ")
    print("-" * 60)

    for epoch_i in range(epochs):

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            # b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels = tuple(t for t in batch)
            model.zero_grad()
            logits = model(b_input_ids)

            # logits=torch.reshape(logits, b_labels.shape)
            b_labels = b_labels.long()

            # print(b_labels)
            # print(logits)

            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            loss.backward()

            # Update params
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        if val_dataloader is not None:

            val_loss, val_accuracy, val_f1, val_recall, val_precision = evaluate(model, val_dataloader)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy


            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss: ^ 10.6f} |   {val_accuracy: ^ 9.2f} |  {val_precision: ^ 12.2f} | {val_recall: ^ 10.2f} | {val_f1: ^ 12.2f} | {time_elapsed: ^ 9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")


def evaluate(model, val_dataloader):
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    val_f1 = []
    val_precision = []
    val_recall = [0]

    for batch in val_dataloader:
        # b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = tuple(t for t in batch)
        b_labels = b_labels.long()

        with torch.no_grad():
            logits = model(b_input_ids)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        #print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(preds, b_labels)*100
        val_precision.append(precision)
        #print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(preds, b_labels)*100
        val_recall.append(recall)
        # print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(preds, b_labels)*100
        #print('f1 score: %f' % f1)

        val_f1.append(f1)

        #print('F1 score: %f' % f1)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    val_f1 = np.mean(val_f1)
    val_precision = np.mean(val_precision)
    val_recall = np.mean(val_recall)

    return val_loss, val_accuracy, val_f1, val_recall, val_precision
