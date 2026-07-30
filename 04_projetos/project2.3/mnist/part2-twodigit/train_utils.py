"""Training utilities."""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""

    def forward(self, input):
        return input.view(input.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append(
            {
                'x': torch.tensor(x_data[i : i + batch_size], dtype=torch.float32),
                'y': torch.tensor(
                    [y_data[0][i : i + batch_size], y_data[1][i : i + batch_size]],
                    dtype=torch.int64,
                ),
            }
        )
    return batches


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def train_model(
    train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=30
):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
    )

    train_acc_history = []
    dev_acc_history = []
    train_loss_history = []
    dev_loss_history = []

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Training
        train_loss, train_acc = run_epoch(train_data, model.train(), optimizer)
        print(
            'Train | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(
                train_loss[0], train_acc[0], train_loss[1], train_acc[1]
            )
        )

        # Validation
        dev_loss, dev_acc = run_epoch(dev_data, model.eval(), None)
        print(
            'Valid | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(
                dev_loss[0], dev_acc[0], dev_loss[1], dev_acc[1]
            )
        )

        # Save history
        train_acc_history.append(train_acc)
        dev_acc_history.append(dev_acc)
        train_loss_history.append(train_loss)
        dev_loss_history.append(dev_loss)

        # Save model
        torch.save(model, 'mnist_model_fully_connected.pt')

    return train_acc_history, dev_acc_history, train_loss_history, dev_loss_history


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses_first_label = []
    losses_second_label = []
    batch_accuracies_first = []
    batch_accuracies_second = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in data:  # tqdm(data): # tqdm barra de progresso
        # Grab x and y
        x, y = batch['x'], batch['y']

        # Get output predictions for both the upper and lower numbers
        out1, out2 = model(x)

        # Predict and store accuracy
        predictions_first_label = torch.argmax(out1, dim=1)
        predictions_second_label = torch.argmax(out2, dim=1)
        batch_accuracies_first.append(compute_accuracy(predictions_first_label, y[0]))
        batch_accuracies_second.append(compute_accuracy(predictions_second_label, y[1]))

        # Compute both losses
        loss1 = F.cross_entropy(out1, y[0])
        loss2 = F.cross_entropy(out2, y[1])
        losses_first_label.append(loss1.data.item())
        losses_second_label.append(loss2.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            joint_loss = 0.5 * (loss1 + loss2)
            joint_loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses_first_label), np.mean(losses_second_label)
    avg_accuracy = np.mean(batch_accuracies_first), np.mean(batch_accuracies_second)
    return avg_loss, avg_accuracy
