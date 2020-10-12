################## LIBRARIES ##################

# General purposes
import pandas as pd
import numpy as np
# pytorch
import torch
# Dataset
from torch.utils.data import Dataset, DataLoader 
# Optimizers
import torch.optim as optim
# This helps us display a summary of our model
from torchsummary import summary

################## SOME AUXILIARY FUNCTIONS ##################

# Shows a summary of the network
def network_summary(network, input_size):
    summary(network, input_data = input_size)
    return None

# Calculates the labels correspondig to the network's output
def predict_labels(output):
    _, pred_labels = torch.max(output, dim = 1)
    return pred_labels

# Calculates the number of matches between the predicted labels and the true labels
def correct_predictions(output, labels):
    pred_labels = predict_labels(output)
    corr_pred = torch.sum(pred_labels == labels).item()
    return corr_pred

# Print the progress and the metrics values during the training

def print_training_progress(epoch, epochs, batch_num, total_batches, notebook = False):
    if notebook:
        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ', Batch: ' + str(batch_num + 1) + '/' + str(total_batches))
    else:
        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs) + ', Batch: ' + str(batch_num + 1) + '/' + str(total_batches), end = '\r')
    return None

def print_performance(metrics):
    epoch = metrics['epoch']
    train_loss = metrics['train_loss']
    train_acc = metrics['train_acc']
    valid_loss = metrics['valid_loss']
    valid_acc = metrics['valid_acc']
    print('Epoch: ' + str(epoch + 1) + ' || Train loss = ' + str(round(train_loss,2)) + ', Train Acc = ' + str(round(train_acc,2)) + ', Valid Loss = ' + str(round(valid_loss,2)) + ', Valid Acc = ' + str(round(valid_acc,2)))
    return None

###################### EVALUATE ######################
# This function evaluates the model's performance on a given data set

def evaluate(network, dataloader, loss_fn, device = None):
    
    # Sends the network to the device
    if device:
        network.to(device)
    
    # Sets the metrics to zero
    network.eval()
    loss_value = 0
    acc_value = 0
    total_images = 0
    # Loops over the dataset
    for data in dataloader:
        # Gets the data
        images, labels = data
        # Sends the data to the device
        if device:
            images, labels = images.to(device), labels.to(device)
        # Evaluates
        output = network(images)
        # Updates the metrics
        loss = loss_fn(output, labels)
        loss_value = loss_value + loss.item()
        acc_value = acc_value + correct_predictions(output, labels)
        total_images = total_images + labels.size(0)

    loss_value = loss_value / len(dataloader)
    acc_value = (acc_value / total_images) * 100

    return loss_value, acc_value

################## TRAINIGN ROUTINE ##################

def train_network(network, epochs, train_dataloader, valid_dataloader, loss_fn, optimizer, save_last_model = False, save_best_model = False, device = None, save_history = True):

    # Keeps track of the best model's performance
    best_accuracy = - np.inf
    # Stores the training history during
    training_history = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])

    # Sends the network to the device
    if device:
        network.to(device)

    # Training loop
    for epoch in range(epochs):

        #TRAINING
        network.train()
        # Set the metrics to 0 for this epoch
        train_loss = 0
        train_acc = 0
        total_images_train = 0

        # Trains over the training set
        for batch_num, data in enumerate(train_dataloader):
            # Print progess 
            #print_training_progress(epoch, len(train_dataloader), batch_num)
            # Gets the data
            images, labels = data
            # Sends the data to the device
            if device:
                images, labels = images.to(device), labels.to(device)
            # training step
            optimizer.zero_grad()
            output = network(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            # Updates the metrics
            train_loss = train_loss + loss.item()
            train_acc = train_acc + correct_predictions(output, labels)
            total_images_train = total_images_train + labels.size(0)
        
        train_loss = train_loss / len(train_dataloader)
        train_acc = (train_acc / total_images_train) * 100

        # VALIDATION
        valid_loss, valid_acc = evaluate(
                                    network=network, 
                                    dataloader=valid_dataloader, 
                                    loss_fn=loss_fn,
                                    device=device
                                    )

        # Saves the metrics per epoch
        metrics_epoch = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc}
        # Stores the epoch's metrics
        training_history = training_history.append(metrics_epoch, ignore_index = True)
        # Saves the training history so far
        if save_history:
            training_history.to_csv('training_history.csv', index = False)
        # Print the performance for the epoch
        print_performance(metrics_epoch)
        # Saves the model
        if save_last_model:
            torch.save(network.state_dict, 'last_weights.pt')
        # Saves the best model
        if save_best_model and valid_acc > best_accuracy:
            torch.save(network.state_dict(), 'best_weights.pt')
            best_accuracy = valid_acc     
            
    print('Finished Training, Hurray!!! :D')
    
    # Returns the network and the trainig history
    return network, training_history