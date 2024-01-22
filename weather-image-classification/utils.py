import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_history(history, model_name):
    """
    Saves the history of the model to the disk.
    """
    with open(os.path.join("training_histories", f"{model_name}_history.pkl"), "wb") as file:
        pickle.dump(history, file)


def load_history_from_file(model_name):
    with open(os.path.join("training_histories", f"{model_name}_history.pkl"), "rb") as file:
        history = pickle.load(file)
    return history


def plot_training_history(history_dict, title):
    """
    Plots the training history of the model.
    """
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)

    ax1.plot(epochs, loss, label="Training loss")
    ax1.plot(epochs, val_loss, label="Validation loss")
    ax1.set_title("Training and validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, acc, label="Training accuracy")
    ax2.plot(epochs, val_acc, label="Validation accuracy")
    ax2.set_title("Training and validation accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle(title)

    plt.show()


def plot_confusion_matrix(test_labels, predictions, title, classes_names):
    """
    Plots the confusion matrix of the model.
    """
    cm = confusion_matrix(test_labels, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, ax=ax, annot=True, xticklabels=classes_names, yticklabels=classes_names, cmap="Blues")
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
