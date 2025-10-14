#!/usr/bin/env python3
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm  
import os
from plot import plot_results, ensure_results_dir
import math
from train_test_split_only import knn_classifier_train_test_split_only

def l1_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))

def l2_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(file_path):
    batch = unpickle(file_path)
    data = batch[b'data']  # (length, 3072) shape
    labels = batch[b'labels']  # (length,) shape
    
    return data, np.array(labels)

def load_data():    
    # load train data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
        train_data.append(data)
        train_labels.append(labels)
        print(f"Loaded batch {i} train data")

    train_data = np.vstack(train_data)  # (50000, 3072)
    train_labels = np.hstack(train_labels)  # (50000,)

    # load test data
    test_data, test_labels = load_cifar10_batch('cifar-10-batches-py/test_batch')
    print("Loaded test data")
    
    return train_data[:500], train_labels[:500], test_data[:100], test_labels[:100]

def save_results_to_txt(results, filepath='results/knn_results.txt'):
    ensure_results_dir(os.path.dirname(filepath))
    lines = ['k,distance_metric,test_acc,precision,recall,f1']
    for r in sorted(results, key=lambda r: (r['distance_metric'], r['k'])):
        lines.append(f"{r['k']},{r['distance_metric']},{r['test_acc']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}")
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {filepath}")

def main():
    train_data, train_labels, test_data, test_labels = load_data()
    results = knn_classifier_train_test_split_only(train_data, train_labels, test_data, test_labels)
    save_results_to_txt(results, 'results/train_test_split_only/knn_results.txt')
    plot_results(results, 'results/train_test_split_only')

if __name__ == '__main__':
    main()