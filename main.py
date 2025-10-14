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
from train_valid_test_split import knn_classifier_train_valid_test_split, evaluate_with_best_k_and_distance_metric
from five_fold_cross_validation import knn_classifier_5_fold_cross_validation

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
    
    return train_data, train_labels, test_data, test_labels

def load_data_train_valid_test():
    # valid = 10% of train
    train_data, train_labels, test_data, test_labels = load_data()
    valid_data = train_data[:5000]
    valid_labels = train_labels[:5000]
    train_data = train_data[5000:]
    train_labels = train_labels[5000:]
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


def save_results_to_txt(results, filepath='results/knn_results.txt', test_results=None):
    ensure_results_dir(os.path.dirname(filepath))
    lines = ['k,distance_metric,test_acc,precision,recall,f1']
    for r in sorted(results, key=lambda r: (r['distance_metric'], r['k'])):
        lines.append(f"{r['k']},{r['distance_metric']},{r['test_acc']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}")
    
    # 최고 성능을 달성한 k와 distance metric 찾기
    best_result = max(results, key=lambda x: x['test_acc'])
    lines.append("")
    lines.append(f"Best validation performance:")
    lines.append(f"k={best_result['k']}, distance_metric={best_result['distance_metric']}, validation_acc={best_result['test_acc']:.4f}")
    
    # 테스트 결과 추가
    if test_results:
        lines.append("")
        lines.append("Test results with best k and distance_metric:")
        lines.append(f"Test Accuracy: {test_results['test_acc']:.4f}, Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}, F1: {test_results['f1']:.4f}")
    
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {filepath}")
    
def run_knn(case: str):  # train_test_split_only, train_valid_test_split, 5-fold cross validation
    if case == 'train_test':
        # train_test_split_only
        train_data, train_labels, test_data, test_labels = load_data()
        results = knn_classifier_train_test_split_only(train_data, train_labels, test_data, test_labels)
        save_results_to_txt(results, 'results/train_test_split_only/knn_results.txt')
        plot_results(results, 'results/train_test_split_only')
    
    elif case == 'train_valid_test':
        # train_valid_test_split
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_data_train_valid_test()
        results = knn_classifier_train_valid_test_split(train_data, train_labels, valid_data, valid_labels)
        plot_results(results, 'results/train_valid_test_split')
        test_results = evaluate_with_best_k_and_distance_metric(results, train_data, train_labels, test_data, test_labels)
        
        save_results_to_txt(results, 'results/train_valid_test_split/knn_results.txt', test_results)
        print(f"Test Accuracy: {test_results['test_acc']:.4f}, Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}, F1: {test_results['f1']:.4f}")

    elif case == '5-fold':
        # 5-fold cross validation
        train_data, train_labels, test_data, test_labels = load_data()
        results = knn_classifier_5_fold_cross_validation(train_data, train_labels)
        plot_results(results, 'results/5-fold_cross_validation')
        test_results = evaluate_with_best_k_and_distance_metric(results, train_data, train_labels, test_data, test_labels)
        save_results_to_txt(results, 'results/5-fold_cross_validation/knn_results.txt', test_results)
        print(f"Test Accuracy: {test_results['test_acc']:.4f}, Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}, F1: {test_results['f1']:.4f}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="KNN experiment runner")
    parser.add_argument('--classifier', type=str, default='train_test', choices=['train_test', 'train_valid_test', '5-fold'],
                        help='Choose evaluation scheme: train_test, train_valid_test, or 5-fold')
    args = parser.parse_args()
    run_knn(args.classifier)
    

if __name__ == '__main__':
    main()