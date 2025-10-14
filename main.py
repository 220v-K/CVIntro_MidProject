import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

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

def knn_classifier(train_data, train_labels, test_data, test_labels):
    results = []
    for k in [1, 3, 5, 7, 9]:
        for distance_metric in [1, 2]:
            knn = KNeighborsClassifier(n_neighbors=k, p=distance_metric)
            knn.fit(train_data, train_labels)
            result = knn.predict(test_data)
            
            # calculate accuracy
            test_acc = accuracy_score(test_labels, result)
            precision = precision_score(test_labels, result, average='macro')
            recall = recall_score(test_labels, result, average='macro')
            f1 = f1_score(test_labels, result, average='macro')
            
            results.append({
                'k': k,
                'distance_metric': distance_metric,
                'test_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            print("--------------------------------")
            print("knn(k={}, distance_metric={}) result:".format(k, distance_metric))
            print("test_acc: {}, precision: {}, recall: {}, f1: {}".format(test_acc, precision, recall, f1))
            print("--------------------------------")
            
    return results

def plot_results(results):
    # plot test acc, precision, recall, f1 with on k values.
    # plot 2 graphs for each distance metric.
    
    # Separate results by distance metric
    l1_results = [r for r in results if r['distance_metric'] == 1]
    l2_results = [r for r in results if r['distance_metric'] == 2]
    
    # Extract data for L1 distance (Manhattan)
    l1_k = [r['k'] for r in l1_results]
    l1_test_acc = [r['test_acc'] for r in l1_results]
    l1_precision = [r['precision'] for r in l1_results]
    l1_recall = [r['recall'] for r in l1_results]
    l1_f1 = [r['f1'] for r in l1_results]
    
    # Extract data for L2 distance (Euclidean)
    l2_k = [r['k'] for r in l2_results]
    l2_test_acc = [r['test_acc'] for r in l2_results]
    l2_precision = [r['precision'] for r in l2_results]
    l2_recall = [r['recall'] for r in l2_results]
    l2_f1 = [r['f1'] for r in l2_results]
    
    # Plot L1 distance results
    plt.figure(figsize=(12, 6))
    plt.plot(l1_k, l1_test_acc, marker='o', label='Test Accuracy')
    plt.plot(l1_k, l1_precision, marker='s', label='Precision')
    plt.plot(l1_k, l1_recall, marker='^', label='Recall')
    plt.plot(l1_k, l1_f1, marker='d', label='F1 Score')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Score')
    plt.title('KNN Performance with L1 Distance (Manhattan)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(l1_k)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/knn_l1_distance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/knn_l1_distance.png")
    plt.close()
    
    # Plot L2 distance results
    plt.figure(figsize=(12, 6))
    plt.plot(l2_k, l2_test_acc, marker='o', label='Test Accuracy')
    plt.plot(l2_k, l2_precision, marker='s', label='Precision')
    plt.plot(l2_k, l2_recall, marker='^', label='Recall')
    plt.plot(l2_k, l2_f1, marker='d', label='F1 Score')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Score')
    plt.title('KNN Performance with L2 Distance (Euclidean)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(l2_k)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/knn_l2_distance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/knn_l2_distance.png")
    plt.close()


def main():
    train_data, train_labels, test_data, test_labels = load_data()
    results = knn_classifier(train_data, train_labels, test_data, test_labels)
    plot_results(results)

if __name__ == '__main__':
    main()