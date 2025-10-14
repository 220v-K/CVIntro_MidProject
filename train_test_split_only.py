from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np

def predict_with_progress(knn, test_data, batch_size=100):
    # predict with progress, batch size is 100, tqdm is used to show the progress
    n_samples = len(test_data)
    predictions = []
    
    for i in tqdm(range(0, n_samples, batch_size), desc="  Predicting", ncols=80):
        batch_end = min(i + batch_size, n_samples)
        batch_predictions = knn.predict(test_data[i:batch_end])
        predictions.extend(batch_predictions)
    
    return np.array(predictions)

def knn_classifier_train_test_split_only(train_data, train_labels, test_data, test_labels):
    results = []
    for k in [1, 3, 5, 7, 9]:
        for distance_metric in [1, 2]:
            knn = KNeighborsClassifier(n_neighbors=k, p=distance_metric)
            knn.fit(train_data, train_labels)
            
            print("--------------------------------")
            print("knn(k={}, distance_metric={}) predicting...".format(k, distance_metric))
            result = predict_with_progress(knn, test_data, batch_size=100)
            
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
            print("test_acc: {}, precision: {}, recall: {}, f1: {}".format(test_acc, precision, recall, f1))
            print("--------------------------------")
            
    return results