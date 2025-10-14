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

def knn_classifier_train_valid_test_split(train_data, train_labels, valid_data, valid_labels):
    results = []
    for k in [1, 3, 5, 7, 9]:
        for distance_metric in [1, 2]:
            knn = KNeighborsClassifier(n_neighbors=k, p=distance_metric)
            knn.fit(train_data, train_labels)
            
            print("--------------------------------")
            print("knn(k={}, distance_metric={}) predicting...".format(k, distance_metric))
            result = predict_with_progress(knn, valid_data, batch_size=100)
            
            # calculate accuracy
            test_acc = accuracy_score(valid_labels, result)
            precision = precision_score(valid_labels, result, average='macro')
            recall = recall_score(valid_labels, result, average='macro')
            f1 = f1_score(valid_labels, result, average='macro')
            
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

def evaluate_with_best_k_and_distance_metric(results, train_data, train_labels, test_data, test_labels):
    # 가장 test acc가 높은 k, distance metric를 찾아서, 그 모델로 test data에 대해 eval.
    best_result = max(results, key=lambda x: x['test_acc'])
    knn = KNeighborsClassifier(n_neighbors=best_result['k'], p=best_result['distance_metric'])
    knn.fit(train_data, train_labels) # train_data, train_labels는 이미 있음
    result = predict_with_progress(knn, test_data, batch_size=100)
    
    test_acc = accuracy_score(test_labels, result)
    precision = precision_score(test_labels, result, average='macro')
    recall = recall_score(test_labels, result, average='macro')
    f1 = f1_score(test_labels, result, average='macro')
    
    test_results = {
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return test_results