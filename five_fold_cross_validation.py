from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
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


def knn_classifier_5_fold_cross_validation(train_data, train_labels):
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in [1, 3, 5, 7, 9]:
        for distance_metric in [1, 2]:
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []

            for train_index, valid_index in skf.split(train_data, train_labels):
                X_train, X_valid = train_data[train_index], train_data[valid_index]
                y_train, y_valid = train_labels[train_index], train_labels[valid_index]

                knn = KNeighborsClassifier(n_neighbors=k, p=distance_metric)
                knn.fit(X_train, y_train)

                predictions = predict_with_progress(knn, X_valid, batch_size=100)

                fold_accuracies.append(accuracy_score(y_valid, predictions))
                fold_precisions.append(precision_score(y_valid, predictions, average='macro'))
                fold_recalls.append(recall_score(y_valid, predictions, average='macro'))
                fold_f1s.append(f1_score(y_valid, predictions, average='macro'))

            mean_acc = float(np.mean(fold_accuracies))
            mean_precision = float(np.mean(fold_precisions))
            mean_recall = float(np.mean(fold_recalls))
            mean_f1 = float(np.mean(fold_f1s))

            results.append({
                'k': k,
                'distance_metric': distance_metric,
                'test_acc': mean_acc,
                'precision': mean_precision,
                'recall': mean_recall,
                'f1': mean_f1
            })

    return results

