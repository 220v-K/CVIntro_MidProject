import matplotlib.pyplot as plt
import os
import math


def ensure_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)


def plot_results(results, results_dir):
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
    
    ensure_results_dir(results_dir)

    def round_ylim(min_val, max_val):
        min_rounded = math.floor(min_val / 0.05) * 0.05
        max_rounded = (math.ceil(max_val / 0.05)+1) * 0.05
        # Ensure min and max are within [0, 1]
        return max(min_rounded, 0), min(max_rounded, 1)

    def annotate_all_top(ax, k_list, series_list, series_names, y_offset=0.015):
        # series_list: list of y value lists
        # series_names: list of str
        num_k = len(k_list)
        num_series = len(series_list)
        max_y_per_k = []
        for idx in range(num_k):
            col = [series_list[j][idx] for j in range(num_series)]
            max_y_per_k.append(max(col))
        ann_y_top = max(max_y_per_k)
        for i, k in enumerate(k_list):
            texts = []
            for sidx, y in enumerate([series[sidx][i] for series, sidx in zip([series_list]*num_series, range(num_series))]):
                txt = f"{series_names[sidx]}: {y:.3f}"
                texts.append(txt)
            # Stack the annotations vertically above the corresponding x at the top end, not to overlap with data points
            for t_idx, txt in enumerate(texts):
                ax.annotate(
                    txt,
                    (k, max_y_per_k[i] + y_offset*(t_idx+1)*2),
                    ha='center',
                    fontsize=9,
                    color=['C0','C1','C2','C3'][t_idx], # match plot colors
                    fontweight='bold',
                    alpha=0.95
                )

    # L1 distance plot
    l1_all = [l1_test_acc, l1_precision, l1_recall, l1_f1]
    l1_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
    l1_yvals = [y for series in l1_all for y in series]
    l1_min, l1_max = min(l1_yvals), max(l1_yvals)
    l1_ylim = round_ylim(l1_min, l1_max)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(l1_k, l1_test_acc, marker='o', label='Test Accuracy')
    ax.plot(l1_k, l1_precision, marker='s', label='Precision')
    ax.plot(l1_k, l1_recall, marker='^', label='Recall')
    ax.plot(l1_k, l1_f1, marker='d', label='F1 Score')
    annotate_all_top(ax, l1_k, l1_all, l1_labels, y_offset=0.012 * (l1_ylim[1] - l1_ylim[0]))
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('Score')
    ax.set_title('KNN Performance with L1 Distance (Manhattan)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(l1_k)
    ax.set_ylim([l1_ylim[0], l1_ylim[1]])
    fig.tight_layout()
    plt.savefig(f'{results_dir}/knn_l1_distance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/knn_l1_distance.png")
    plt.close(fig)

    # L2 distance plot
    l2_all = [l2_test_acc, l2_precision, l2_recall, l2_f1]
    l2_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
    l2_yvals = [y for series in l2_all for y in series]
    l2_min, l2_max = min(l2_yvals), max(l2_yvals)
    l2_ylim = round_ylim(l2_min, l2_max)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(l2_k, l2_test_acc, marker='o', label='Test Accuracy')
    ax.plot(l2_k, l2_precision, marker='s', label='Precision')
    ax.plot(l2_k, l2_recall, marker='^', label='Recall')
    ax.plot(l2_k, l2_f1, marker='d', label='F1 Score')
    annotate_all_top(ax, l2_k, l2_all, l2_labels, y_offset=0.012 * (l2_ylim[1] - l2_ylim[0]))
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('Score')
    ax.set_title('KNN Performance with L2 Distance (Euclidean)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(l2_k)
    ax.set_ylim([l2_ylim[0], l2_ylim[1]])
    fig.tight_layout()
    plt.savefig(f'{results_dir}/knn_l2_distance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/knn_l2_distance.png")
    plt.close(fig)

