# Cálculo de métricas e resumo de resultados

import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
    return np.mean(precisions)

def recall(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec = tp / (tp + fn)
        recalls.append(rec)
    return np.mean(recalls)

def f1_score(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)

def summarize_results(results, out_path):
    lines = []
    header = "Classificador,Acurácia,Precisão,Recall,F1-Score,Tempo Treino (s),Tempo Teste (s)\n"
    lines.append(header)
    for name, res in results:
        line = (
            f"{name},"
            f"{res['acc_mean']:.4f},"
            f"{res['prec_mean']:.4f},"
            f"{res['rec_mean']:.4f},"
            f"{res['f1_mean']:.4f},"
            f"{res['train_mean']:.4f},"
            f"{res['test_mean']:.4f}\n"
        )
        lines.append(line)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)