# Cálculo de métricas e resumo de resultados

import numpy as np

def accuracy(y_true, y_pred):
    """Calcula a acurácia: proporção de acertos."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """Calcula a precisão macro (média entre as classes)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(prec)
    return np.mean(precisions)


def recall(y_true, y_pred):
    """Calcula o Recall (Sensibilidade) macro (média entre classes)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)
    return np.mean(recalls)


def f1_score(y_true, y_pred):
    """Calcula o F1-Score macro (média entre as classes)."""
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
    """
    Gera tabela de resultados em formato CSV, no padrão pedido pelo enunciado.
    results: lista de (nome_modelo, dict com métricas e desvios)
    """
    header = [
        "Classificador",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-Score",
        "Tempo Treino (s)",
        "Tempo Teste (s)"
    ]

    lines = [",".join(header) + "\n"]

    for name, res in results:
        line = (
            f"{name},"
            f"{res['acc_mean']:.2f} ± {res['acc_std']:.2f},"
            f"{res['prec_mean']:.2f} ± {res['prec_std']:.2f},"
            f"{res['rec_mean']:.2f} ± {res['rec_std']:.2f},"
            f"{res['f1_mean']:.2f} ± {res['f1_std']:.2f},"
            f"{res['train_mean']:.2f} ± {res['train_std']:.2f},"
            f"{res['test_mean']:.2f} ± {res['test_std']:.2f}\n"
        )
        lines.append(line)

    with open(out_path.replace(".txt", ".csv"), "w", encoding="utf-8") as f:
        f.writelines(lines)
