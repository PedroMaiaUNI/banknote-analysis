import numpy as np
import time
from utils import load_data, normalize
from evaluation_wrapper import accuracy, precision, recall, f1_score, evaluate_model_with_data
from evaluation import summarize_results
from k_nearest_neighbors import KNN
from bayes_univariado import NaiveBayesUnivariado
from bayes_multivariado import BayesMultivariado
from cross_validation import train_test_split

def main():
    X, y = load_data("data/data_banknote_authentication.txt")
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("KNN (Classificação)", KNN(k=5, task='classification')),
        ("KNN (Regressão)", KNN(k=5, task='regression')),
        ("Bayes Univariado", NaiveBayesUnivariado()),
        ("Bayes Multivariado", BayesMultivariado()),
    ]

    results = []
    for name, model in models:
        print(f"\nTreinando e testando {name}...")

        # medir tempo
        t0 = time.time()
        model.fit(X_train, y_train)
        t1 = time.time()
        y_pred = model.predict(X_test)
        t2 = time.time()

        # calcular métricas
        if hasattr(model, "task") and model.task == "regression":
            acc = prec = rec = f1 = np.nan  # não aplicável
        else:
            acc = accuracy(y_test, y_pred)
            prec = precision(y_test, y_pred)
            rec = recall(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)   
        train_time = t1 - t0
        test_time = t2 - t1

        res = {
            "acc_mean": acc, "acc_std": 0.0,
            "prec_mean": prec, "prec_std": 0.0,
            "rec_mean": rec, "rec_std": 0.0,
            "f1_mean": f1, "f1_std": 0.0,
            "train_mean": train_time, "train_std": 0.0,
            "test_mean": test_time, "test_std": 0.0
        }

        results.append((name, res))

    summarize_results(results, "results/results_table.csv")
    print("Resultados salvos em results/results_table.csv")

if __name__ == '__main__':
    main()