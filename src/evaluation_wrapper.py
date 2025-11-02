import numpy as np, time
from evaluation import accuracy, precision, recall, f1_score

def evaluate_model_with_data(model, X, y, folds):
    accs, precs, recs, f1s = [], [], [], []
    train_times, test_times = [], []
    for train_idx, test_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        t0 = time.time()
        model.fit(X_train, y_train)
        t1 = time.time()
        y_pred = model.predict(X_test)
        t2 = time.time()

        train_times.append(t1 - t0)
        test_times.append(t2 - t1)

        accs.append(accuracy(y_test, y_pred))
        precs.append(precision(y_test, y_pred))
        recs.append(recall(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    res = {
        'accs': accs, 'precs': precs, 'f1s': f1s,
        'train_times': train_times, 'test_times': test_times,
        'acc_mean': np.mean(accs), 'acc_std': np.std(accs, ddof=1),
        'prec_mean': np.mean(precs), 'prec_std': np.std(precs, ddof=1),
        'rec_mean': np.mean(recs), 'rec_std': np.std(recs, ddof=1),
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s, ddof=1),
        'train_mean': np.mean(train_times), 'train_std': np.std(train_times, ddof=1),
        'test_mean': np.mean(test_times), 'test_std': np.std(test_times, ddof=1),
    }
    return res
