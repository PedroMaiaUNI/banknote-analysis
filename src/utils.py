# Funções auxiliares: leitura, normalização, etc.

import numpy as np

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 5:
                continue
            nums = [float(x) for x in parts[:4]]
            label = int(float(parts[4]))
            data.append(nums + [label])
    arr = np.array(data)
    X = arr[:, :4]
    y = arr[:, 4].astype(int)
    return X, y

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_replaced = np.where(std == 0, 1.0, std)
    return (X - mean) / std_replaced