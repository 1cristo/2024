# split_data.py se ejecutara cuarto
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter

def filter_classes(X, y, min_count=2):
    # Contar ejemplos por clase
    counter = Counter(y)
    # Filtrar clases con menos de min_count ejemplos
    filtered_X = []
    filtered_y = []
    for i, label in enumerate(y):
        if counter[label] >= min_count:
            filtered_X.append(X[i])
            filtered_y.append(y[i])
    return filtered_X, filtered_y

def split_data(X, y, base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    X_filtered, y_filtered = filter_classes(X, y)
    
    if len(y_filtered) == 0:
        raise ValueError("No hay suficientes datos después de filtrar las clases con menos de 2 ejemplos.")
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_filtered, y_filtered, test_size=(1 - train_ratio), stratify=y_filtered)
    
    if len(y_train) == 0 or len(y_temp) == 0:
        raise ValueError("No hay suficientes datos para dividir en conjuntos de entrenamiento y prueba.")
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=y_temp)

    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for dataset_type, (X_data, y_data) in datasets.items():
        dataset_dir = os.path.join(base_dir, dataset_type)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        for i, (data, label) in enumerate(zip(X_data, y_data)):
            class_dir = os.path.join(dataset_dir, label)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            seq_dir = os.path.join(class_dir, f'sample_{i + 1}')
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            
            for j, frame in enumerate(data):
                np.save(os.path.join(seq_dir, f'{j + 1}.npy'), frame)

if __name__ == "__main__":
    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    
    try:
        split_data(X, y, base_dir='sign_language_dataset')
    except ValueError as e:
        print(e)
