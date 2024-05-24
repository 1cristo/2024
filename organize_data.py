# organize_data.py se ejecutara tercero

import os
import numpy as np

def organize_data(base_dir):
    X = []
    y = []
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for seq in os.listdir(class_dir):
            seq_dir = os.path.join(class_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            keypoints_seq = []
            for frame in os.listdir(seq_dir):
                if frame.endswith('.npy'):
                    keypoints = np.load(os.path.join(seq_dir, frame), allow_pickle=True)
                    keypoints_seq.append(keypoints)
            if keypoints_seq:  # Solo agrega secuencias que contienen datos
                X.append(keypoints_seq)
                y.append(class_name)
    return X, y  # Devolvemos listas en lugar de numpy arrays

if __name__ == "__main__":
    data_dir = 'sign_language_dataset'
    X, y = organize_data(data_dir)
    # Guardamos como numpy arrays con dtype=object para manejar secuencias de longitud variable
    np.save('X.npy', np.array(X, dtype=object))
    np.save('y.npy', np.array(y, dtype=object))
