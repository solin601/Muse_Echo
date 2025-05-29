import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, lfilter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D,
    SpatialDropout2D, Flatten, Dense, GlobalAveragePooling2D,
    Reshape, Multiply
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def se_block(input_tensor, ratio=8):
    channel_axis = -1
    channels = input_tensor.shape[channel_axis]
    se_shape = (1, 1, channels)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([input_tensor, se])
    return x

def EEGNet_modified(nb_classes, Chans=2, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = SpatialDropout2D(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)

    block2 = Activation('elu')(block2)
    # SE block 추가
    block2 = se_block(block2)

    block2 = AveragePooling2D((1, 8))(block2)
    block2 = SpatialDropout2D(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.25))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


# ---------------- Processing ----------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)


def baseline_correction(data, fs, baseline_sec=0.1):
    baseline_samples = int(baseline_sec * fs)
    baseline_mean = np.mean(data[:, :baseline_samples], axis=1, keepdims=True)  # 채널별 평균
    corrected = data - baseline_mean
    return corrected

# ---------------- Data Loading ----------------
def load_all_sessions(data_dir='backup_logs', fs=256, window_sec=0.6, channels=["TP9","TP10"]):
    X_all, y_all = [], []
    window_size = int(window_sec * fs)

    eeg_files = sorted(glob.glob(os.path.join(data_dir, '**', 'eeg_data_log_*.csv'), recursive=True))
    stim_files = sorted(glob.glob(os.path.join(data_dir, '**', 'stimulus_log_*.csv'), recursive=True))

    for eeg_file, stim_file in zip(eeg_files, stim_files):
        eeg_df = pd.read_csv(eeg_file)
        stim_df = pd.read_csv(stim_file)

        for _, stim in stim_df.iterrows():
            stim_time = stim['time']
            stim_type = stim['stimulus_type']

            segment = eeg_df[(eeg_df['timestamp'] >= stim_time) &
                             (eeg_df['timestamp'] < stim_time + window_sec)]


            if segment.shape[0] < window_size:
                continue

            data = segment[channels].values.T[:, :window_size]

            # 1. Baseline Correction
            data_baseline_corrected = baseline_correction(data, fs)

            # 2. Bandpass Filter
            filtered = np.array([bandpass_filter(ch, 1,30, fs) for ch in data_baseline_corrected])

            X_all.append(filtered)
            y_all.append(stim_type)

            X_all = np.array(X_all)
            y_all = np.array(y_all)

    return X_all, y_all

def focal_loss(gamma=1.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.mean(K.sum(weight * cross_entropy, axis=1))
    return loss_fn

# ---------------- Training and Evaluation ----------------
def train_model_with_evaluation():
    X, y = load_all_sessions()
    print(f"Total samples: {len(X)}")

    X_scaled = X[..., np.newaxis]  # (n_trials, channels, time, 1)
    # one-hot encoding
    y_cat = to_categorical(y, num_classes=2)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"class weights: {class_weights_dict}")

    model = EEGNet_modified(nb_classes=2, Chans=2, Samples=X.shape[2])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=2,
        class_weight=class_weights_dict
    )

    model.save('model.h5')

    print("\nTest Evaluation")
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_labels, target_names=["non-target", "target"]))

    cm = confusion_matrix(y_true, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["non-target", "target"],
                yticklabels=["non-target", "target"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    f1 = f1_score(y_true, y_pred_labels, average='weighted')
    print(f"F1-score (weighted): {f1:.4f}")

    auc = roc_auc_score(y_true, y_pred[:, 1])
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_labels):.4f}")

    return model

if __name__ == "__main__":
    trained_model = train_model_with_evaluation()
