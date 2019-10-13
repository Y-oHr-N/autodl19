import librosa
import numpy as np
import pandas as pd

# parameters
SAMPLING_FREQ = 16000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024  # 0.064 sec
FMIN = 20
FMAX = SAMPLING_FREQ // 2


def melspectrogram(
    X  # np.array: len
):
    melspec = librosa.feature.melspectrogram(
        X,
        sr=SAMPLING_FREQ,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    ).astype(np.float32)
    return melspec  # n_mel x n_frame


def logmelspectrogram(
    X  # np.array: len
):
    melspec = librosa.feature.melspectrogram(
        X,
        sr=SAMPLING_FREQ,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    ).astype(np.float32)
    
    logmelspec = librosa.power_to_db(melspec)

    return logmelspec  # n_mel x n_frame


def get_num_frame(n_sample, n_fft, n_shift):
    return np.floor(
        (n_sample / n_fft) * 2 + 1
    ).astype(np.int)


def crop_time(
    X,  # array: len
    # X,  # n_mel, n_frame
    len_sample=5,   # 1サンプル当たりの長さ[sec]
    min_sample=5,   # 切り出すサンプル数の最小個数
    max_sample=10,  # 切り出すサンプルの最大個数
):
    if len(X) < len_sample * max_sample * SAMPLING_FREQ:
        XX = X
    else:
        # TODO: 本当はシフト幅分余分にとりたい
        XX = X[:len_sample * max_sample * SAMPLING_FREQ]

    X = logmelspectrogram(XX)
    
    # len_sampleに対応するframe長
    n_frame = get_num_frame(
        n_sample=len_sample * SAMPLING_FREQ,
        n_fft=N_FFT,
        n_shift=HOP_LENGTH
    )

    # データのframe数(X.shape[1])がlen_sample * min_sampleに満たない場合のrepeat数
    n_repeat = np.ceil(n_frame * min_sample / X.shape[1]).astype(int)

    X_repeat = np.zeros(
        [X.shape[0], X.shape[1] * n_repeat],
        np.float32
    )

    for i in range(n_repeat):
        # X_noisy = add_noise(X_noisy)
        # X_repeat[:, i * n_frame: (i + 1) * n_frame] = X_noisy
        X_repeat[:, i * X.shape[1]: (i + 1) * X.shape[1]] = X

    # 最低限, min_sampleを確保
    if X.shape[1] <= n_frame * min_sample:
        n_sample = min_sample
    elif (n_frame * min_sample) < X.shape[1] <= (n_frame * max_sample):
        n_sample = (X.shape[1] // n_frame).astype(int)
    else:
        n_sample = max_sample

    # Make New log-mel spectrogram
    X_new = np.zeros(
        [X.shape[0], n_frame, n_sample],
        np.float32
    )
    for i in range(n_sample):
        X_new[:, :, i] = X_repeat[:, i * n_frame: (i + 1) * n_frame]

    return X_new


def crop_logmel(
    X,  # n_mel, n_frame
    len_sample=5,   # 1サンプル当たりの長さ[sec]
    min_sample=5,   # 切り出すサンプル数の最小個数
    max_sample=10,  # 切り出すサンプルの最大個数
):
    # len_sampleに対応するframe長
    n_frame = get_num_frame(
        n_sample=len_sample * SAMPLING_FREQ,
        n_fft=N_FFT,
        n_shift=HOP_LENGTH
    )

    # データのframe数(X.shape[1])がlen_sample * min_sampleに満たない場合のrepeat数
    n_repeat = np.ceil(n_frame * min_sample / X.shape[1]).astype(int)

    X_repeat = np.zeros(
        [X.shape[0], X.shape[1] * n_repeat],
        np.float32
    )

    for i in range(n_repeat):
        # X_noisy = add_noise(X_noisy)
        # X_repeat[:, i * n_frame: (i + 1) * n_frame] = X_noisy
        X_repeat[:, i * X.shape[1]: (i + 1) * X.shape[1]] = X

    # 最低限, min_sampleを確保
    if X.shape[1] <= n_frame * min_sample:
        n_sample = min_sample
    elif (n_frame * min_sample) < X.shape[1] <= (n_frame * max_sample):
        n_sample = (X.shape[1] // n_frame).astype(int)
    else:
        n_sample = max_sample

    # Make New log-mel spectrogram
    X_new = np.zeros(
        [X.shape[0], n_frame, n_sample],
        np.float32
    )
    for i in range(n_sample):
        X_new[:, :, i] = X_repeat[:, i * n_frame: (i + 1) * n_frame]

    return X_new


def make_cropped_dataset_5sec(
    X_list,  # list(array):
    y_list=None,  # array: n_sample x dim_label
    len_sample=5,   # 1サンプル当たりの長さ[sec]
    min_sample=1,   # 切り出すサンプル数の最小個数
    max_sample=1,  # 切り出すサンプルの最大個数
):
    # さしあたりmin_sample == max_sample == 1
    # -> y_results.shape == (len(X_list), dim_label) かつ，len(X_results) == len(X_list)
    X_results = []
    # y_results = np.zeros_like(y_list)
    if y_list is not None:
        y_results = np.zeros(
            [len(X_list), y_list.shape[1]],
            np.float32
        )

    for i in range(len(X_list)):
        # logmels: n_mel, n_frame x n_sample
        logmels = crop_time(
            X_list[i],
            len_sample=len_sample,
            min_sample=min_sample,
            max_sample=max_sample
        )
        X_results.append(logmels[:, :, 0].transpose())
        if y_list is not None:
            y_results[i, :] = y_list[i, :]

    if y_list is None:
        y_results = None

    return X_results, y_results


def describe(train_x, train_y):
    """Descrive train data.
    """
    info = pd.DataFrame({
        'len': list(map(lambda x: len(x), train_x)),
        'label': np.argmax(train_y, axis=1)
    })
    
    print('*' * 10, '全体','*' * 10)
    print('クラス数:{}'.format(info['label'].max() + 1))
    print('平均サンプル長: {} sample({} sec)'.format(info['len'].mean(), info['len'].mean() / SAMPLING_FREQ))
    print('最大サンプル長: {} sample({} sec)'.format(info['len'].max(), info['len'].max() / SAMPLING_FREQ))
    print('最小サンプル長: {} sample({} sec)'.format(info['len'].min(), info['len'].min() / SAMPLING_FREQ))
    
    print('*' * 10, 'ラベル単位','*' * 10)
    df = info.groupby('label') \
    .agg(['count', 'mean', 'max', 'min', 'sum']) \
    .droplevel(0, axis=1) \
    .rename({
        'count': 'num_sample', 
        'mean': 'len_mean[sec]',
        'max': 'len_max[sec]',
        'min': 'len_min[sec]',
        'sum': 'len_total[sec]'
    }, axis=1)
    df.loc[:, ['len_mean[sec]', 'len_max[sec]', 'len_min[sec]', 'len_total[sec]']] = \
        df.loc[:, ['len_mean[sec]', 'len_max[sec]', 'len_min[sec]', 'len_total[sec]']] / SAMPLING_FREQ
    print(df)