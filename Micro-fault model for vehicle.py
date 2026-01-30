import os
import numpy as np
import pandas as pd
from dateutil.parser import parse
from scipy.signal import detrend
from sklearn.cluster import DBSCAN
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view


def moving_average_full(x, diff_time_relative, window_size=10, segment_threshold=60):
    n = len(x)
    break_indices = np.where(diff_time_relative > segment_threshold)[0]

    segments, start = [], 0
    for idx in break_indices:
        segments.append((start, idx + 1))
        start = idx + 1
    segments.append((start, n))

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    smoothed = np.zeros_like(x, dtype=float)

    for col in range(x.shape[1]):
        for s, e in segments:
            seg = x[s:e, col]
            for i in range(len(seg)):
                l = max(0, i - window_size // 2)
                r = min(len(seg), i + window_size // 2 + 1)
                smoothed[s + i, col] = np.mean(seg[l:r])

    return smoothed.ravel() if smoothed.shape[1] == 1 else smoothed


def fill_nan_inf_with_neighbors(arr):
    arr = arr.copy()
    rows, cols = arr.shape
    global_mean = np.nanmean(arr)

    for i in range(rows):
        for j in range(cols):
            if np.isnan(arr[i, j]) or np.isinf(arr[i, j]):
                up = arr[i - 1, j] if i > 0 and not np.isnan(arr[i - 1, j]) else None
                down = arr[i + 1, j] if i < rows - 1 and not np.isnan(arr[i + 1, j]) else None

                if up is not None and down is not None:
                    arr[i, j] = (up + down) / 2
                elif up is not None:
                    arr[i, j] = up
                elif down is not None:
                    arr[i, j] = down
                else:
                    arr[i, j] = global_mean
    return arr


def data_processing(data, diff_time_relative):
    data[(data >= 640) & (data <= 660)] = np.nan
    data = fill_nan_inf_with_neighbors(data)

    out = np.zeros_like(data)
    for i in range(data.shape[1]):
        out[:, i] = moving_average_full(
            data[:, i], diff_time_relative, window_size=15
        )
    return out


def compute_variance_features(data, window_size, slide_step):
    var_list = []

    for i in range(3, data.shape[0], slide_step):
        if i <= window_size:
            window_data = data[:i, :]
        else:
            window_data = data[i - window_size:i, :]

        column_var = []
        for j in range(window_data.shape[1]):
            detrended = detrend(window_data[:, j])
            column_var.append(np.var(detrended, ddof=1))

        var_list.append(column_var)

    return np.array(var_list)


def channel_to_module(fbg_point):
    channel = int(fbg_point[fbg_point.find('T') + 1:fbg_point.find('_')])
    point = int(fbg_point[fbg_point.find('_') + 1:])
    module = int(np.ceil(channel / 2))
    return f'M{module}-{point}'


def extract_channel_indices(columns):
    channels = []
    for name in columns:
        t, u = name.find('T'), name.find('_')
        if t != -1 and u != -1:
            channels.append(int(name[t + 1:u]))
    return np.array(channels)


def detect_fault_dbscan(var_row, col_titles, min_samples_ratio):
    epsilon = 0.5 * (np.max(var_row) - np.min(var_row))
    min_samples = int(np.ceil(min_samples_ratio * len(var_row)))

    model = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = model.fit_predict(var_row.reshape(-1, 1))

    if -1 not in labels:
        return np.nan

    fault_points = col_titles[labels == -1]
    return np.array([channel_to_module(p) for p in fault_points], dtype=object)


def flatten_valid_elements(lst):
    out = []
    for item in lst:
        if isinstance(item, np.ndarray):
            out.extend(item.tolist())
        elif not (isinstance(item, float) and np.isnan(item)):
            out.append(item)
    return out


def compute_fault_frequency(data):
    valid_rows = [row for row in data if isinstance(row, list)]
    all_elements = sorted(set(e for row in valid_rows for e in row))

    counter = Counter()
    result = []

    for row in data:
        if isinstance(row, list):
            counter.update(row)
        result.append([counter.get(e, 0) for e in all_elements])

    return pd.DataFrame(result, columns=all_elements)


def compute_continuous_alarm_ratio(Fault_Warning_Frequency, window_size=20):
    fault_array = Fault_Warning_Frequency.values

    windows = sliding_window_view(fault_array, (window_size, 1))
    windows = windows[:, :, :, 0]

    ratio_array = (windows.max(axis=2) - windows.min(axis=2)) / window_size

    result = np.zeros_like(fault_array, dtype=float)
    result[window_size - 1:, :] = ratio_array

    return pd.DataFrame(result,
                        columns=Fault_Warning_Frequency.columns,
                        index=Fault_Warning_Frequency.index)


def compute_fault_for_file(file_path, window_size, slide_step, min_samples_ratio):
    df = pd.read_csv(file_path)

    times = [parse(t).timestamp() for t in df.iloc[:, 0]]
    time_relative = np.array(times) - times[0]
    diff_time_relative = np.diff(time_relative)

    raw_temper = df.iloc[:, 1:]
    raw_temper = raw_temper.loc[:, raw_temper.isna().mean() <= 0.9]
    titles = raw_temper.columns.values

    temper = data_processing(raw_temper.values, diff_time_relative)
    channels = extract_channel_indices(titles)

    idx_loc1 = np.where(np.isin(channels, np.arange(1, 36, 2)))[0]
    idx_loc2 = np.where(np.isin(channels, np.arange(2, 37, 2)))[0]

    var1 = compute_variance_features(temper[:, idx_loc1], window_size, slide_step)
    var2 = compute_variance_features(temper[:, idx_loc2], window_size, slide_step)

    faults_1, faults_2 = [], []

    for i in range(var1.shape[0]):
        faults_1.append(detect_fault_dbscan(var1[i], titles[idx_loc1], min_samples_ratio))
        faults_2.append(detect_fault_dbscan(var2[i], titles[idx_loc2], min_samples_ratio))

    common = list(set(flatten_valid_elements(faults_1)) &
                  set(flatten_valid_elements(faults_2)))

    return common if len(common) > 0 else np.nan


# ======================================================
# main
# ======================================================
if __name__ == "__main__":

    folder_path = 'E:/IMPORTANT_论文数据_处理后/Data for battery pack/Vehicle 3'
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    window_size = 600
    slide_step = 30
    min_samples_ratio = 0.9

    fault_results = []

    for fname in files:
        try:
            res = compute_fault_for_file(
                os.path.join(folder_path, fname),
                window_size, slide_step, min_samples_ratio)
            fault_results.append(res)
            print(f"{fname} Complete!")
        except Exception:
            fault_results.append(np.nan)

    fault_frequency = compute_fault_frequency(fault_results)

    final_alarm_ratio = compute_continuous_alarm_ratio(
        fault_frequency, window_size=20)
