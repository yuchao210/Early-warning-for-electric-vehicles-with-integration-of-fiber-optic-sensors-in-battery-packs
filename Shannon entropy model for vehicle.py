import os
import pandas as pd
import numpy as np
from dateutil.parser import parse
import math


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
        for (s, e) in segments:
            seg = x[s:e, col]
            if len(seg) == 0:
                continue
            for i in range(len(seg)):
                l = max(0, i - window_size // 2)
                r = min(len(seg), i + window_size // 2 + 1)
                smoothed[s + i, col] = np.mean(seg[l:r])

    return smoothed.ravel() if smoothed.shape[1] == 1 else smoothed


def fill_nan_inf_with_neighbors(arr):
    arr = arr.copy()
    rows, cols = arr.shape

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
                    arr[i, j] = np.nanmean(arr)
    return arr


def dataProcessing(data, diff_time_relative):
    data[(data >= 640) & (data <= 660)] = np.nan
    data = fill_nan_inf_with_neighbors(data)

    out = np.zeros_like(data)
    for i in range(data.shape[1]):
        out[:, i] = moving_average_full(
            data[:, i], diff_time_relative, window_size=15)
    return out


def shannon_entropy(x):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan

    start = math.floor(np.min(x))
    end = math.ceil(np.max(x))
    if start == end:
        return 0.0

    bins = np.arange(start, end + 1)
    freq, _ = np.histogram(x, bins=bins)
    prob = freq / np.sum(freq)
    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))


def entropy_boundary(mean_delta_T, a, b, c):
    return a * np.exp(-b * mean_delta_T) + c + 0.3


def compute_delta_temperature(temper, diff_time_relative, section_standard):
    delta_temper = np.zeros_like(temper)

    split_idx = np.where(diff_time_relative > section_standard)[0] + 1
    section_points = np.concatenate(([0], split_idx, [temper.shape[0]]))

    seg_id = 0
    base_temp = temper[0]

    for i in range(temper.shape[0]):
        if i >= section_points[seg_id + 1]:
            seg_id += 1
            base_temp = temper[section_points[seg_id]]
        delta_temper[i] = temper[i] - base_temp

    return delta_temper, section_points


def extract_temperature_rise_indices(mean_delta_T, time_relative, section_points,data_windows):
    k_data = np.zeros_like(mean_delta_T)

    for i in range(len(section_points) - 1):
        s, e = section_points[i], section_points[i + 1]
        for j in range(data_windows, e - s):
            y = mean_delta_T[s + j - data_windows:s + j]
            x = time_relative[s + j - data_windows:s + j]
            k_data[s + j] = np.polyfit(x, y, 1)[0]
        k_data[s:s + data_windows] = k_data[s + data_windows]

    state = np.where(k_data < 0, 0, 1)
    change_pts = np.where(np.diff(state) != 0)[0] + 1
    merge_pts = np.sort(np.concatenate((change_pts, section_points[1:-1])))

    segments = np.split(np.arange(len(state)), merge_pts)
    states = np.split(state, merge_pts)

    rows_temperRise = []
    for i, (seg, st) in enumerate(zip(segments, states)):
        if st[0] == 1:
            if i == 0 or mean_delta_T[seg[0]] == 0:
                rows_temperRise.extend(seg)

    return np.array(rows_temperRise)


def compute_SE_warning_for_file(file_path, section_standard=7200, data_windows=500):
    df = pd.read_csv(file_path)

    raw_time = df.iloc[:, 0]
    times = [parse(t).timestamp() for t in raw_time]
    time_relative = np.array(times) - times[0]
    diff_time_relative = np.diff(time_relative)

    raw_temper = df.iloc[:, 1:].values
    raw_temper = raw_temper[:, np.nanmean(np.isnan(raw_temper), axis=0) <= 0.9]
    temper = dataProcessing(raw_temper, diff_time_relative)

    delta_temper, section_points = compute_delta_temperature(
        temper, diff_time_relative, section_standard)

    mean_delta_T = np.nanmean(delta_temper, axis=1)

    rows_temperRise = extract_temperature_rise_indices(mean_delta_T,time_relative,section_points,data_windows)
    
    # Shannon entropy model parameters
    a, b, c = -1.419, 0.397, 2.693
    SE_warning = np.zeros(len(rows_temperRise), dtype=int)

    for i, idx in enumerate(rows_temperRise):
        SE = shannon_entropy(delta_temper[idx, :])
        SE_bound = entropy_boundary(mean_delta_T[idx], a, b, c)
        SE_warning[i] = int(SE > SE_bound)

    return SE_warning


# ======================================================
# main
# ======================================================
if __name__ == "__main__":

    folder_path = 'E:/IMPORTANT_论文数据_处理后/Test'
    data_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    SE_warning_all_files = []

    for fname in data_files:
        file_path = os.path.join(folder_path, fname)
        try:
            SE_warning = compute_SE_warning_for_file(file_path)
            SE_warning_all_files.append(SE_warning)
        except Exception:
            continue

    SE_warn = np.concatenate(SE_warning_all_files)

