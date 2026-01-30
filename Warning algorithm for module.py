import os
import math
import numpy as np
import pandas as pd
from dateutil.parser import parse


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    raw_time = data.iloc[:, 0]
    time_list = []
    invalid_idx = []

    for i, t in enumerate(raw_time):
        try:
            parsed = parse(t)
            time_list.append(parsed)
        except Exception:
            invalid_idx.append(i)

    time_series = np.array([t.timestamp() for t in time_list])
    time_relative = time_series - time_series[0]
    delta_time = np.diff(time_relative)

    temperature_data = data.iloc[:, 1:]
    if invalid_idx:
        temperature_data.drop(index=invalid_idx, inplace=True)

    temperature_data = temperature_data.loc[
        :, temperature_data.isna().mean() <= 0.9
    ].values

    return time_relative, delta_time, temperature_data


def fill_nan_inf(data):
    filled = data.copy()
    rows, cols = filled.shape
    global_mean = np.nanmean(filled)

    for i in range(rows):
        for j in range(cols):
            if np.isnan(filled[i, j]) or np.isinf(filled[i, j]):
                up = filled[i - 1, j] if i > 0 and np.isfinite(filled[i - 1, j]) else None
                down = filled[i + 1, j] if i < rows - 1 and np.isfinite(filled[i + 1, j]) else None

                if up is not None and down is not None:
                    filled[i, j] = 0.5 * (up + down)
                elif up is not None:
                    filled[i, j] = up
                elif down is not None:
                    filled[i, j] = down
                else:
                    filled[i, j] = global_mean
    return filled


def shannon_entropy(data):
    if np.all(np.isnan(data)):
        return np.nan

    min_val = math.floor(np.nanmin(data))
    max_val = math.ceil(np.nanmax(data))

    if max_val <= min_val:
        return 0

    bins = np.linspace(min_val, max_val, max_val - min_val + 1)
    freq, _ = np.histogram(data, bins=bins)

    prob = freq / np.sum(freq)
    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))


def entropy_boundary(mean_delta_T, a, b, c):
    return a * np.exp(-b * mean_delta_T) + c + 0.3


# =============================================================================
# Entropy-based warning model
# =============================================================================
def entropy_warning_model(temperature_data, delta_time, sliding_window=500, section_interval=7200):
    
    num_samples, num_points = temperature_data.shape
    temperature_data = fill_nan_inf(temperature_data)
    delta_temperature = np.zeros_like(temperature_data)

    time_gap_idx = np.where(np.abs(delta_time) > section_interval)[0] + 1
    section_points = np.concatenate(([0], time_gap_idx, [num_samples]))

    segment_id = 0
    reference_temp = temperature_data[0]

    for i in range(num_samples):
        if segment_id + 1 < len(section_points) and i >= section_points[segment_id + 1]:
            segment_id += 1
            reference_temp = temperature_data[section_points[segment_id]]
        delta_temperature[i] = temperature_data[i] - reference_temp

    mean_delta_T = np.nanmean(delta_temperature, axis=1)

    # Shannon entropy boundary parameters
    a, b, c = -1.419, 0.397, 2.693

    SE_warning = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):

        if i < sliding_window:
            slope_positive = True
        else:
            y = mean_delta_T[i - sliding_window:i]
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            slope_positive = slope > 0

        if slope_positive:
            SE = shannon_entropy(delta_temperature[i, :])
            SE_bound = entropy_boundary(mean_delta_T[i], a, b, c)
            SE_warning[i] = int(SE > SE_bound)
        else:
            SE_warning[i] = 0

    return SE_warning


if __name__ == "__main__":

    DATA_FOLDER = 'E:/IMPORTANT_论文数据_处理后/Test'
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(DATA_FOLDER, file)

        time_rel, delta_t, temp_data = load_and_preprocess_data(file_path)

        SE_warn = entropy_warning_model(temp_data, delta_t, sliding_window=500, section_interval=7200)
