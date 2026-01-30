import os
import numpy as np
import pandas as pd
from dateutil.parser import parse


def fill_invalid_values(data):
    filled = data.copy()
    rows, cols = filled.shape

    for i in range(rows):
        for j in range(cols):
            if not np.isfinite(filled[i, j]):
                upper = filled[i - 1, j] if i > 0 and np.isfinite(filled[i - 1, j]) else None
                lower = filled[i + 1, j] if i < rows - 1 and np.isfinite(filled[i + 1, j]) else None

                if upper is not None and lower is not None:
                    filled[i, j] = 0.5 * (upper + lower)
                elif upper is not None:
                    filled[i, j] = upper
                elif lower is not None:
                    filled[i, j] = lower
                else:
                    filled[i, j] = np.nanmean(data)

    return filled


def moving_average_segmented(data, delta_time, window_size=15, segment_threshold=60):
    n = data.shape[0]
    break_idx = np.where(delta_time > segment_threshold)[0]

    segments, start = [], 0
    for idx in break_idx:
        segments.append((start, idx + 1))
        start = idx + 1
    segments.append((start, n))

    smoothed = np.zeros_like(data, dtype=float)

    for ch in range(data.shape[1]):
        for s, e in segments:
            segment = data[s:e, ch]
            if len(segment) == 0:
                continue

            for i in range(len(segment)):
                left = max(0, i - window_size // 2)
                right = min(len(segment), i + window_size // 2 + 1)
                smoothed[s + i, ch] = np.mean(segment[left:right])

    return smoothed


def preprocess_temperature_data(raw_data, delta_time):
    # Remove abnormal values
    raw_data[(raw_data >= 640) & (raw_data <= 660)] = np.nan

    filled = fill_invalid_values(raw_data)
    smoothed = moving_average_segmented(filled, delta_time)

    return smoothed


def segmented_temperature_rate(temperature, time_relative,segment_threshold=60):
    n = temperature.shape[0]
    rate = np.zeros_like(temperature)

    break_idx = np.where(np.diff(time_relative) > segment_threshold)[0]

    segments, start = [], 0
    for idx in break_idx:
        segments.append((start, idx + 1))
        start = idx + 1
    segments.append((start, n))

    for ch in range(temperature.shape[1]):
        for s, e in segments:
            if e - s < 2:
                continue

            dT = np.diff(temperature[s:e, ch])
            dt = np.diff(time_relative[s:e])

            seg_rate = np.divide(dT, dt, out=np.zeros_like(dT),where=dt != 0)

            rate[s:e - 1, ch] = seg_rate

    return rate


def main(data_folder):
    T_THRESHOLD = 100
    TR_THRESHOLD = 1 

    records = []

    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder, file_name)

        try:
            df = pd.read_csv(file_path)

            # ---------------- Original time ----------------
            raw_time = df.iloc[:, 0]
            parsed_time = [parse(t) for t in raw_time]

            time_seconds = np.array([t.timestamp() for t in parsed_time])
            time_relative = time_seconds - time_seconds[0]
            delta_time = np.diff(time_relative)

            # ---------------- Temperature data ----------------
            temp_raw = df.iloc[:, 1:]
            temp_raw = temp_raw.dropna(axis=1, thresh=0.1 * len(temp_raw))
            temp_raw = temp_raw.values

            temp_processed = preprocess_temperature_data(temp_raw, delta_time)
            temp_rate = segmented_temperature_rate(temp_processed, time_relative)

            # ---------------- Per-sample threshold decision ----------------
            for i in range(len(raw_time)):
                T_i = np.nanmax(temp_processed[i, :])
                TR_i = np.nanmax(temp_rate[i, :])

                warning_flag = int((T_i > T_THRESHOLD) or(TR_i > TR_THRESHOLD))

                records.append([raw_time.iloc[i], warning_flag])

        except Exception:
            continue   

    return pd.DataFrame(records,columns=["time", "warning_flag"])


# ======================================================
# main
# ======================================================
if __name__ == "__main__":
    DATA_FOLDER = "E:/IMPORTANT_论文数据_处理后/Data for battery pack/Vehicle 3"   
    output = main(DATA_FOLDER)
    output.to_csv("warning results.csv", index=False)
