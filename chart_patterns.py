import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

def find_peaks_troughs(price, distance=5, prominence=0.01):
    peaks, _ = find_peaks(price, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-price, distance=distance, prominence=prominence)
    return peaks, troughs

def detect_double_top_bottom(price, peaks, troughs, threshold=0.02):
    if len(peaks) >= 3:
        last_peaks = peaks[-3:]
        heights = price[last_peaks]
        if np.abs(heights[-1] - heights[-3]) / heights[-3] < threshold:
            return 'double_top'
    if len(troughs) >= 3:
        last_troughs = troughs[-3:]
        lows = price[last_troughs]
        if np.abs(lows[-1] - lows[-3]) / lows[-3] < threshold:
            return 'double_bottom'
    return None

def detect_head_shoulders(price, peaks, troughs, threshold=0.03):
    if len(peaks) >= 5:
        last_peaks = peaks[-5:]
        heights = price[last_peaks]
        if heights[-1] < heights[-2] and heights[-3] < heights[-2]:
            if np.abs(heights[-1] - heights[-3]) / heights[-3] < threshold:
                return 'head_shoulders'
    if len(troughs) >= 5:
        last_troughs = troughs[-5:]
        lows = price[last_troughs]
        if lows[-1] > lows[-2] and lows[-3] > lows[-2]:
            if np.abs(lows[-1] - lows[-3]) / lows[-3] < threshold:
                return 'inverse_head_shoulders'
    return None

def detect_triangle(high, low, lookback=20):
    x = np.arange(lookback).reshape(-1, 1)
    y_high = high[-lookback:].values
    y_low = low[-lookback:].values
    model_high = LinearRegression().fit(x, y_high)
    model_low = LinearRegression().fit(x, y_low)
    slope_high = model_high.coef_[0]
    slope_low = model_low.coef_[0]
    if slope_low > 0.001 and abs(slope_high) < 0.001:
        return 'ascending_triangle'
    if slope_high < -0.001 and abs(slope_low) < 0.001:
        return 'descending_triangle'
    if slope_high < -0.001 and slope_low > 0.001:
        y_high_end = model_high.predict(x[-1].reshape(1, -1))[0]
        y_low_end = model_low.predict(x[-1].reshape(1, -1))[0]
        y_high_start = model_high.predict(x[0].reshape(1, -1))[0]
        y_low_start = model_low.predict(x[0].reshape(1, -1))[0]
        if y_high_end - y_low_end < y_high_start - y_low_start:
            return 'symmetrical_triangle'
    return None

def detect_chart_patterns(df, lookback=40):
    price = df['close'].values
    high = df['high'].values
    low = df['low'].values
    peaks, troughs = find_peaks_troughs(price, distance=5, prominence=0.01)
    patterns = []
    dt = detect_double_top_bottom(price, peaks, troughs)
    if dt:
        patterns.append(dt)
    hs = detect_head_shoulders(price, peaks, troughs)
    if hs:
        patterns.append(hs)
    tri = detect_triangle(df['high'], df['low'], lookback)
    if tri:
        patterns.append(tri)
    return patterns