

import numpy as np
from Generate_data import generate_data_stream  # Assuming this is in a separate file
import matplotlib.pyplot as plt

class ImprovedEMAnomalyDetector:
    def __init__(self, alpha=0.3, threshold_multiplier=2, warmup_period=20, max_threshold=50):
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        self.warmup_period = warmup_period
        self.max_threshold = max_threshold
        self.ema = None
        self.mad = None
        self.anomalies = []
        self.data_points = []

    def update(self, value):
        self.data_points.append(value)

        if self.ema is None:
            self.ema = value
            return False

        # Updating EMA (Exponential Moving Averages)
        self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        # Calculating MAD (mean absolute deviation) for robust variance estimation
        if len(self.data_points) >= self.warmup_period:
            recent_points = self.data_points[-self.warmup_period:]
            mad = np.median(np.abs(recent_points - np.median(recent_points)))
            self.mad = self.alpha * mad + (1 - self.alpha) * (self.mad or mad)

            # Calculating dynamic threshold
            threshold = min(self.threshold_multiplier * self.mad, self.max_threshold)

            # Checking for anomaly
            is_anomaly = abs(value - self.ema) > threshold

            if is_anomaly:
                self.anomalies.append(value)

            return is_anomaly
        
        return False

def detect_anomalies(data_stream, detector):
    for i, value in enumerate(data_stream):
        is_anomaly = detector.update(value)
        yield value, (i, value) if is_anomaly else None


class ImprovedEMAnomalyDetector_with_recoveryPeriod:
    def __init__(self, alpha=0.3, threshold_multiplier=2, warmup_period=20, max_threshold=50, recovery_period=5):
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        self.warmup_period = warmup_period
        self.max_threshold = max_threshold
        self.recovery_period = recovery_period
        self.ema = None
        self.mad = None
        self.anomalies = []
        self.data_points = []
        self.last_anomaly_index = -float('inf')

    def update(self, value):
        self.data_points.append(value)

        if self.ema is None:
            self.ema = value
            return False

        # Updating EMA (Exponential Moving Averages)
        self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        # Calculate MAD (Median absolute deviation) for robust variance estimation
        if len(self.data_points) >= self.warmup_period:
            recent_points = self.data_points[-self.warmup_period:]
            mad = np.median(np.abs(recent_points - np.median(recent_points)))
            self.mad = self.alpha * mad + (1 - self.alpha) * (self.mad or mad)

            # Calculating dynamic threshold
            threshold = min(self.threshold_multiplier * self.mad, self.max_threshold)

            # Checking for anomaly
            is_anomaly = abs(value - self.ema) > threshold

            # Only flag as anomaly if we're not in the recovery period
            if is_anomaly and len(self.data_points) - self.last_anomaly_index > self.recovery_period:
                self.anomalies.append(value)
                self.last_anomaly_index = len(self.data_points) - 1
                return True
            
            # If in recovery period, update EMA and MAD more aggressively
            if len(self.data_points) - self.last_anomaly_index <= self.recovery_period:
                recovery_alpha = min(0.5, self.alpha * 2)  # Increase alpha, but cap it at 0.5
                self.ema = recovery_alpha * value + (1 - recovery_alpha) * self.ema
                self.mad = recovery_alpha * mad + (1 - recovery_alpha) * self.mad

        return False

def detect_anomalies(data_stream, detector):
    for i, value in enumerate(data_stream):
        is_anomaly = detector.update(value)
        yield value, (i, value) if is_anomaly else None

# Example usage and visualization
if __name__ == "__main__":
    detector = ImprovedEMAnomalyDetector_with_recoveryPeriod(alpha=0.3, threshold_multiplier=4, recovery_period=7)
    #detector = ImprovedEMAnomalyDetector(alpha=0.3, threshold_multiplier=4)
    data_stream = generate_data_stream(num_samples=200, anomaly_probability=0.05, anomaly_scale=5)
    
    data_points = []
    detected_anomalies = []
    ema_values = []
    thresholds = []
    
    for value, anomaly in detect_anomalies(data_stream, detector):
        data_points.append(value)
        ema_values.append(detector.ema)
        
        # Calculate threshold only after warmup period
        if len(data_points) >= detector.warmup_period and detector.mad is not None:
            threshold = min(detector.threshold_multiplier * detector.mad, detector.max_threshold)
        else:
            threshold = 0  # or some default value
        thresholds.append(threshold)
        
        if anomaly:
            detected_anomalies.append(anomaly)

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(data_points, label='Data Stream', alpha=0.7)
   
    plt.plot(ema_values, label='EMA', color='red')
    plt.plot([ema + thresh for ema, thresh in zip(ema_values, thresholds)], '--', color='green', label='Upper Threshold')
    plt.plot([ema - thresh for ema, thresh in zip(ema_values, thresholds)], '--', color='green', label='Lower Threshold')
    anomaly_indices, anomaly_values = zip(*detected_anomalies) if detected_anomalies else ([], [])
    plt.scatter(anomaly_indices, anomaly_values, color='purple', label='Detected Anomalies')
    plt.title("Data Stream with EMA and Dynamic Thresholds")
    plt.legend()
    
    # plt.subplot(2, 1, 2)
    # plt.plot(thresholds, label='Threshold', color='green')
    # plt.title("Dynamic Threshold Over Time")
    # plt.legend()
    
    # plt.tight_layout()
    plt.show()

    print(f"Number of data points: {len(data_points)}")
    print(f"Number of detected anomalies: {len(detected_anomalies)}")