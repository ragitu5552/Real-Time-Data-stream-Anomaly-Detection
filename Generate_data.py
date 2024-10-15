

# # import numpy as np
# # import random
# # import time
# # import matplotlib.pyplot as plt

# # def generate_data_stream(num_samples=100, mean=0, std_deviation=1, noise_level=0.5):
# #     for t in range(num_samples):
# #         # Regular pattern (e.g., linear trend)
# #         trend = 0.05 * t  # Linear increase over time

# #         # Seasonal pattern (e.g., sinusoidal)
# #         seasonality = 10 * np.sin(2 * np.pi * t / 30)  # Monthly cycle (period of 30)

# #         # Generate base data point
# #         base_data = mean + trend + seasonality

# #         # Add random noise
# #         noise = random.uniform(-noise_level, noise_level)
# #         data_point = base_data + noise

# #         yield data_point

# #         # Simulate real-time data streaming
# #         time.sleep(0.1)  # Adjust as necessary

# # # Example usage
# # if __name__ == "__main__":
# #     l = []
# #     for data in generate_data_stream(num_samples=100):
# #         l.append(data)
        
# #         # print(data)
# #     plt.plot(l)
# #     plt.show()

# import numpy as np
# import random
# import time
# import matplotlib.pyplot as plt

# def generate_data_stream(num_samples=50, mean=0, std_deviation=1, noise_level=0.5,
#                          trend_factor=0.5, seasonality_amplitude=10, seasonality_period=30,
#                          anomaly_probability=0.10, anomaly_scale=3):
#     for t in range(num_samples):
#         # Regular pattern (linear trend)
#         trend = trend_factor * t

#         # Seasonal pattern (sinusoidal)
#         seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)

#         # Generate base data point with some randomness
#         base_data = np.random.normal(mean, std_deviation) + trend + seasonality

#         # Add random noise
#         noise = random.uniform(-noise_level, noise_level)
#         data_point = base_data + noise

#         # Occasionally inject anomalies
#         if random.random() < anomaly_probability:
#             data_point *= anomaly_scale

#         yield data_point

#         # Simulate real-time data streaming
#         time.sleep(0.1)  # Adjust as necessary

# # Example usage
# if __name__ == "__main__":
#     data_points = list(generate_data_stream(num_samples=50))
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(data_points)
#     plt.title("Flexible Data Stream")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.show()

#     # Print some statistics
#     print(f"Mean: {np.mean(data_points):.2f}")
#     print(f"Standard Deviation: {np.std(data_points):.2f}")
#     print(f"Min: {np.min(data_points):.2f}")
#     print(f"Max: {np.max(data_points):.2f}")

# import numpy as np
# import random
# import time
import matplotlib.pyplot as plt

# def generate_data_stream(num_samples=100, mean=0, std_deviation=1, noise_level=0.5,
#                          trend_factor=0.5, seasonality_amplitude=10, seasonality_period=30,
#                          anomaly_probability=0.10, anomaly_scale=3):
#     data_points = []
#     trend_component = []
#     seasonal_component = []
#     noise_component = []
#     anomaly_component = []

#     for t in range(num_samples):
#         # Regular pattern (linear trend)
#         trend = trend_factor * t
#         trend_component.append(trend)

#         # Seasonal pattern (sinusoidal)
#         seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
#         seasonal_component.append(seasonality)

#         # Generate base data point with some randomness
#         base_data = np.random.normal(mean, std_deviation) + trend + seasonality

#         # Add random noise
#         noise = random.uniform(-noise_level, noise_level)
#         noise_component.append(noise)

#         data_point = base_data + noise

#         # Occasionally inject anomalies
#         if random.random() < anomaly_probability:
#             anomaly = data_point * (anomaly_scale - 1)
#             data_point *= anomaly_scale
#             anomaly_component.append(anomaly)
#         else:
#             anomaly_component.append(0)

#         data_points.append(data_point)

#         # Simulate real-time data streaming
#         time.sleep(0.01)  # Reduced for quicker execution

#     return data_points, trend_component, seasonal_component, noise_component, anomaly_component

# # Example usage
# if __name__ == "__main__":
#     data_points, trend, seasonality, noise, anomalies = generate_data_stream()
    
#     plt.figure(figsize=(15, 10))

#     plt.subplot(3, 1, 1)
#     plt.plot(data_points, label='Data Stream')
#     plt.plot(trend, label='Trend', linestyle='--')
#     plt.title("Data Stream with Trend")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(3, 1, 2)
#     plt.plot(seasonality, label='Seasonality')
#     plt.plot(noise, label='Noise')
#     plt.title("Seasonality and Noise Components")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(3, 1, 3)
#     plt.stem(anomalies, label='Anomalies')
#     plt.title("Anomaly Component")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

#     # Print some statistics
#     print(f"Mean: {np.mean(data_points):.2f}")
#     print(f"Standard Deviation: {np.std(data_points):.2f}")
#     print(f"Min: {np.min(data_points):.2f}")
#     print(f"Max: {np.max(data_points):.2f}")

import numpy as np
import random
import time

def generate_data_stream(num_samples=200, mean=0, std_deviation=1, noise_level=0.5,
                         trend_factor=0.5, seasonality_amplitude=10, seasonality_period=30,
                         anomaly_probability=0.05, anomaly_scale=1.2):
    data_points = []

    for t in range(num_samples):
        # Regular pattern (linear trend)
        trend = trend_factor * t
        
        # Seasonal pattern (sinusoidal)
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        
        # Generate base data point with randomness
        base_data = np.random.normal(mean, std_deviation) + trend + seasonality
        
        # Add random noise
        noise = random.uniform(-noise_level, noise_level)
        
        data_point = base_data + noise

        #Occasionally inject anomalies
        if random.random() < anomaly_probability:
            data_point *= anomaly_scale  # Scale the data point to create an anomaly

        data_points.append(data_point)
        
        # Simulate real-time data streaming
        time.sleep(0.01)  # Reduced for quicker execution

    return data_points  # Return only the generated data points

# Example usage
if __name__ == "__main__":
    data_stream = generate_data_stream(num_samples=100, anomaly_probability=0.02, anomaly_scale=5)
    plt.plot(data_stream)
    plt.show()
    
    # Now pass this data_stream to your anomaly detector
    # Example: detector = EMAnomalyDetector(alpha=0.1, threshold_multiplier=3)
    # detected_anomalies = detect_anomalies(data_stream, detector)
