import numpy as np
import random
import time
import matplotlib.pyplot as plt

def generate_data_stream(num_samples=200, mean=0, std_deviation=1, noise_level=0.5,
                         trend_factor=0.5, seasonality_amplitude=10, seasonality_period=30,
                         anomaly_probability=0.05, anomaly_scale=1.2):
    data_points = []

    for t in range(num_samples):
        # Regular pattern (linear trend)
        trend = trend_factor * t
        
        # Seasonal pattern (sinusoidal)
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        
        # Generating base data point with randomness
        base_data = np.random.normal(mean, std_deviation) + trend + seasonality
        
        # creating random noise
        noise = random.uniform(-noise_level, noise_level)
        
        data_point = base_data + noise

        #Randomly selecting data points and making it anomaly
        if random.random() < anomaly_probability:
            data_point *= anomaly_scale  

        data_points.append(data_point)
        
        # Simulating real-time data streaming
        time.sleep(0.01)  

    return data_points  


if __name__ == "__main__":
    data_stream = generate_data_stream(num_samples=100, anomaly_probability=0.02, anomaly_scale=5)
    plt.plot(data_stream)
    plt.show()
    
    