import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_with_confidence_interval(x_lists, x_axis=None, confidence=0.95):
    """Plot mean with confidence interval"""
    # Convert to numpy array (n_lists, n_points)
    data = np.array(x_lists)
    
    # Calculate statistics
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)  # sample std
    n = len(x_lists)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * std / np.sqrt(n)
    
    # Create x-axis if not provided
    if x_axis is None:
        x_axis = np.arange(len(mean))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, mean, 'b-', linewidth=2, label='Mean')
    plt.fill_between(x_axis, mean - margin_error, mean + margin_error, 
                     alpha=0.3, color='blue', label=f'{confidence*100}% CI')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()

def plot_with_std_deviation(x_lists, x_axis=None, num_std=1):
    """Plot mean with standard deviation bands"""
    # Convert to numpy array
    data = np.array(x_lists)
    
    # Calculate statistics
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Create x-axis if not provided
    if x_axis is None:
        x_axis = np.arange(len(mean))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, mean, 'r-', linewidth=2, label='Mean')
    plt.fill_between(x_axis, mean - num_std*std, mean + num_std*std, 
                     alpha=0.3, color='red', label=f'±{num_std}σ')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()

# Example usage:
x1 = [1.0, 2.1, 3.2, 4.0, 5.1]
x2 = [1.1, 2.0, 3.1, 4.2, 5.0]
x3 = [0.9, 2.2, 3.0, 3.9, 5.2]
plot_with_confidence_interval([x1, x2, x3])
plot_with_std_deviation([x1, x2, x3])
