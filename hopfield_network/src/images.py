import numpy as np
import matplotlib.pyplot as plt

def generate_hopfield_training_data(num_patterns, pattern_size):
    """
    Generates a set of random bipolar patterns for training a Hopfield network.

    Args:
        num_patterns (int): The number of patterns to generate.
        pattern_size (int): The size (length) of each pattern vector.
                            Should be a perfect square for plotting.

    Returns:
        numpy.ndarray: A 2D array where each row is a training pattern.
    """
    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))

def generate_hopfield_testing_data(original_patterns, noise_level=0.2):
    """
    Generates a set of distorted patterns for testing a Hopfield network.

    Args:
        original_patterns (numpy.ndarray): The original, clean patterns.
        noise_level (float): The proportion of bits to flip in each pattern (0.0 to 1.0).

    Returns:
        numpy.ndarray: A 2D array of the same shape as original_patterns,
                       but with noise introduced.
    """
    testing_patterns = original_patterns.copy()
    num_patterns, pattern_size = original_patterns.shape
    num_bits_to_flip = int(pattern_size * noise_level)

    for i in range(num_patterns):
        flip_indices = np.random.choice(pattern_size, size=num_bits_to_flip, replace=False)
        testing_patterns[i, flip_indices] *= -1

    return testing_patterns

def plot_patterns(patterns, title):
    """
    Plots a set of patterns as images.

    Args:
        patterns (numpy.ndarray): A 2D array where each row is a pattern.
        title (str): The title for the entire plot.
    """
    num_patterns, pattern_size = patterns.shape
    image_dim = int(np.sqrt(pattern_size))

    # Check if the pattern size is a perfect square
    if image_dim * image_dim != pattern_size:
        print(f"Cannot plot patterns of size {pattern_size} as a square image.")
        return

    fig, axes = plt.subplots(1, num_patterns, figsize=(num_patterns * 2, 2.5))
    if num_patterns == 1:
        axes = [axes] # Make it iterable if there's only one pattern
        
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes):
        image = patterns[i].reshape((image_dim, image_dim))
        ax.imshow(image, cmap='binary')
        ax.set_title(f"Pattern {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_results(patterns, titles):
    """Plots a set of patterns as images."""
    num_patterns = len(patterns)
    image_dim = int(np.sqrt(patterns[0].size))
    
    fig, axes = plt.subplots(1, num_patterns, figsize=(num_patterns * 3, 4))
    fig.suptitle("Hopfield Network Recall", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = patterns[i].reshape((image_dim, image_dim))
        ax.imshow(image, cmap='binary')
        ax.set_title(titles[i])
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()