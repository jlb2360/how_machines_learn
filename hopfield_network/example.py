import numpy as np

from src.images import generate_hopfield_testing_data, generate_hopfield_training_data, plot_patterns, plot_results

from src.hopfield import Hopfield


if __name__ == "__main__":
    # Define parameters
    NUM_PATTERNS = 4
    PATTERN_SIZE = 100 # 10x10 image
    NOISE_LEVEL = 0.15 # 15% noise

    training_patterns = generate_hopfield_training_data(NUM_PATTERNS, PATTERN_SIZE)

    testing_patterns = generate_hopfield_testing_data(training_patterns, noise_level=NOISE_LEVEL)

    plot_patterns(training_patterns, "Original Training Patterns")

    plot_patterns(testing_patterns, f"Testing Patterns ({int(NOISE_LEVEL*100)}% Noise)")

    hop_net = Hopfield()

    hop_net.store_images(training_patterns)

    rand_idx = np.random.randint(0, NUM_PATTERNS)
    corrupted_image = testing_patterns[rand_idx]
    true_image = training_patterns[rand_idx]

    recalled_image = hop_net.recall(corrupted_image, asynchronous=True)

    plot_results(
        patterns=[true_image, corrupted_image, recalled_image],
        titles=["Original", "Corrupted", "Recalled"]
    )



