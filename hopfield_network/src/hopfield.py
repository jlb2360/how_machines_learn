import numpy as np


class Hopfield():
    weights = None


    def store_images(self, images):
        """
        This assumes the images are already in vector form so a 2D matrix should be passed
        """

        self.num_neurons = len(images[0])
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        for image in images:
            self.weights += np.outer(image, image)

        np.fill_diagonal(self.weights, 0)

        self.weights /= len(images)

    def recall(self, image:np.ndarray, max_iter=20, asynchronous=True):
        """
        This assumes the image is already in vector form.
        """

        current = image.copy()

        for i in range(max_iter):

            last = current.copy()

            if asynchronous:
                neuron_indices = np.random.permutation(self.num_neurons)
                for neuron_idx in neuron_indices:
                    activation = np.dot(self.weights[neuron_idx, :], current)
                    current[neuron_idx] = 1 if activation > 0 else -1
            else:
                activations = np.dot(self.weights, current)

                current = np.sign(activations)
                current[current==0] = last[current==0]

            if np.array_equal(current, last):
                return current
            

        print("Network did not stabilize")
        return current