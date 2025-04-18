import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

LEN = 20000

class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="random"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from.
            class_choice: a string indicating how class will be selected for every
            sample.
                "random": class is chosen uniformly at random.
                "cycle": the sampler cycles through the classes sequentially.
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.map = []
        for class_ in range(self.labels.shape[1]):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.map.append(lst)

        print("counting-----")
        for i in range(len(self.map)):
            print(f"class {i} has {len(self.map[i])} samples:")

        # Only use classes that have samples
        self.valid_classes = [i for i, idxs in enumerate(self.map) if len(idxs) > 0]

        assert class_choice in ["random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        # Create a fresh generator each time.
        for _ in range(LEN):
            yield self.sample()

    # def __next__(self):
    #     print("count:", self.count)
    #     if self.count >= LEN:
    #         print("stop iteration")
    #         raise StopIteration
    #     self.count += 1
    #     return self.sample()

    def sample(self):
        if self.class_choice == "random":
            class_ = random.choice(self.valid_classes)
        elif self.class_choice == "cycle":
            class_ = self.valid_classes[self.current_class]
            self.current_class = (self.current_class + 1) % len(self.valid_classes)

        class_indices = self.map[class_]
        return np.random.choice(class_indices)

    def __len__(self):
        return LEN
