import design_bench
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import numpy as np
import flexs


def int_to_rna(integer_sequence):
    # Create a list of nucleotides
    nucleotides = ['U', 'G', 'C', 'A']

    # Convert each integer sequence in the list to a DNA sequence
    sequence = [''.join([nucleotides[i] for i in integer_sequence])]

    return sequence
# define a custom dataset subclass of ContinuousDataset
class RNABindDataset(DiscreteDataset):

    def __init__(self, **kwargs):
        # define a set of inputs and outputs of a quadratic function
        x = np.load("data/RNA1_x.npy")
        y = np.load("data/RNA1_y.npy").reshape(-1, 1)
        print(x.shape, y.shape)

        # pass inputs and outputs to the base class
        super(RNABindDataset, self).__init__(x, y, num_classes=4, **kwargs)


class RNABindOracle(ExactOracle):

    name = "toy_prediction"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {RNABindDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return False

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return True

    def protected_predict(self, x):
        x1 = int_to_rna(x.tolist())
        s = np.array(self.landscape.get_fitness(x1)).reshape(-1)
        return np.float32(s)

    def __init__(self, dataset: ContinuousDataset, **kwargs):
        problem = flexs.landscapes.rna.registry()['L14_RNA1']
        self.landscape = flexs.landscapes.RNABinding(**problem['params'])
        # initialize the oracle using the super class
        super(RNABindOracle, self).__init__(
            dataset, internal_batch_size=1, is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=None, **kwargs)


if __name__ == '__main__':
    # register the new dataset with design_bench
    design_bench.register('RNABind-Exact-v0', RNABindDataset, RNABindOracle,
                          # keyword arguments for building the dataset
                          dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                          # keyword arguments for building the exact oracle
                          oracle_kwargs=dict(noise_std=0.0)
                          )

    # build the new task (and train a model)
    task = design_bench.make("RNABind-Exact-v0")

    # evaluate the performance of the solution x_star
    print(task.x.shape, task.y.shape)
    print(task.predict(task.x).shape)
    print(task.y.min(), task.y.max())

    # task.map_to_logits()
    # task.map_normalize_x()
    # task.map_normalize_y()
    # print(task.x.shape, task.y.shape)
    # print(task.predict(task.x).shape)
    # print(task.y.min(), task.y.max())

    # dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    # dic2y["RNABind-Exact-v0"] = (task.y.min(), task.y.max())
    # print((task.y.min(), task.y.max()))
    # np.save("npy/dic2y_new.npy", dic2y)

    dic2y = np.load("npy/dic2y_new.npy", allow_pickle=True).item()
    y_min, y_max = dic2y["RNABind-Exact-v0"]
    print((task.y.max() - y_min) / (y_max - y_min))
