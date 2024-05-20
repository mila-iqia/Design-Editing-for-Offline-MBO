import design_bench
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import numpy as np


# define a custom dataset subclass of ContinuousDataset
class RNABindDataset(DiscreteDataset):

    def __init__(self, **kwargs):
        # define a set of inputs and outputs of a quadratic function
        x = np.load("data/RNA1_x.npy")
        y = np.load("data/RNA1_y.npy").reshape(-1, 1)
        print(x.shape, y.shape)

        # pass inputs and outputs to the base class
        super(RNABindDataset, self).__init__(x, y, num_classes=4, **kwargs)


if __name__ == '__main__':
    # register the new dataset with design_bench
    design_bench.register('RNABind-RandomForest-v0', RNABindDataset, RandomForestOracle,
                          # keyword arguments for building the dataset
                          dataset_kwargs=dict(
                              max_samples=None,
                              distribution=None,
                              max_percentile=100,
                              min_percentile=0,
                          ),

                          # keyword arguments for building RandomForest oracle
                          oracle_kwargs=dict(
                              noise_std=0.0,
                              max_samples=2000,
                              distribution=None,
                              max_percentile=100,
                              min_percentile=0,

                              # parameters used for building the model
                              model_kwargs=dict(n_estimators=100,
                                                max_depth=100,
                                                max_features="auto"),
                          ))

    # build the new task (and train a model)
    task = design_bench.make("RNABind-RandomForest-v0")

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
    # dic2y["RNABind-RandomForest-v0"] = (task.y.min(), task.y.max())
    # print((task.y.min(), task.y.max()))
    # np.save("npy/dic2y_new.npy", dic2y)

    dic2y = np.load("npy/dic2y_new.npy", allow_pickle=True).item()
    y_min, y_max = dic2y["RNABind-RandomForest-v0"]
    print((task.y.max() - y_min) / (y_max - y_min))
