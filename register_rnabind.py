import design_bench
from design_bench.oracles.sklearn import RandomForestOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import numpy as np


# define a custom dataset subclass of ContinuousDataset
class RNABindDataset(DiscreteDataset):

    def __init__(self, **kwargs):
        # define a set of inputs and outputs of a quadratic function
        x = np.load("data/RNA1_x.npy")
        y = np.load("data/RNA1_y.npy").reshape(-1,1)
        print(x.shape, y.shape)

        # pass inputs and outputs to the base class
        super(RNABindDataset, self).__init__(x, y, num_classes=4,  **kwargs)


if __name__ == '__main__':
    # register the new dataset with design_bench
    design_bench.register('RNABind-RandomForest-v0', RNABindDataset, RandomForestOracle,
                          # keyword arguments for building the dataset
                          dataset_kwargs=dict(
                              max_samples=None,
                              distribution=None,
                              max_percentile=80,
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


    def solve_optimization_problem(x0, y0):
        return x0  # solve a model-based optimization problem


    # evaluate the performance of the solution x_star
    x_star = solve_optimization_problem(task.x, task.y)
    y_star = task.predict(x_star)
    y_star = y_star.reshape(-1, 1)
    print(x_star[0], y_star[0])
    print(task.x[0], task.y[0])
    print(task.x.shape, task.y.shape)
    print(x_star.shape, y_star.shape)

    print(task.y.max(), task.y.min())