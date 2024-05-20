from bayeso_benchmarks import Levy
import design_bench
from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.continuous_dataset import ContinuousDataset
import numpy as np
import random


# define a custom dataset subclass of ContinuousDataset
class LevyDataset(ContinuousDataset):

    def __init__(self, **kwargs):
        # define a set of inputs and outputs of a quadratic function
        obj_fun = Levy(dim_problem=60)  # dimension of x
        bounds = obj_fun.get_bounds().astype(np.float32)

        x = obj_fun.sample_uniform(30000, seed=123).astype(np.float32)  # how many samples
        x = np.maximum(x, bounds[:, 0] + 1e-6)
        x = np.minimum(x, bounds[:, 1] - 1e-6)

        y = -obj_fun.output(x)

        # pass inputs and outputs to the base class
        super(LevyDataset, self).__init__(x, y, **kwargs)


class LevyOracle(ExactOracle):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    external_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which points to
        the mutable task dataset for a model-based optimization problem

    internal_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has frozen
        statistics and is used for training the oracle

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    internal_batch_size: int
        an integer representing the number of design values to process
        internally at the same time, if None defaults to the entire
        tensor given to the self.score method
    internal_measurements: int
        an integer representing the number of independent measurements of
        the prediction made by the oracle, which are subsequently
        averaged, and is useful when the oracle is stochastic

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    expect_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    """

    name = "toy_prediction"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {LevyDataset}

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
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """
        obj_fun = Levy(dim_problem=60)  # dimension of x
        bounds = obj_fun.get_bounds().astype(np.float32)
        x = np.maximum(x, bounds[:, 0] + 1e-6)
        x = np.minimum(x, bounds[:, 1] - 1e-6)
        y = -obj_fun.output(x).astype(np.float32).squeeze(-1)
        return y

    def __init__(self, dataset: ContinuousDataset, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        noise_std: float
            the standard deviation of gaussian noise added to the prediction
            values 'y' coming out of the ground truth score function f(x)
            in order to make the optimization problem difficult
        internal_measurements: int
            an integer representing the number of independent measurements of
            the prediction made by the oracle, which are subsequently
            averaged, and is useful when the oracle is stochastic

        """

        # initialize the oracle using the super class
        super(LevyOracle, self).__init__(
            dataset, internal_batch_size=1, is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=None, **kwargs)


if __name__ == '__main__':
    # register the new dataset with design_bench
    design_bench.register('Levy-Exact-v0', LevyDataset, LevyOracle,
                          # keyword arguments for building the dataset
                          dataset_kwargs=dict(max_samples=None, distribution=None, max_percentile=50, min_percentile=0),
                          # keyword arguments for building the exact oracle
                          oracle_kwargs=dict(noise_std=0.0)
                          )

    # build the new task (and train a model)
    task = design_bench.make("Levy-Exact-v0")


    def solve_optimization_problem(x0, y0):
        return x0  # solve a model-based optimization problem


    # evaluate the performance of the solution x_star
    x_star = solve_optimization_problem(task.x, task.y)
    y_star = task.predict(x_star)
    print(task.is_discrete)
    print(task.y.shape)
    print(y_star.shape)
    y_star = y_star.reshape(-1, 1)
    print(x_star[0], y_star[0])
    print(task.x[0], task.y[0])
    print(task.x.shape, task.y.shape)
    print(x_star.shape, y_star.shape)

    # dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    # dic2y["Levy-Exact-v0"] = (task.y.min(), task.y.max())
    # print((task.y.min(), task.y.max()))
    # np.save("npy/dic2y_new.npy", dic2y)

    dic2y = np.load("npy/dic2y_new.npy", allow_pickle=True).item()
    y_min, y_max = dic2y["Levy-Exact-v0"]
    print((task.y.max() - y_min) / (y_max - y_min))
