from bayeso_benchmarks import Rosenbrock

obj_fun = Rosenbrock(dim_problem=60)  # dimension of x
bounds = obj_fun.get_bounds()

X = obj_fun.sample_uniform(5)  # how many samples

Y = obj_fun.output(X)
print(X)
print(Y)