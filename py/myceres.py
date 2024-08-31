import numpy as np
import matplotlib.pyplot as plt
import pyceres as ceres

# N = 80
# t_step = 1
# t0 = 40
# s0 = 400
# v0 = 10
# target_obj = [[s0+v0*(t-t0),t] for t in range(t0,N+1)]
# target_obj = np.array(target_obj)
# target_obj_t = target_obj[:,1]
# target_obj_s = target_obj[:,0]
# plt.plot(target_obj_t,target_obj_s)


# init_s = 0
# init_v = 15
# init_a = 0

import numpy as np

import pyceres


# ref: examples/helloworld_analytic_diff.cc
class HelloworldCostFunction(pyceres.CostFunction):
    def __init__(self):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0][0]
        residuals[0] = 10.0 - x
        if jacobians is not None:
            jacobians[0][0] = -1.0
        return True


def test_python_cost():
    x = np.array(5.0)
    x_ori = x.copy()
    prob = pyceres.Problem()
    cost = HelloworldCostFunction()
    prob.add_residual_block(cost, None, [x])
    options = pyceres.SolverOptions()
    options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())
    print(f"{x_ori} -> {x}")


if __name__ == "__main__":
    test_python_cost()


