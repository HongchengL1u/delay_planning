import numpy as np
import pyceres

# ref: examples/helloworld_analytic_diff.cc
class linefitCostFunction(pyceres.CostFunction):
    def __init__(self,point):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2]) 
        self.x_ = point[0]
        self.y_ = point[1]

    def Evaluate(self, parameters, residuals, jacobians):
        # fit y = ax+b
        a = parameters[0][0]
        b = parameters[0][1]
        residuals[0] = self.y_ - a*self.x_ -b
        if(jacobians is not None):
            jacobians[0][0] = -self.x_
            jacobians[0][1] = -1
        return True

class curvefitCostFunction(pyceres.CostFunction):
    def __init__(self,point):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1,1]) 
        self.x_ = point[0]
        self.y_ = point[1]

    def Evaluate(self, parameters, residuals, jacobians):
        # fit y = ax+b
        a = parameters[0][0]
        b = parameters[1][0]
        residuals[0] = self.y_ - a*self.x_ -b
        if(jacobians is not None):
            jacobians[0][0] = -self.x_
            jacobians[1][0] = -1
        return True

# class Ja_trajCostFunction(pyceres.CostFunction):
#     def __init__(self,Wa,jref,aref,vref,vu,vl,au,al,ju,jl,N):
#         pyceres.CostFunction.__init__(self)
#         self.set_num_residuals(1)
#         self.set_parameter_block_sizes([N]) 

#     # TODO auto diff
#     def Evaluate(self, parameters, residuals, jacobians):
#         # fit y = ax+b
#         a = parameters[0][0]
#         b = parameters[1][0]
#         residuals[0] = self.y_ - a*self.x_ -b
#         if(jacobians is not None):
#             jacobians[0][0] = -self.x_
#             jacobians[1][0] = -1
#         return True

def test_python_cost():
    x1 = np.array(3.0)
    x1_ori = x1.copy()
    x2 = np.array(1.0)
    x2_ori = x2.copy()
    prob = pyceres.Problem()
    
    for point in [[0,3],[1,4],[2,5]]:
        cost = curvefitCostFunction(point)
        prob.add_residual_block(cost, None, [x1,x2])
    options = pyceres.SolverOptions()
    options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())
    print(f"{x1_ori} -> {x1}")
    print(f"{x2_ori} -> {x2}")

if __name__ == "__main__":
    test_python_cost()
