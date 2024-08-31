import sys
sys.path.append("./")
import numpy as np
from my_commonroad.CRD_Scenario import scenario_pr
from my_commonroad.CRD_Obstacle import dy_obstacle3
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt



s = scenario_pr()
filename = "ZAM_Zip-1_68_T-1.xml"
s.init_from_xml("../scenario/",filename)
rm_obs_id = 3
for obs in s.scenario.dynamic_obstacles:
    if obs.obstacle_id == rm_obs_id:
        s.scenario.remove_obstacle(obs)
# s.plot()
        
# scenario and ref
rp = RoutePlanner(s.scenario,
                    list(s.problem_setting.planning_problem_dict.values())[0],
                    backend=RoutePlanner.Backend.NETWORKX,
                    reach_goal_state=False)
rp_result = candidate_holder = rp.plan_routes()
rp_result = rp_result.retrieve_first_route()
rnd = MPRenderer()
ref = np.array(rp_result.reference_path)
s.scenario.draw(rnd)
rnd.render()
rnd.ax.plot(ref[:,0],ref[:,1],zorder=3000,color='purple')
plt.savefig(f"../result/{filename}_ref.png",dpi=300)

target_id = 1
ego_id = 2
target_obs = None
ego_obs = None
for obs in s.scenario.dynamic_obstacles:
    if obs.obstacle_id == ego_id:
        ego_obs = dy_obstacle3(obs)
    if obs.obstacle_id == target_id:
        target_obs = dy_obstacle3(obs)
assert((target_obs is not None) and (ego_obs is not None))

ref_xy = ref[:,0:2].tolist()
target_obs.xyt2slt(ref_xy)
ego_obs.xyt2slt(ref_xy)

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(20,10))
# ego_obs slt plot
ego_obs.st_plot(ax1)
ego_obs.lt_plot(ax2)
plt.savefig("../result/ego_slt.png")

# target_obs slt plot
target_obs.st_plot(ax1)
target_obs.lt_plot(ax2)
plt.savefig("../result/target_obs_slt.png")



import pyceres

class delay_planning(pyceres.CostFunction):
    def __init__(self):
        pyceres.CostFunction.__init__(self)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0][0]
        residuals[0] = 10.0 - x
        
        # if jacobians is not None:
        #     jacobians[0][0] = -1.0
        return True

# J = Jo+Ja+Jb
# Jo = 0-tpin的Cost
# Ja = Jva + Jvv + Jvj + Jvcollision + Jra + Jrv + Jrj
# Jb = Jva + Jvv + Jvj + Jvcollision + Jra + Jrv + Jrj
# v = ((x[i]-x[i-1])/dt)  
# a = ((x[i]+x[i-2]-2*x[i-1])/dt**2) 
# j = ((x[i]-3*x[i-1]+3*x[i-2]+1)/dt**3)


# target_obs_st = 
# 需要目标车辆CUT_IN的概率
# 忽略或者考虑
# 忽略的情况下
# 不忽略的情况下，需要目标车辆的ST
# 

