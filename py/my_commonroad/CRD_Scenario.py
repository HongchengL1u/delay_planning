import os,sys
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad.scenario.state import ExtendedPMState,InitialState,CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle,ObstacleType


import matplotlib.pyplot as plt
import numpy as np
import imageio
import Traj 

from CRD_Obstacle import dy_obstacle

def one_read(xml_dir,filename):
    filepath = os.path.join(xml_dir,filename)
    s,p = CommonRoadFileReader(filepath).open()
    return s,p

def batch_read(xml_dir):
    scenario = []
    ps = []
    file_list = os.listdir(xml_dir)
    for file_name in file_list:
        filepath = os.path.join(xml_dir,file_name)
        scenario_,ps_ = CommonRoadFileReader(filepath).open()
        scenario.append(scenario_)
        ps.append(ps_)

    return scenario,ps,file_list

def plot(scenario:Scenario,filename,gif=True,filepath=None):
    frames = []
    fig,ax = plt.subplots(dpi=300)
    
    t_min = [];
    t_max = [];
    
    dynamic_obstacles_list = scenario.dynamic_obstacles
    for do in dynamic_obstacles_list:
        t_min.append(do.initial_state.time_step)
        # t_min.append(do.prediction.initial_time_step) 
        t_max.append(do.prediction.final_time_step)
    t_min = min(t_min)
    t_max = max(t_max)
    for i in range(t_min,t_max+1):
        rnd = MPRenderer(ax=ax)
        rnd.draw_params.time_begin = i
        rnd.draw_params.dynamic_obstacle.show_label = True
        scenario.lanelet_network.draw(rnd)
        dl = scenario.dynamic_obstacles
        for v in dl:
            if hasattr(v.state_at_time(i),'position'):
                v.draw(rnd)
            else:
                continue
        rnd.render()

        if(gif):
            frame = fig
            frame.canvas.draw()  # 渲染图形
            image = np.frombuffer(frame.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(frame.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
        else:
            plt.pause(0.05)
            plt.show(block=False)
    if(gif and filepath is None):
        imageio.mimsave(f'../result/{filename}_Scenario.gif', frames, fps=10)
    elif (gif and filepath is not None):
        imageio.mimsave(os.path.join(filepath+f'{filename}_Scenario.gif'), frames, fps=10)
    else:
        plt.show()

class scenario_pr:
    def __init__(self):
        self.__scenario = None
        self.__problem_setting = None
        self.__dt = None
        self.__filename = None
        
    @property
    def scenario(self):
        return self.__scenario

    
    @property
    def problem_setting(self):
        return self.__problem_setting

    def get_dt(self):
        return self.__dt
    
    def get_car_from_id(self,id):
        scenario = self.__scenario
        dls = scenario.dynamic_obstacles
        for dl in dls:
            if dl.obstacle_id == id:
                return dl
        return None   
    def init_from_Scenario(self,s:Scenario,filename):
        if(self.__scenario is not None):
            print("Error: This obj has initiated!")
            return None
        else:
            self.__scenario = s
            self.__filename = filename
            self.__dt = s.dt
    
    def init_from_xml(self,xml_dir,filename=None):
        if(self.__scenario is not None):
            print("Error: This obj has initiated!")
            return None
        else:
            s,p = one_read(xml_dir,filename)
            self.__scenario = s
            self.__problem_setting = p
            self.__filename = filename
            self.__dt = s.dt

    def t_range(self):
        '''
            get the t_range of this scenario 
            probably not the same with the ego traj t_range
        '''
        scenario = self.__scenario
        t_min = []
        t_max = []
        
        dynamic_obstacles_list = scenario.dynamic_obstacles
        for do in dynamic_obstacles_list:
            t_min.append(do.initial_state.time_step)
            # t_min.append(do.prediction.initial_time_step) 
            t_max.append(do.prediction.final_time_step)
        return min(t_min),max(t_max)
    
    def plot(self,gif=True):
        tmin,tmax = self.t_range()
        print(f"time: {[tmin,tmax]}")
        frames = []
        scenario = self.__scenario
        fig,ax = plt.subplots(dpi=300)
        for i in range(tmin,tmax+1):
            rnd = MPRenderer(ax=ax)
            rnd.draw_params.time_begin = i
            rnd.draw_params.dynamic_obstacle.show_label = True
            scenario.lanelet_network.draw(rnd)
            dl = scenario.dynamic_obstacles
            for v in dl:
                if hasattr(v.state_at_time(i),'position'):
                    v.draw(rnd)
                else:
                    continue
            rnd.render()
            if(gif):
                frame = fig
                frame.canvas.draw()  # 渲染图形
                image = np.frombuffer(frame.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(frame.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
            else:
                plt.pause(0.05)
                plt.show(block=False)
        if(gif):
            imageio.mimsave(f'../result/{self.__filename}_Scenario.gif', frames, fps=10)
        else:
            plt.show()

    def replace_dynamic_ob (self,traj_xyt,ego_id):
        scenario = self.__scenario
        traj = list(np.array(traj_xyt)[:,0:2])
        t_range = self.t_range()

        ego_ob = None
        dy_obs = scenario.dynamic_obstacles;
        for dy_ob in dy_obs:
            if dy_ob.obstacle_id == ego_id:
                ego_ob = dy_ob
                break
        if(ego_ob is None):
            print("Not find replaced dynamic obstacle")
            return None
        
        init_state = InitialState(position=traj[0],orientation=np.arctan2(traj[1][1]-traj[0][1],traj[1][0]-traj[0][0]),time_step=int(t_range[0]))

        # convert planned trajectory to use point-mass states to include lateral velocity
        # orientation = (
        #     planning_problem.initial_state.orientation
        # )  # assume the ego vehicle keeps its orientation from the initial state
        state_list = [
            CustomState(
                time_step=t,
                position=np.array(traj[i]),
                orientation= np.arctan2(traj[i][1]-traj[i-1][1],traj[i][0]-traj[i-1][0])
            )
            for i,t in zip(range(1,len(traj)),range(int(t_range[0]+1),int(t_range[1]+1))) 
        ]
        ego_vehicle_trajectory = Trajectory(initial_time_step=int(t_range[0]+1), state_list=state_list)

        # create the prediction using the planned trajectory and the shape of the ego vehicle
        # vehicle3 = parameters_vehicle3.parameters_vehicle3()
        ego_vehicle_shape = Rectangle(length=ego_ob.obstacle_shape.length , width=ego_ob.obstacle_shape.width)
        ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)

        # the ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_type = ObstacleType.CAR
        ego_vehicle = DynamicObstacle(
            obstacle_id=ego_id,
            obstacle_type=ego_vehicle_type,
            obstacle_shape=ego_vehicle_shape,
            initial_state=init_state,
            prediction=ego_vehicle_prediction,
        )
        scenario.remove_obstacle(ego_ob)
        scenario.add_objects([ego_vehicle])
        self.__scenario = scenario
        return ego_vehicle
    
    @staticmethod
    def one_read(xml_dir,filename=None):
        if(filename is None):
            filepath = os.path.join(xml_dir,filename)
        else:
            filepath = xml_dir
        s,p = CommonRoadFileReader(filepath).open()
        return s,p


class LC_scenario_pr(scenario_pr):
    '''
        Lane_Change_Scenario
        现在所处理的场景为一个左转的场景
        右转可能需要把Boundary改变一下
    '''
    def __init__(self):
        # super(LC_scenario_pr,self).__init__()
        scenario_pr.__init__(self)
        self.__map = dict()
        self.__ego_id = None
        self.__LC_F_id = None
        self.__LC_B_id = None
        self.__CIPV_id = None
        self.__ego = None
        self.__LC_F_car = None
        self.__LC_B_car = None
        self.__CIPV_car = None

        self.__ref = None
    
    def set_all_from_id(self,ego_id,LCF_id,LCB_id,CIPV_id):
        self.set_ego_from_id(ego_id)
        self.set_CIPV_from_id(CIPV_id)
        self.set_LCB_from_id(LCB_id)
        self.set_LCF_from_id(LCF_id)

    def set_ego_from_id(self,id):
        ego = self.get_car_from_id(id)
        if (ego is not None):
            self.__ego_id = id
            self.__ego = dy_obstacle(ego)
            self.__map[id] = self.__ego
        else:
            print(f"ego_id_invalid:{id}")
    
    def set_LCF_from_id(self,id):
        ego = self.get_car_from_id(id)
        if (ego is not None):
            self.__LC_F_id = id
            self.__LC_F_car = dy_obstacle(ego)
            self.__map[id] = self.__LC_F_car
        else:
            print(f"LCF_id_invalid:{id}")
            
    def set_LCB_from_id(self,id):
        ego = self.get_car_from_id(id)
        if (ego is not None):
            self.__LC_B_id = id
            self.__LC_B_car = dy_obstacle(ego)
            self.__map[id] = self.__LC_B_car
        else:
            print(f"LCB_id_invalid:{id}")
    
    def set_CIPV_from_id(self,id):
        ego = self.get_car_from_id(id)
        if (ego is not None):
            self.__CIPV_id = id
            self.__CIPV_car = dy_obstacle(ego)
            self.__map[id] = self.__CIPV_car
        else:
            print(f"CIPV_id_invalid:{id}")

    def test(self):

        if(self.__ego is not None):
            ref_xyt = self.__ego.get_traj_xyt()
            ego_st = self.__ego.car_forward_proj(ref_xyt)
            ref_sl = self.__ego.get_traj_sl()
        else:
            print("Error! ref is None")
            return None

        if(self.__CIPV_car is not None):
            CIPV_st = self.__CIPV_car.car_forward_proj(ref_xyt)

        else:
            print("CIPV is None")
            CIPV_st = None
        
        if(self.__LC_B_car is not None):
            LCB_st = self.__LC_B_car.car_forward_proj(ref_xyt)
        else:
            print("LCB is None")
            LCB_st = None
        
        if(self.__LC_F_car is not None):
            LCF_st = self.__LC_F_car.car_forward_proj(ref_xyt)
        else:
            print("LCF is None")
            LCF_st = None

        l_boundary = []
        u_boundary = []
        for i in range(0,len(ref_xyt)):
            if LCB_st is not None:
                LCB_s = LCB_st[i][0]
            else:
                LCB_s = None
            if LCF_st is not None:
                LCF_s = LCF_st[i][0]
            else:
                LCF_s = None
            if CIPV_st is not None:
                CIPV_s = CIPV_st[i][0]
            else:
                CIPV_s = None

            if LCB_s is not None:
                l = LCB_s+3
            else:
                l = -1
            if LCF_s is not None and CIPV_s is not None:
                u = min(CIPV_s-5,LCF_s-3)
            elif LCF_s is not None:
                u = LCF_s-3
            elif CIPV_s is not None:
                u = CIPV_s -5
            else:
                u = 200
            l_boundary.append(l)
            u_boundary.append(u)
        return l_boundary,u_boundary

    def res_reporj(self,res_st):
        res_sl = self.__ego.get_traj_sl()
        ref_xyt = self.__ego.get_traj_xyt()
        ref_xy = list(np.array(ref_xyt)[:,0:2])
        res_xyt = Traj.trajproj_st2xyt(res_sl,res_st,ref_xy)
        return res_xyt


    def get_ego_init(self):
        if(self.__ego == None):
            print("ego not be set")
            return None
        else:
            return self.__ego.get_init_state()
        

    







