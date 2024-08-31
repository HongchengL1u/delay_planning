import os,sys
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import PMState
import Traj
import numpy as np
import matplotlib.pyplot as plt


import os,sys
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)

class dy_obstacle:
    def __init__(self,car=None):
        self.__id = None
        self.__raw = None # the dynamicObstacle in Commonroad
        self.__traj_xyt = None
        self.__shape:Rectangle = None
        self.__traj_sl = None
        self.__traj_t = None
        self.__traj_st = None
        if car is not None:
            self.init_by_Obstacle(car)

    def get_init_state(self):
        if(self.__raw == None):
            print('this object not be set')
            return None
        else:
            a = self.__raw.initial_state.acceleration
            v = self.__raw.initial_state.velocity
            return [v,a]
            

    def get_traj_t(self):
        return self.__traj_t
    def get_traj_xyt(self):
        return self.__traj_xyt

    def get_traj_sl(self):
        return self.__traj_sl

    def set_traj_sl(self,traj_sl):
        self.__traj_sl = traj_sl

    def init_by_Obstacle(self,car:DynamicObstacle):
        self.__shape = car.obstacle_shape

        # xyt
        # get init state
        ego_traj_res = [[car.initial_state.position[0],car.initial_state.position[1],car.initial_state.time_step]]
        _t = car.initial_state.time_step +1
        # the traj is from trajprediciton
        # so the initial state is not included in the traj from this way, so I do the first step
        traj:Trajectory = car.prediction.trajectory
        traj = traj.state_list
        for state in traj:
            # state is class PMState
            ego_traj_res.append([state.position[0],state.position[1],_t])
            _t+=1
        
        self.__raw  = car
        self.__traj_xyt = ego_traj_res
        self.__traj_xy = np.array(ego_traj_res)[:,0:2]
        self.__traj_t = np.array(ego_traj_res)[:,2]

    def car_forward_proj(self,ref_xyt):
        traj_xyt = self.__traj_xyt
        traj_xy = list(np.array(traj_xyt)[:,0:2])
        traj_t =  list(np.array(traj_xyt)[:,2])
        ref_t = list(np.array(ref_xyt)[:,2])
        ref_xy = list(np.array(ref_xyt)[:,0:2])

        if(traj_t != ref_t):
            print("traj time not align!")
            return None

        traj_sl = Traj.trajproj_xy2sl(ref_xy,traj_xy)
        self.__traj_sl = traj_sl

        l_offset = 2
        traj_st = Traj.sl_filter_st(l_offset,traj_sl,traj_t)
        self.__traj_st = traj_st
        return traj_st
    
class dy_obstacle2:
    def __init__(self,car=None):
        self.__id = None
        self.__raw = None # the dynamicObstacle in Commonroad
        self.__traj_xyt = None
        self.__shape:Rectangle = None
        self.__traj_sl = None
        self.__traj_t = None
        self.__traj_st = None
        self.__init_state = None
        self.__traj_v = None
        self.__traj_a = None
        
        if car is not None:
            self.init_by_Obstacle(car)

    @property
    def init_state(self):
        return self.__init_state
            
    @property
    def traj_t(self):
        return self.__traj_t
    
    @property
    def traj_xyt(self):
        return self.__traj_xyt

    @property
    def traj_sl(self):
        return self.__traj_sl

    @property
    def traj_v(self):
        return self.__traj_v
    
    @property
    def traj_a(self):
        return self.__traj_a
    
    @property
    def traj_x(self):
        return self.__traj_x
    
    @property
    def traj_y(self):
        return self.__traj_y
    
    @property
    def traj_heading(self):
        return self.__traj_heading

    @property
    def traj_xy(self):
        return self.__traj_xy

    def set_traj_sl(self,traj_sl):
        self.__traj_sl = traj_sl

    def init_by_Obstacle(self,car:DynamicObstacle):
        self.__shape = car.obstacle_shape

        # xyt
        # get init state
        ego_traj_res = [[car.initial_state.position[0],car.initial_state.position[1],car.initial_state.time_step,\
                         car.initial_state.velocity,car.initial_state.acceleration,car.initial_state.orientation]]
        _t = car.initial_state.time_step +1
        # the traj is from trajprediciton
        # so the initial state is not included in the traj from this way, so I do the first step
        traj:Trajectory = car.prediction.trajectory
        traj = traj.state_list
        for state in traj:
            # state is class PMState
            temp = [state.position[0],state.position[1],_t]
            if(hasattr(state,'velocity')):
                temp.append(state.velocity)
            else:
                temp.append(None)
            if(hasattr(state,'acceleration')):
                temp.append(state.acceleration)
            else:
                temp.append(None)

            if(hasattr(state,'orientation')):
                temp.append(state.orientation)
            else:
                temp.append(None) 
            ego_traj_res.append(temp)
            _t+=1
        
        self.__raw  = car
        self.__traj_xyt = ego_traj_res
        ego_traj_res = np.array(ego_traj_res)
        self.__traj_xy = list(ego_traj_res[:,0:2])
        self.__traj_x = list(ego_traj_res[:,0])
        self.__traj_y = list(ego_traj_res[:,1])
        self.__traj_t = list(ego_traj_res[:,2])
        self.__traj_v = list(ego_traj_res[:,3])
        self.__traj_a = list(ego_traj_res[:,4])
        self.__traj_heading = list(ego_traj_res[:,5])
        self.__init_state = [self.__raw.initial_state.velocity,self.__raw.initial_state.acceleration]
        
    def car_forward_proj(self,ref_xyt):
        traj_xyt = self.__traj_xyt
        traj_xy = list(np.array(traj_xyt)[:,0:2])
        traj_t =  list(np.array(traj_xyt)[:,2])
        ref_t = list(np.array(ref_xyt)[:,2])
        ref_xy = list(np.array(ref_xyt)[:,0:2])

        if(traj_t != ref_t):
            print("traj time not align!")
            return None

        traj_sl = Traj.trajproj_xy2sl(ref_xy,traj_xy)
        self.__traj_sl = traj_sl

        l_offset = 2
        traj_st = Traj.sl_filter_st(l_offset,traj_sl,traj_t)
        self.__traj_st = traj_st
        return traj_st
    
class dy_obstacle3:
    def __init__(self,car=None):
        self.__id = None
        self.__raw = None # the dynamicObstacle in Commonroad
        self.__traj_xytvah = None
        self.__traj_slt = None
        self.__shape:Rectangle = None
        self.__init_state = None

        if car is not None:
            self.init_by_Obstacle(car)

    @property
    def init_state(self):
        return self.__init_state
    
    @property
    def traj_xytvah(self):
        return self.__traj_xytvah
    
    @property
    def traj_slt(self):
        return self.__traj_slt
    

    def init_by_Obstacle(self,car:DynamicObstacle):
        self.__shape = car.obstacle_shape
        # xyt
        # get init state
        ego_traj_res = [[car.initial_state.position[0],car.initial_state.position[1],car.initial_state.time_step,\
                         car.initial_state.velocity,car.initial_state.acceleration,car.initial_state.orientation]]
        _t = car.initial_state.time_step +1
        # the traj is from trajprediciton
        # so the initial state is not included in the traj from this way, so I do the first step
        traj:Trajectory = car.prediction.trajectory
        traj = traj.state_list
        for state in traj:
            # state is class PMState
            temp = [state.position[0],state.position[1],_t]
            if(hasattr(state,'velocity')):
                temp.append(state.velocity)
            else:
                temp.append(None)
            if(hasattr(state,'acceleration')):
                temp.append(state.acceleration)
            else:
                temp.append(None)
            if(hasattr(state,'orientation')):
                temp.append(state.orientation)
            else:
                temp.append(None) 
            ego_traj_res.append(temp)
            _t+=1
        
        self.__raw  = car
        self.__traj_xytvah = ego_traj_res
        self.__init_state = [self.__raw.initial_state.velocity,self.__raw.initial_state.acceleration]
        
    def xyt2slt(self,ref_xy):
        traj_xytvah = self.__traj_xytvah
        traj_xytvah = np.array(traj_xytvah)
        traj_t =  traj_xytvah[:,2]
        traj_sl = self.xy2sl(ref_xy)
        traj_sl = np.array(traj_sl)
        traj_s = traj_sl[:,0]
        traj_l = traj_sl[:,1]
        traj_slt = list(zip(traj_s,traj_l,traj_t))
        self.__traj_slt = traj_slt
        return traj_slt

    def xy2sl(self,ref_xy):
        traj_xytvah = self.__traj_xytvah
        traj_xytvah = np.array(traj_xytvah)
        traj_xy = traj_xytvah[:,0:2]
        traj_sl = Traj.trajproj_xy2sl(ref_xy,traj_xy)
        return traj_sl
    
    def st_plot(self,ax):
        slt = self.__traj_slt
        slt = np.array(slt)
        s = slt[:,0]
        # l = slt[:,1]
        t = slt[:,2]
        f = lambda x: x/10
        t = list(map(f,t))
        ax.cla()
        ax.set_title("s-t graph")
        ax.plot(t,s,color='green')
        # ax.plot(t,l)
    
    def lt_plot(self,ax):
        slt = self.__traj_slt
        slt = np.array(slt)
        l = slt[:,1]
        t = slt[:,2]
        f = lambda x: x/10
        t = list(map(f,t))
        ax.cla()
        ax.set_title("l-t graph")
        # ax.plot(t,s)
        ax.plot(t,l,color='purple')

