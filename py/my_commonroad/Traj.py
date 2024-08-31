import os,sys
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def trajxy_plot(self,show=False,ax=None,zorder=None):
    if ax is None:
        fig,ax = plt.subplots()
    else:
        show = False
    traj = np.array(self.__traj)
    if zorder is not None:
        ax.plot(traj[:,0],traj[:,1],color='purple',label='xyraw',zorder=zorder)
    else:
        ax.plot(traj[:,0],traj[:,1],color='purple',label='xyraw')

    if (show):
        # ax.set_aspect(1)
        plt.show()

def trajproj_xy2sl(traj_A,traj_B):
    '''
        B proj to A
        A is [[x0,y0],[x1,y1],...]
        B is [[x0,y0],[x1,y1],...]
    '''
    traj_sl = []
    
    for point_B in traj_B:
        traj_sl.append(pointproj_xy2sl(traj_A,point_B))
    return traj_sl

def pointproj_xy2sl(traj_A,point_B):
    '''
        point = [x,y]
        can use after init()
        point proj into traj_origin
    '''
    traj_origin = traj_A
    traj_origin_s = __calculate_cumulative_length(traj_A)
    
    # get the dis list between point and traj points
    dis_list = [np.linalg.norm(np.array(traj_point_origin)-np.array(point_B)) for traj_point_origin in traj_origin]
    # find the min dis
    min_index = dis_list.index(min(dis_list))
    n = None
    d = np.array(point_B)-np.array(traj_origin[min_index])
    if (min_index == len(traj_origin)-1):
        n = np.array(traj_origin[min_index])-np.array(traj_origin[min_index-1])
    elif (min_index == 0):
        n = np.array(traj_origin[1])-np.array(traj_origin[0])
    else:
        pointb = traj_origin[min_index-1]
        pointf = traj_origin[min_index+1]
        n = np.array(pointf)-np.array(pointb)


    # d is the vector from matched point to the point
    # n is the tangential vector at the matched point
    # l = .cross(n,d)
    # s = .dot(d,n)    
    l = (n[0]*d[1]-n[1]*d[0]) / np.sqrt(n[0]**2+n[1]**2)
    delta_s = (n[0]*d[0]+n[1]*d[1]) / np.sqrt(n[0]**2+n[1]**2)
    s = traj_origin_s[min_index] + delta_s

    # n = n/np.linalg.norm(n)

    return [s,l]

def __calculate_cumulative_length(points):
    
    # 初始化累计长度
    cumulative_length = 0
    cumulative_length_list = [0]
    # 计算相邻点之间的距离并累加
    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        distance = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        cumulative_length += distance
        cumulative_length_list.append(cumulative_length)

    return cumulative_length_list

def __xy_add_t (xy,t):
    x = list(np.array(xy)[:,0])
    y = list(np.array(xy)[:,1])
    return list(zip(x,y,t))

def sl_filter_st(l_offset,traj_sl,traj_t):
    assert(len(traj_sl) == len(traj_t))
    res = []
    for i in range(0,len(traj_sl)):
        point = traj_sl[i]
        t = traj_t[i]
        if(np.fabs(point[1])<l_offset):
            res.append([point[0],t])
        else:
            res.append([None,t])
    return res

def trajproj_st2xyt(traj_sl,traj_st,ref_xy):
    '''
        proj from st to xyt
        first: use ref_xy, to get x(s) y(s)
        then: use s(t) to get xyt
        last: return xyt
    '''

    # first
    origin_traj_sl = traj_sl
    planning_traj_st = traj_st
    if(origin_traj_sl is None):
        print("Please give the traj_sl info of this traj")
        return None
    origin_traj_sl = np.array(origin_traj_sl)
    interp_func = interp1d(origin_traj_sl[:,0], origin_traj_sl[:,1], kind='linear', fill_value='extrapolate')
    new_traj_s = np.array(planning_traj_st)[:,0]
    new_traj_l = interp_func(new_traj_s)   
    new_traj_t = np.array(planning_traj_st)[:,1]
    new_traj_slt = list(zip(new_traj_s,new_traj_l,new_traj_t))
    new_traj_slt = np.array(new_traj_slt)


    # second
    # 分离参考线的x和y坐标
    res_xy = trajproj_sl2xy(new_traj_slt[:,0:2],ref_xy)
    res_xy = np.array(res_xy)
    res_x = res_xy[:,0]
    res_y = res_xy[:,1]
    res_x = map(float,res_x)
    res_y = map(float,res_y)
    res_t = new_traj_slt[:,2]
    res_t = map(float,res_t)
    return list(zip(res_x,res_y,res_t))

def trajproj_sl2xy(traj_sl,ref_xy):
    ref_xy = np.array(ref_xy)
    x_ref = ref_xy[:, 0]
    y_ref = ref_xy[:, 1]

    # 计算沿参考线的弧长
    s_ref = np.zeros_like(x_ref)
    for i in range(1, len(s_ref)):
        s_ref[i] = s_ref[i - 1] + np.linalg.norm(ref_xy[i] - ref_xy[i - 1])

    # 插值函数，用于根据s_ref计算参考线的x和y
    interp_x = interp1d(s_ref, x_ref, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(s_ref, y_ref, kind='linear', fill_value='extrapolate')


    traj_sl = np.array(traj_sl)

    s_frenet = traj_sl[:,0]
    d_frenet = traj_sl[:,1]

    # 计算对应的参考线点
    x_frenet = interp_x(s_frenet)
    y_frenet = interp_y(s_frenet)

    # 计算法向量并归一化
    dx_ds = np.gradient(x_ref, s_ref)
    dy_ds = np.gradient(y_ref, s_ref)
    norm = np.sqrt(dx_ds**2 + dy_ds**2)
    nx = -dy_ds / norm
    ny = dx_ds / norm

    # 插值法向量
    interp_nx = interp1d(s_ref, nx, kind='linear', fill_value='extrapolate')
    interp_ny = interp1d(s_ref, ny, kind='linear', fill_value='extrapolate')

    # 计算笛卡尔坐标
    x_cartesian = x_frenet + d_frenet * interp_nx(s_frenet)
    y_cartesian = y_frenet + d_frenet * interp_ny(s_frenet)
    x_cartesian = map(float,x_cartesian)
    y_cartesian = map(float,y_cartesian)
    return list(zip(x_cartesian,y_cartesian))