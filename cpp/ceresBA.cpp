#include <ceres/ceres.h>
#include <iostream>
#include "BALProblem.h"
#include <string>
#include <eigen3/Eigen/Dense>
using namespace std;

template<typename T>
inline void AngleVectorRotate(const T camera[3],const T p[3],T result[3])
{
    Eigen::Matrix<T,3,1> Mcamera(camera);
    T theta2 = Mcamera.dot(Mcamera);
    
    if(theta2 > T(std::numeric_limits<double>::epsilon()))
    {
        //avoid divide 0
        
        Eigen::Matrix<T,3,1> Mp(p);
        Eigen::Matrix<T,3,1> Mresult(result);
        T theta = sqrt(theta2);
        Eigen::Matrix<T,3,1> Mn = Mcamera/theta;
        Mresult = cos(theta)*Mp+sin(theta)*(Mn.cross(Mp))+(T(1.0)-cos(theta))*Mn*Mn.transpose()*Mp;
        result[0]=Mresult(0);
        result[1]=Mresult(1);
        result[2]=Mresult(2);
    }
        
    else 
    {
        //theta near 0
        //R = I + theta * hat(n)
        result[0]=p[0];
        result[1]=p[1];
        result[2]=p[2];
    }
    return ;
    
}

struct CostFunctor
{
    CostFunctor(double obs_x,double obs_y): _obs_x(obs_x),_obs_y(obs_y) {}
    template<typename T>
    bool operator()(const T* const camera,const T* const points, T* residual) const
    {
        // Pc = R*Pw+t
        T p[3];
        AngleVectorRotate(camera,points,p); 
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // pc normalized
        T x = -p[0]/p[2]; //BALproblem so '-'
        T y = -p[1]/p[2];
        T rr = x*x+y*y;

        // p = f*distortion*pc 
        T k1 = camera[7];
        T k2 = camera[8];
        T distortion = T(1.0)+ k1*rr + k2*rr*rr;
        T f = camera[6];
        T predict_x = x * f * distortion;
        T predict_y = y * f * distortion;

        // residual
        residual[0] = T(_obs_x) - predict_x;
        residual[1] = T(_obs_y) - predict_y;
        return true;

    }
    private:
        double _obs_x,_obs_y;
};



int main()
{
    ceres::Problem problem;
    string data_filepath = "../data/problem-16-22106-pre.txt";
    BALProblem balproblem(data_filepath);
    balproblem.WriteToPLYFile("initial.ply");
    cout << endl;

    //
    double* cameras = balproblem.mutable_cameras();
    double* points = balproblem.mutable_points();
    const double* observations = balproblem.observations();

    const int point_block_size = balproblem.point_block_size();
    const int camera_block_size = balproblem.camera_block_size();
    for(int i = 0; i < balproblem.num_observations(); ++i)
    {
        problem.AddResidualBlock(
        // 2 residual dimension 
        // 9 optimization1 
        // 3 optimization2
        new ceres::AutoDiffCostFunction<CostFunctor, 2, 9, 3>(new CostFunctor(observations[2*i],observations[2*i+1])),
        nullptr,
        cameras + camera_block_size * balproblem.camera_index()[i],
        points + point_block_size * balproblem.point_index()[i]); 
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR; //need to know further
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    balproblem.WriteToPLYFile("result.ply");
    return 0;
    
    
}