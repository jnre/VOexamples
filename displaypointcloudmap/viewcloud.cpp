#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/registration/icp.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <math.h>
#define PI           3.14159265358979323846 
int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud3 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud4 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud5 (new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("t1.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("t2.pcd", *cloud2) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("t3.pcd", *cloud3) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("t4.pcd", *cloud4) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }

    /*
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd01.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd02.pcd", *cloud2) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }

    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd03.pcd", *cloud3) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd04.pcd", *cloud4) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd05.pcd", *cloud5) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    */
    std::cout << "Loaded "
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;
    //for ply
    /*
    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("test_pcd01.ply", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("test_pcd02.ply", *cloud2) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    
    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("test_pcd03.ply", *cloud3) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("test_pcd04.ply", *cloud4) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }
    
    if (pcl::io::loadPLYFile<pcl::PointXYZ> ("test_pcd05.ply", *cloud5) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file  \n");
        return (-1);
    }    
    */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud2, 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color3(cloud3, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color4(cloud4, 255, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color5(cloud5, 0, 0, 0);

    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->addPointCloud<pcl::PointXYZ> (cloud2, single_color2, "sample cloud2");
    //viewer->addPointCloud<pcl::PointXYZ> (cloud3, single_color3, "sample cloud3");
    //viewer->addPointCloud<pcl::PointXYZ> (cloud4, single_color4, "sample cloud4");
    //viewer->addPointCloud<pcl::PointXYZ> (cloud5, single_color5, "sample cloud5");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud3");
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud4");
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud5");
    //green,blue,red,purple,black
    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud2);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout<<"hasconverged:" << icp.hasConverged() << "score: " << icp.getFitnessScore() << std::endl;
    std::cout<< icp.getFinalTransformation() << std::endl;


    Eigen::Matrix4f qq;
    Eigen::Affine3f t;
    qq << cos(PI), -sin(PI), 0, 0,
            sin(PI),cos(PI),0, 0,
            0,0,1 ,0,
            0,0,0,1;
    t = qq;
    
    viewer->addCoordinateSystem(300.0,"reference",0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

  return (0);
}
