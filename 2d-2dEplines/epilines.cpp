#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "epipolarlinesdraw.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;



const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ../testfilesdataset/20.png | image 00.}"
    "{@input1 | ../testfilesdataset/21.png | image 01.}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv ,keys);
    Mat src0 = imread(parser.get<String>("@input0"));
    Mat src1 = imread(parser.get<String>("@input1"));


    if (src0.empty() || src1.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "<Input image>" <<endl;
        return -1;
    }
    
    Mat grey0, grey1,descriptors0, descriptors1;
    cvtColor(src0,grey0, COLOR_BGR2GRAY);
    cvtColor(src1,grey1, COLOR_BGR2GRAY);

    //requires intrinsic parameters of camera, use desktop logitech for now
    FileStorage fs;
    if( !fs.open("/home/joseph/Desktop/out_camera_data_logitech.xml", FileStorage::READ))
    {
        cout<<"fail to read camera parameters"<<endl;
        return -1;
    }
    //camera matrix and dist coeffs
    Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"]>> cameraMatrix;
    fs["distortion_coefficients"]>>distCoeffs;

    std::vector<Point2f> corners0, corners1;
    Size patternSize(9,6);
    bool found0 = findChessboardCorners(grey0,patternSize,corners0);    //find pixel of corners
    bool found1 = findChessboardCorners(grey1,patternSize,corners1);

    Mat essentialMask,E,F,R,R1,t,ans;
    //essential matrix and fundamental matrix, fundamental seems to be wrong    
    //E = cv::findEssentialMat(corners0,corners1,cameraMatrix,RANSAC,0.999,1.0, essentialMask);
    F = cv::findFundamentalMat(corners0,corners1,FM_RANSAC,1.,0.99);
    cout << "F :" << F <<endl;
    Mat grey0_copy_pose, grey1_copy_pose;
    cvtColor(grey0, grey0_copy_pose, COLOR_GRAY2BGR);
    cvtColor(grey1, grey1_copy_pose, COLOR_GRAY2BGR);

   

    //drawEpipolarLines<float,float>("showpic", F, grey0_copy_pose, grey1_copy_pose, corners0,corners1);
    //cout << "findEssentialMat:" << E << endl;

    
    std::vector<cv::Vec3f> epilines0,epilines1;
 
    computeCorrespondEpilines(corners0,1,F,epilines0);
    computeCorrespondEpilines(corners1,2,F,epilines1);
    cout <<"epilines1: " << epilines1[0][0] <<endl;
    cout <<"epilines1: " << epilines1[0][1] <<endl;
    cout <<"epilines1: " << epilines1[0][2] <<endl;
    RNG rng(0);

    for(size_t i=0; i <corners0.size();i++)
    {
        Point point0(0, -epilines1[i][2]/epilines1[i][1]);
        Point point1(grey1_copy_pose.cols,-(epilines1[i][2]+epilines1[i][0]*grey1_copy_pose.cols)/epilines1[i][1]);
        //cout <<" Point1 " << point1 <<endl;
        Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255),CV_8UC3); //random color
        line(grey0_copy_pose, point0,point1,color,2);

        Point point2(0, -epilines0[i][2]/epilines0[i][1]);
        Point point3(grey0_copy_pose.cols,-(epilines0[i][2]+epilines0[i][0]*grey0_copy_pose.cols)/epilines0[i][1]);
        //cout <<" Point1 " << point1 <<endl;
        line(grey1_copy_pose, point2,point3,color,2);
        
    }
    
    imshow("epilines for 0",grey0_copy_pose);
    imwrite("epilinesFor0.png",grey0_copy_pose);
    imshow("epilines for 1",grey1_copy_pose);
    imwrite("epilinesFor1.png",grey1_copy_pose);
    waitKey();
    
    return 0;
}

#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif  


