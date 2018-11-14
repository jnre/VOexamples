#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ../testfilesdataset/10.png | image 00.}"
    "{@input1 | ../testfilesdataset/11.png | image 01.}"
    "{@input2 | ../testfilesdataset/12.png | image 02.}"
    "{@input3 | ../testfilesdataset/13.png | image 03.}"
    "{@input4 | ../testfilesdataset/14.png | image 04.}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv ,keys);
    Mat src0 = imread(parser.get<String>("@input0"));
    Mat src1 = imread(parser.get<String>("@input1"));
    Mat src2 = imread(parser.get<String>("@input2"));
    Mat src3 = imread(parser.get<String>("@input3"));
    Mat src4 = imread(parser.get<String>("@input4"));

    if (src0.empty() || src1.empty() || src2.empty() || src3.empty() || src4.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "<Input image>" <<endl;
        return -1;
    }
    
    Mat grey0, grey1, grey2, grey3, grey4, descriptors0, descriptors1;
    cvtColor(src0,grey0, COLOR_BGR2GRAY);
    cvtColor(src1,grey1, COLOR_BGR2GRAY);
    cvtColor(src2,grey2, COLOR_BGR2GRAY);
    cvtColor(src3,grey3, COLOR_BGR2GRAY);
    cvtColor(src4,grey4, COLOR_BGR2GRAY);
    
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
    E = cv::findEssentialMat(corners0,corners1,cameraMatrix,RANSAC,0.999,1.0, essentialMask);
    F = cv::findFundamentalMat(corners0,corners1,FM_RANSAC,1.,0.99);
    cout << "F :" << F <<endl;
    cout << "findEssentialMat:" << E << endl;
    //decomposeEssentialMat(E,R,R1,t);
    recoverPose(E,corners0,corners1,cameraMatrix,R,t,essentialMask);
    Mat tWithScale, cameraMatrixInv, tCameraMatrixInv;
    cameraMatrixInv = cameraMatrix.inv();
    transpose(cameraMatrixInv,tCameraMatrixInv);
    ans = tCameraMatrixInv*E*cameraMatrixInv; 
    ans /= ans.at<double>(2,2);

    Mat Jcorners0transpose;
    Mat Jcorners0 = (Mat_<double>(3,1) << corners0[3].x,corners0[3].y,1);
    Mat Jcorners1 = (Mat_<double>(3,1) << corners1[3].x,corners1[3].y,1);
    transpose(Jcorners0,Jcorners0transpose);
    Mat new1 = Jcorners0transpose*F*Jcorners1;
    cout<< "new1: " <<new1<<endl;

    //drawing epilines
    

    Mat grey0_copy_pose, grey1_copy_pose, epilines0,epilines1;
    cvtColor(grey0, grey0_copy_pose, COLOR_GRAY2BGR);
    cvtColor(grey1, grey1_copy_pose, COLOR_GRAY2BGR); 
    computeCorrespondEpilines(corners0,1,F,epilines0);
    computeCorrespondEpilines(corners1,2,F,epilines1);
    cout <<"epilines0: " <<epilines0 <<endl;
    RNG rng(12345);


    for(size_t i=0; i <corners0.size();i++)
    {
        Point point0(0, (int)(-epilines0.at<double>(i,2)/epilines0.at<double>(i,1)));
        Point point1(grey0_copy_pose.cols,-(int)(epilines0.at<double>(i,2)+epilines0.at<double>(i,0)*grey0_copy_pose.cols)/epilines0.at<double>(i,1));
        //cout <<" Point1 " << point1 <<endl;
        Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255),CV_8UC3); //random color
        line(grey0_copy_pose, point0,point1,color,2);
    }
    imshow("epilines for 0",grey0_copy_pose);
    cout << "ans: " << ans << endl;
    cout <<"R: " << R <<endl;
    //cout <<"R1: " << R1 <<endl;
    cout <<"T: " << t <<endl;
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

    
    
