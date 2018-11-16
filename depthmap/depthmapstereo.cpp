#include <iostream>
#include <stdio.h>
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

static void print_help()
{
    printf("prints the stereo matching\n");
} 



const char* keys =
    
    "{@input0 | ../testfilesdataset/30.png | image 00.}"
    "{@input1 | ../testfilesdataset/31.png | image 01.}"
    "{help h |             | Print help message.}"
    "{max-disparity|16|}"
    "{blocksize|3|}"
    "{no-display|0|}"
    "{scale|1|}"
    "{intrinsic||}"
    "{extrinsic||}"
    "{o||}"
    "{p||}";

int main(int argc, char** argv)
{
    std::string img0_filename = "";
    std::string img1_filename = "";
    std::string disparity_filename = "";
   
    //for point cloud
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";    
    std::string point_cloud_filename = "";    

    enum{
        STEREO_BM=0,
        STEREO_SGBM=1};
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities =16;
    float scale;
   
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    CommandLineParser parser(argc,argv ,keys);
    if(parser.has("help"))
    {
        print_help();
        return 0;
    }   

    img0_filename = parser.get<std::string>("@input0");
    img1_filename = parser.get<std::string>("@input1"); 
    scale = parser.get<float>("scale");
    SADWindowSize = parser.get<int>("blocksize");
    numberOfDisparities = parser.get<int>("max-disparity");
    
    Mat img0 = imread(img0_filename,-1);

    Mat img1 = imread(img1_filename,-1);
    /*if (src0.empty() || src1.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << " <Input image>" <<endl;
        return -1;
    }*/
    
    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize :3;
    sgbm ->setBlockSize(sgbmWinSize);
    int cn = img0.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    Mat grey0, grey1,descriptors0, descriptors1;
    //cvtColor(src0,grey0, COLOR_BGR2GRAY);
    //cvtColor(src1,grey1, COLOR_BGR2GRAY);

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

    Mat grey2,disp8;
    //check stereomatch.cpp
    int type;
    sgbm->compute(img0,img1,grey2);
    cout << grey2 <<endl; 
    //cv::StereoMatcher::compute(grey0,grey1,grey2);
    grey2.convertTo(disp8, CV_8U);
    imshow("left image",img0);
    imshow("right image",img1);
    imshow("disparitiy",disp8);

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
