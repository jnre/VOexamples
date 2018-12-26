#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ./newpoints/left05.jpg | left_image 01.}"
    "{@input1 | ./newpoints/right05.jpg  | right_image 01.}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv ,keys);
    Mat src0 = imread(parser.get<String>("@input0"));
    Mat src1 = imread(parser.get<String>("@input1"));

    if (src0.empty() || src1.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "  <Input image>" <<endl;
        return -1;
    }
    
    Mat grey0, grey1, descriptors0, descriptors1;
    cvtColor(src0,grey0, COLOR_BGR2GRAY);
    cvtColor(src1,grey1, COLOR_BGR2GRAY);
    //cout<<"grey0: " <<src0.at<uchar>(0,0)<<endl;

    //for camera calibration
    FileStorage fs("../intrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open intrinsics\n");
    }
    Mat M1,M2,D1,D2;
    fs["M1"]>> M1;
    fs["M2"]>> M2;
    fs["D1"]>> D1;
    fs["D2"]>> D2;


    fs.open("../extrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open extrinsics\n");
    }
    Mat R,R1,R2,T,P1,P2,Q;
    fs["R"]>>R;
    fs["T"]>>T;
    fs["R1"]>>R1;
    fs["R2"]>>R2;
    fs["P1"]>>P1;
    fs["P2"]>>P2;
    fs["Q"]>>Q;
   
/*    Size imageSize = grey0.size();
    Mat rmap[2][2];
    initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    Mat canvas;
    double sf;
    int w,h;
    sf = 600./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h, w*2, CV_8UC3);

    imshow("rectified", canvas);*/

    //rectify dont seems to be working properlly
    Mat rmap[2][2],newgrey0,canvas,newgrey1;
    double sf;
    int w,h;
    sf = 600./MAX(grey0.size().width, grey0.size().height);
    w = cvRound(grey0.size().width*sf);
    h = cvRound(grey0.size().height*sf);
    canvas.create(h,w*2,CV_8UC1); //creating window
        
    initUndistortRectifyMap(M1,D1,R1,P1,grey0.size(),CV_32FC1,rmap[0][0],rmap[0][1]);
    initUndistortRectifyMap(M2,D2,R2,P2,grey1.size(),CV_32FC1,rmap[1][0],rmap[1][1]);

    //remap(grey0,newgrey0,rmap[0][0],rmap[0][1],INTER_LINEAR); //remapping with distortions into newgrey for img 0 //flipping problem
    //imshow("newgrey0",newgrey0);    //already flipped

    remap(grey0,newgrey0,rmap[0][0],rmap[0][1],INTER_LINEAR);
    Mat canvasPart0 = canvas(Rect(0,0,w,h)); //drawing box for img0
    resize(newgrey0, canvasPart0,canvasPart0.size(),0,0,INTER_AREA);  //map newgrey into canvas for img 0
    
    remap(grey1,newgrey1,rmap[1][0],rmap[1][1],INTER_LINEAR);
    Mat canvasPart1 = canvas(Rect(w,0,w,h));
    resize(newgrey1, canvasPart1,canvasPart1.size(),0,0,INTER_AREA);
    
    imshow("canvas01",canvas);
    imwrite("canvas05.jpg", canvas);

    //ORB detector of points
    Ptr<ORB> orbDetector = ORB::create();
    std::vector<KeyPoint> keypoint0, keypoint1;
    //detector->detect( grey, keypoints );
    orbDetector -> detectAndCompute(newgrey0, noArray(), keypoint0, descriptors0);  //
    orbDetector -> detectAndCompute(newgrey1, noArray(), keypoint1, descriptors1);  //

    //test print out keypoint0
    //cout <<"keypoint0: " << keypoint0[1].angle <<endl;    

    //convert descriptors to cv_32f for usage in flannbased matching
    if(descriptors0.type()!=CV_32F) {
    descriptors0.convertTo(descriptors0, CV_32F);
    }

    if(descriptors1.type()!=CV_32F) {
    descriptors1.convertTo(descriptors1, CV_32F);
    }
        
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<DMatch> > matches;
    matcher.knnMatch(descriptors0,descriptors1,matches,2);


    //filter using lowe's ratio test, other alternative is to use RANSAC
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches, inline_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    cout<<"keypoint0 size: " <<keypoint0.size()<<endl;
    cout<<"good match size: " <<good_matches.size()<<endl;
    cout<<"keypoint0: "<< fabs(keypoint0[good_matches[0].queryIdx].pt.y - keypoint1[good_matches[0].trainIdx].pt.y)<<endl;



    //cout<<"inline_matches:" <<inline_matches[0].queryIdx <<endl;
    std::vector<Point2f> goodmatchespoints0, goodmatchespoints1;   
    
    //query for matched points
    for(int i =0;i<good_matches.size();i++)
    {
        //test for horizontal matching ( no change in y va
        if(fabs((keypoint0[good_matches[i].queryIdx].pt.y)-(keypoint1[good_matches[i].trainIdx].pt.y))<=20.0)
        {              
            if((keypoint0[good_matches[i].queryIdx].pt.x)>(keypoint1[good_matches[i].trainIdx].pt.x))            
            {    
                inline_matches.push_back(good_matches[i]); 
            }
        }
    }
    
    //query for keypoints of matches
    for(int i=0;i<inline_matches.size();i++)
    {
        goodmatchespoints0.push_back(keypoint0[inline_matches[i].queryIdx].pt); 
        goodmatchespoints1.push_back(keypoint1[inline_matches[i].trainIdx].pt);        
        cout<<"goodmatches pt0:" << goodmatchespoints0[i] <<endl;
        cout<<"goodmatches pt1: " <<goodmatchespoints1[i] <<endl;
    }
    //https://github.com/stefanbo92/Visual-Odometry/blob/master/Code/localisation/src/main.cpp
    //http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A12-StereoVO.pdf
    //https://gist.github.com/cashiwamochi/8ac3f8bab9bf00e247a01f63075fedeb
    
    
    cv::Mat Rt0 = cv::Mat::eye(3,4,CV_64FC1);
    cv::Mat Rt1 = cv::Mat::eye(3,4,CV_64FC1);
    R.copyTo(Rt1.rowRange(0,3).colRange(0,3));
    T.copyTo(Rt1.rowRange(0,3).col(3));

    cout<<"newP2" << M2*Rt1<<endl;
    
       
    Mat pnts3D(1,goodmatchespoints0.size(),CV_64FC4);
    
    //cv::triangulatePoints(M1*Rt0,M2*Rt1,goodmatchespoints0,goodmatchespoints1,pnts3D);  //dont use p1 and p2 to triangulate
    cv::triangulatePoints(P1,P2,goodmatchespoints0,goodmatchespoints1,pnts3D);
    //cout<<"pnts3D raw: "<<pnts3D<<endl;
    //cout<<"camMatrixIE0: " << camMatrixIE0.row(1) <<endl;
    cout<<"pnts3d raw size" <<pnts3D.size().width<<endl;        //width is col. pnts3d original is (n x 4)- n is column
    pnts3D = pnts3D.t();
    cout<<"pnts3D no reshape" << pnts3D << endl;                //after transpose is (4xn)
       
    
    std::vector<Point3f> worldpnts;
    Mat worldmat;
    convertPointsFromHomogeneous(pnts3D,worldmat);
    cout << " worldmat at (1,0): " << worldmat.at<float>(1,0) << endl; 
       
    //cout<<"size of good matchespoints col: " <<worldmat.size().row<<endl;
    Mat newworldmat = worldmat.reshape(1);
     
    //float p;
    /*for(int i=0;i<goodmatchespoints0.size();i++)
    {
        float x = worldmat.at<float>(i,0);
        float y = worldmat.at<float>(i,1);
        float z = worldmat.at<float>(i,2);
        worldpnts.push_back(Point3f(x,y,z));
          
    }
     */   
    cout<<"size of worldmat: " <<worldmat.size()<<endl;     //original worldmat is channelbased so (1xn)
    cout<<" worldmat: " <<worldmat <<endl;
    cout<<"size of worldpnts: " <<newworldmat.size()<<endl;    //after reshape, (3xn)- 3 is column
    cout<<"worldpnts: " <<newworldmat<<endl;
    //cout<<"worldpnts3D: " << worldpnts3Darray <<endl;
    //-- Draw Matches
    Mat img_matches;
    drawMatches( newgrey0,keypoint0,newgrey1,keypoint1, inline_matches,img_matches,Scalar::all(-1),Scalar::all(-1),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); //
    cv::resize(img_matches,img_matches, cv::Size(w*2,h),0,0,CV_INTER_LINEAR);


    //pointcloud
    pcl::visualization::PCLVisualizer viewer("Viewer");
    viewer.setBackgroundColor (255, 255, 255);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.resize(newworldmat.rows);

    for(int i = 0; i <newworldmat.rows; i++)
    {
        pcl::PointXYZRGB &point = cloud->points[i];
        point.x = newworldmat.at<float>(i,0);
        point.y = -newworldmat.at<float>(i,1);
        point.z = -newworldmat.at<float>(i,2);
        point.r = 0;
        point.g = 0;
        point.b = 255;
        cout<<"pointcloud: " <<cloud->points[i] <<endl;
    }
    
    viewer.addPointCloud(cloud,"Triangulated Point Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"Triangulated Point Cloud");
    viewer.addCoordinateSystem(300.0);
    while (!viewer.wasStopped ()) {
    viewer.spin();
    }

/*    fs.open("newworldmat01.yml", FileStorage::WRITE);
    if(fs.isOpened())
    {
        fs<<"newworldmat01"<<newworldmat;
        fs.release();
    }
*/
    pcl::io::savePLYFile("test_pcd05.ply",*cloud);
    //pcl::io::savePNGFile("te st_png03.png",*cloud,"rgb");

    //-- Show detected (drawn) keypoints
    imshow("orbmatchpoint", img_matches);
    imwrite("newpoints05.jpg", img_matches);

    
    //cout << keypoints[1].pt.x << endl;
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
