#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ./left01.jpg | image 00.}"
    "{@input1 | ./right01.jpg  | image 01.}";

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

    //ORB detector of points
    Ptr<ORB> orbDetector = ORB::create();
    std::vector<KeyPoint> keypoint0, keypoint1;
    //detector->detect( grey, keypoints );
    orbDetector -> detectAndCompute(grey0, noArray(), keypoint0, descriptors0);
    orbDetector -> detectAndCompute(grey1, noArray(), keypoint1, descriptors1);

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
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    cout<<"keypoint0 size: " <<keypoint0.size()<<endl;
    cout<<"good match size: " <<good_matches.size()<<endl;
    cout<<"keypoint0: "<< keypoint0[0].pt<<endl;

    std::vector<Point2f> goodmatchespoints0, goodmatchespoints1;

    //query for matched points
    for(int i =0;i<good_matches.size();i++)
    {
               
        goodmatchespoints0.push_back(keypoint0[good_matches[i].queryIdx].pt);        
        cout<<"goodmatches pt0:" << goodmatchespoints0[i] <<endl;
        goodmatchespoints1.push_back(keypoint1[good_matches[i].trainIdx].pt);
        cout<<"goodmatches pt1: " <<goodmatchespoints1[i] <<endl;
        
    }

    FileStorage fs("intrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open intrinsics\n");
    }
    Mat M1,M2,D1,D2;
    fs["M1"]>> M1;
    fs["M2"]>> M2;
    fs["D1"]>> D1;
    fs["D2"]>> D2;


    fs.open("extrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open extrinsics\n");
    }
    Mat R1,R2,P1,P2,Q;
    fs["R1"]>>R1;
    fs["R2"]>>R2;
    fs["P1"]>>P1;
    fs["P2"]>>P2;
    fs["Q"]>>Q;


       
    Mat pnts3D(1,good_matches.size(),CV_64FC4);
    
    cv::triangulatePoints(P1,P2,goodmatchespoints0,goodmatchespoints1,pnts3D);
    //cout<<"camMatrixIE0: " << camMatrixIE0.row(1) <<endl;
    pnts3D = pnts3D.t();
    cout<<"pnts3D no reshape" << pnts3D << endl;
       
    
    std::vector<Point3f> worldpnts;
    Mat worldmat;
    convertPointsFromHomogeneous(pnts3D,worldmat);
    cout << " worldmat at (1,0): " << worldmat.at<float>(1,0) << endl; 
      
    
    
    //float p;
    /*for(int i=0;i<good_matches.size();i++)
    {
        float x = worldmat.at<float>(i,0);
        float y = worldmat.at<float>(i,1);
        float z = worldmat.at<float>(i,2);
        worldpnts.push_back(Point3f(x,y,z));
          
    }*/
        
    cout<<" worldmat: " <<worldmat <<endl;
    //cout<<"worldpnts: " <<worldpnts <<endl;
    //cout<<"worldpnts3D: " << worldpnts3Darray <<endl;
    //-- Draw Matches
    Mat img_matches;
    drawMatches( grey0,keypoint0,grey1,keypoint1, good_matches,img_matches,Scalar::all(-1),Scalar::all(-1),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    viz::Viz3d myWindow("Viz Demo");
    myWindow.spin();
    cout<<" First evet loop is over" <<endl;

    myWindow.showWidget("coordinate widget",viz::WCoordinateSystem());
    
    viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f),Point3f(1.0f,1.0f,1.0f));
    axis.setRenderingProperty(viz::LINE_WIDTH,4.0);
    myWindow.showWidget("linewidget", axis);

    viz::WCloud pointCloud(worldmat,viz::Color::green());
    pointCloud.setRenderingProperty(viz::POINT_SIZE,10.0);
    myWindow.showWidget("point cloud", pointCloud);

    while(!myWindow.wasStopped())
    {
        myWindow.spinOnce(1,true);
    }
    cout<<"Last event loop is over" << endl;
    


    //-- Show detected (drawn) keypoints
    imshow("orbmatchpoint", img_matches);
    imwrite("orbmatchpoint.jpg", img_matches);
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
