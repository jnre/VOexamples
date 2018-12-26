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

/*const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ../modelFactoryDataSet/00.png | image 00.}"
    "{@input1 | ../modelFactoryDataSet/01.png | image 01.}";
*/
int main(int argc, char** argv)
{
    /*CommandLineParser parser(argc,argv ,keys);
    Mat src0 = imread(parser.get<String>("@input0"));
    Mat src1 = imread(parser.get<String>("@input1"));

    if (src0.empty() || src1.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "<Input image>" <<endl;
        return -1;
    }
    */
    VideoCapture cap0(0);
    VideoCapture cap1(1);
    if(!cap0.isOpened())
        return -1;
    if(!cap1.isOpened())
        return -1;
    Mat previousframe0, previousframe1;

    for(;;)
    {
        
        Mat frame0,frame1,grey0, grey1, descriptors0, descriptors1;
        cap0>>frame0;
        cap1>>frame1;        
        cvtColor(frame0,grey0, COLOR_BGR2GRAY);
        cvtColor(frame1,grey1, COLOR_BGR2GRAY);
    
        //ORB work? haha
        Ptr<ORB> orbDetector = ORB::create();
        std::vector<KeyPoint> keypoint0, keypoint1;
        orbDetector -> detectAndCompute(grey0, noArray(), keypoint0, descriptors0);
        orbDetector -> detectAndCompute(grey1, noArray(), keypoint1, descriptors1);

        //for camera calibration        
        FileStorage fs("./intrinsics.yml", FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open intrinsics\n");
        }
        Mat M1,M2,D1,D2;
        fs["M1"]>> M1;
        fs["M2"]>> M2;
        fs["D1"]>> D1;
        fs["D2"]>> D2;


        fs.open("./extrinsics.yml", FileStorage::READ);
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

        Size imageSize = grey0.size();
        Mat rmap[2][2];
        initUndistortRectifyMap(M1, D1, R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
        initUndistortRectifyMap(M2, D2, R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
        //convert descriptors to cv_32f for usage in flannbased matching
        /*if(descriptors0.type()!=CV_32F) {
        descriptors0.convertTo(descriptors0, CV_32F);
        }

        if(descriptors1.type()!=CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
        }
        
        cv::FlannBasedMatcher matcher;
        std::vector<std::vector<DMatch> > matches;
        matcher.knnMatch(descriptors0,descriptors1,matches,2);
        */
        //flann matcher(hamming distance)
        //orb u got to use flann+ lsh(no idea)(42-53) or brute force +hamming (57-59,!42-50)
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<std::vector<DMatch> > matches;
        matcher->knnMatch(descriptors0,descriptors1,matches,2);

        //filter using low's ratio test, other alternative is to use RANSAC
        const float ratio_thresh = 0.7f;
        std::vector<DMatch> good_matches,inline_matches;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        for(int i =0; i< good_matches.size();i++)
        {
            if(fabs((keypoint0[good_matches[i].queryIdx].pt.y)-(keypoint1[good_matches[i].trainIdx].pt.y))<=20.0)
            {
                if((keypoint0[good_matches[i].queryIdx].pt.x)>(keypoint1[good_matches[i].trainIdx].pt.x))
                {
                    inline_matches.push_back(good_matches[i]);
                }
            }
        }

        //keypoint0 refers to img1, with queryIdx refering to matchings in img1        
        std::vector<Point2f> goodmatchespoints0, goodmatchespoints1;

        //query for keypoints of matches
        for(int i=0;i<inline_matches.size();i++)
        {
            goodmatchespoints0.push_back(keypoint0[inline_matches[i].queryIdx].pt); 
            goodmatchespoints1.push_back(keypoint1[inline_matches[i].trainIdx].pt);        
            //cout<<"goodmatches pt0:" << goodmatchespoints0[i] <<endl;
            //cout<<"goodmatches pt1: " <<goodmatchespoints1[i] <<endl;
        }


        //-- Draw Matches
        Mat img_matches;
        drawMatches( grey0,keypoint0,grey1,keypoint1, inline_matches,img_matches,Scalar::all(-1),Scalar::all(-1),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        //cout<< "size of img_matches" << img_matches.size()<<endl;
        //-- Show detected (drawn) keypoints
        namedWindow("orb match", WINDOW_NORMAL);
        //resizeWindow("orb match", 2560,960);
        imshow("orb match", img_matches);
        imwrite("orb_matchFlannreal.png", img_matches);
        //imwrite("orb_matchBrute.png", img_matches);
        //cout << keypoints[1].pt.x << endl;

        Mat pnts3D(1,goodmatchespoints0.size(),CV_64FC4);
        //show triangulated points
        if(!goodmatchespoints0.empty())
        {        
            triangulatePoints(P1,P2,goodmatchespoints0,goodmatchespoints1, pnts3D); //gives (nx4), but displayed as 1 since C4
            pnts3D=pnts3D.t(); //convert to single channel (4xn)
            Mat world_mat;
            convertPointsFromHomogeneous(pnts3D,world_mat); //convert to 3xn displayed as 1 since C3
            Mat newworld_mat = world_mat.reshape(1);    //newworld_mat with correct shape 
            cout<<"worldpnts: " <<newworld_mat<<endl;
        }
        if(previousframe0.data)
        {
                       
            Size sz0 = previousframe0.size();
            Size sz1 = previousframe1.size();
            Mat contpreviousframe(sz0.height,sz0.width+sz1.width,CV_8UC1);
            previousframe0.copyTo(contpreviousframe(Rect(0,0,sz0.width,sz0.height)));
            previousframe1.copyTo(contpreviousframe(Rect(sz0.width,0,sz1.width,sz1.height)));
            namedWindow("previousframe", WINDOW_NORMAL);            
            imshow("previousframe",contpreviousframe);
             
        }


        previousframe0 = grey0.clone();
        previousframe1 = grey1.clone();      
        waitKey(30);
        
    }
    return 0;
}

#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif   
