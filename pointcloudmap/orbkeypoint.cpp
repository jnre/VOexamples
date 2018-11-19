#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                    | Print help message.}"
    "{@input0 | ../testfilesdataset/30.png | image 00.}"
    "{@input1 | ../testfilesdataset/31.png | image 01.}";

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
        //cout<< "good matches index 0: "<<good_matches[i].trainIdx<<endl;
        //cout<< "keypoint1: " <<keypoint0[good_matches[i].trainIdx]<<endl;        
        goodmatchespoints0.push_back(keypoint0[good_matches[i].queryIdx].pt);        
        cout<<"goodmatches pt0:" << goodmatchespoints0[i] <<endl;
        goodmatchespoints1.push_back(keypoint1[good_matches[i].trainIdx].pt);
        cout<<"goodmatches pt1: " <<goodmatchespoints1[i] <<endl;
        //cout<<" good matches in 1: "<<good_matches[i].queryIdx<<endl;
        //cout<<" good match yes no and operator? " <<good_matches[i].imgIdx<<endl;
    }
    //-- Draw Matches
    Mat img_matches;
    drawMatches( grey0,keypoint0,grey1,keypoint1, good_matches,img_matches,Scalar::all(-1),Scalar::all(-1),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



    //-- Show detected (drawn) keypoints
    imshow("orbmatchpoint", img_matches);
    imwrite("orbmatchpoint.png", img_matches);
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
