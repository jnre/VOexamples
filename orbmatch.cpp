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
    "{@input0 | ./modelFactoryDataSet/00.png | image 00.}"
    "{@input1 | ./modelFactoryDataSet/01.png | image 01.}";

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

    //ORB work? haha
    Ptr<ORB> orbDetector = ORB::create();
    std::vector<KeyPoint> keypoint0, keypoint1;
    orbDetector -> detectAndCompute(grey0, noArray(), keypoint0, descriptors0);
    orbDetector -> detectAndCompute(grey1, noArray(), keypoint1, descriptors1);

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

    //flann matcher(hamming distance)
    //orb u got to use flann+ lsh(no idea)(42-53) or brute force +hamming (57-59,!42-50)
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //std::vector<std::vector<DMatch> > matches;
    //matcher->knnMatch(descriptors0,descriptors1,matches,2);

    //filter using low's ratio test, other alternative is to use RANSAC
    const float ratio_thresh = 0.8f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    //-- Draw Matches
    Mat img_matches;
    drawMatches( grey0,keypoint0,grey1,keypoint1, good_matches,img_matches,Scalar::all(-1),Scalar::all(-1),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    //-- Show detected (drawn) keypoints
    imshow("orb match", img_matches);
    imwrite("orb_matchFlann.png", img_matches);
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
