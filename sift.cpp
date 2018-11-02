#include <iostream>
#include <opencv2/highgui.hpp>
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv::xfeatures2d;
using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    CommandLineParser parser( argc, argv, "{@input | ./modelFactoryDataSet/01.png | input image}" );
    Mat src = imread( parser.get<String>( "@input" ));
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    //SIFT work  
    Mat grey;
    cvtColor(src,grey,COLOR_BGR2GRAY); 
    cv::Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(grey, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(grey, keypoints, output, Scalar::all(-1) , DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("results",output);
    cv::imwrite("sift_result.png", output);
    cv::waitKey();
    

    return 0;
}
