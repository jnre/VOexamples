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

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv , "{@input | ../modelFactoryDataSet/01.png | input image}");
    Mat src = imread(parser.get<String>("@input"));
    if (src.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "<Input image>" <<endl;
        return -1;
    }

    Mat grey, output;
    cvtColor(src,grey,COLOR_BGR2GRAY);

    std::vector<KeyPoint> keypoints;
    FAST(grey,keypoints,50,false); //threshold 50, no nonmaxsuppression as i dont understand it anyway
    cv::drawKeypoints(grey,keypoints,output);
    cv::imshow("results",output);
    cv::imwrite("fast_results.png",output);
    cv::waitKey();

    return 0;
}    
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif 
