#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void cornerHarris_demo(int, void*);


Mat src,src_gray;
int thresh = 200;
int max_thresh = 255;

const string source_window = "Source image";
const string corners_window = "Corners detected";




int main( int argc, char** argv)
{
    CommandLineParser parser( argc, argv, "{@input | ../modelFactoryDataSet/01.png | input image}");
    src = imread( parser.get<String>("@input"));
    if (src.empty())
    {
       cout << "Could not open file\n"<< endl;
       cout << "Usage: " << argv[0] << " <Input image>" << endl;
       return -1;
    }
    cvtColor(src,src_gray, COLOR_BGR2GRAY); //convert to gray -> src_gray

    namedWindow(source_window); //original image
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo); //only above thresh value to be considered corner
    imshow(source_window, src ); //show original image
    cornerHarris_demo(0,0);
    
    waitKey(); //wait infinitly as its a picture
    return 0;
}

void cornerHarris_demo(int,void*)
{
    int blockSize = 2;
    int apertureSize =3;
    double k= 0.04;

    Mat dst = Mat::zeros (src.size(), CV_32FC1);// a mat of zeros filled same size and 32floating pt 1 channel
    cornerHarris( src_gray, dst, blockSize, apertureSize,k); //2by2 gradient covariance into dst

    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); // normalize between 0,255 segmentations(0-1)
    convertScaleAbs( dst_norm, dst_norm_scaled); //convert to 8 bit CV_8U for viewing

    for( int i =0; i< dst_norm.rows; i++)
    {
        for(int j=0; j <dst_norm.cols; j++)
        {
            if( (int)dst_norm.at<float>(i,j) > thresh) //mat at this value greater than thresh
            {
                circle(dst_norm_scaled, Point(j,i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
    
    
}
    
    

    
