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
    "{@input0 | ../modelFactoryDataSet/00.png | image 00.}"
    "{@input1 | ../modelFactoryDataSet/01.png | image 01.}";

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
    /*if(descriptors0.type()!=CV_32F) {
    descriptors0.convertTo(descriptors0, CV_32F);
    }

    if(descriptors1.type()!=CV_32F) {
    descriptors1.convertTo(descriptors1, CV_32F);
    }
    */
    //cv::FlannBasedMatcher matcher;
    //std::vector<std::vector<DMatch> > matches;
    //matcher.knnMatch(descriptors0,descriptors1,matches,2);

    //flann matcher(hamming distance)
    //orb u got to use flann+ lsh(no idea)(42-54)(convert to 32f) or brute force +hamming (58-60,!42-50)
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::vector<DMatch> > matches;
    matcher->knnMatch(descriptors0,descriptors1,matches,2);

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

    //additional part from orbmatch
    //RANSAC for homography transformation
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoint0[ good_matches[i].queryIdx ].pt );// why i can put goodmatches into keypoints
        scene.push_back( keypoint1[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    cout <<"H:\n" << H <<endl;
    Mat src0t,src1t;
    warpPerspective( grey0,src0t, H, Size(grey0.cols, grey0.rows));

    
    //stiching
    //Mat stich(Size(grey0.cols*2, grey0.rows*2),CV_8U);
    //Mat roi0(stich, Rect(0,0,grey1.cols, grey1.rows));
    //Mat roi1(stich, Rect(0,0,src0t.cols, src0t.rows));
    //src0t.copyTo(roi1);
    //grey1.copyTo(roi0);
    //imshow("FINAL",stich);

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
    

    //decompose homography transformation needs Homography and cameraMatrix
    std::vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions = decomposeHomographyMat(H,cameraMatrix,Rs_decomp,ts_decomp,normals_decomp);
    cout<< "Decompose homography transformation of homography through feature matching"<<endl;
    for (int i=0;i< solutions;i++)
    {
        //let scale d be 1 for now
        Mat rvec_decomp;
        Rodrigues(Rs_decomp[i], rvec_decomp);
        cout << "Solution " << i<< ":"<<endl;
        cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
        cout << "tvec from homography decomposition: " <<ts_decomp[i].t() <<"and scaled by d(1): " <<(int)1.0*ts_decomp[i].t()<<endl;
    }
        
            



    //-- Show detected (drawn) keypoints
    imshow("homography0", src0t);
    imwrite("homography0.png", src0t);

    //imshow("homograph1",src1t);
    //imwrite("homography1.png",src1t);
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
