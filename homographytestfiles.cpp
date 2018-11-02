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
    "{@input0 | ./testfilesdataset/10.png | image 00.}"
    "{@input1 | ./testfilesdataset/11.png | image 01.}"
    "{@input2 | ./testfilesdataset/12.png | image 02.}"
    "{@input3 | ./testfilesdataset/13.png | image 03.}"
    "{@input4 | ./testfilesdataset/13.png | image 04.}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv ,keys);
    Mat src0 = imread(parser.get<String>("@input0"));
    Mat src1 = imread(parser.get<String>("@input1"));
    Mat src2 = imread(parser.get<String>("@input2"));
    Mat src3 = imread(parser.get<String>("@input3"));
    Mat src4 = imread(parser.get<String>("@input4"));

    if (src0.empty() || src1.empty() || src2.empty() || src3.empty() || src4.empty())
    {
        cout << "Could not open or find image!\n" <<endl;
        cout << "Usage: " <<argv[0] << "<Input image>" <<endl;
        return -1;
    }
    
    Mat grey0, grey1, grey2, grey3, grey4, descriptors0, descriptors1;
    cvtColor(src0,grey0, COLOR_BGR2GRAY);
    cvtColor(src1,grey1, COLOR_BGR2GRAY);
    cvtColor(src2,grey2, COLOR_BGR2GRAY);
    cvtColor(src3,grey3, COLOR_BGR2GRAY);
    cvtColor(src4,grey4, COLOR_BGR2GRAY);
    
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
    
    //ORB work? haha
    Ptr<ORB> orbDetector = ORB::create();
    std::vector<KeyPoint> keypoint0, keypoint1;
    orbDetector -> detectAndCompute(grey0, noArray(), keypoint0, descriptors0);
    orbDetector -> detectAndCompute(grey1, noArray(), keypoint1, descriptors1);

    //orb u got to use flann+ lsh(no idea)(42-54)(convert to 32f) or brute force +hamming (59-61,!42-50)    
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

    
    //for checkerboard
    std::vector<Point2f> corners0, corners1;
    Size patternSize(9,6);
    bool found0 = findChessboardCorners(grey0,patternSize,corners0);    //find pixel of corners
    bool found1 = findChessboardCorners(grey1,patternSize,corners1);
    Mat H1 = findHomography(corners0,corners1);                         //transform
    cout << "H1\n" << H1 <<endl;
    Mat grey0_warp;
    warpPerspective(grey0, grey0_warp,H1,grey0.size());
    Mat img_draw_matches;
    RNG rng(12345);
    hconcat(grey0,grey1,img_draw_matches);                               //join the 2 images
    Mat img_draw_matches1;
    cvtColor(img_draw_matches,img_draw_matches1, COLOR_GRAY2BGR);       // cvt to color mode to draw lines
    cout<<" img_draw_matches1 type: " << img_draw_matches1.type() <<endl;
    for (size_t i = 0; i< corners0.size(); i++)                         //find corner transform and draw lines
    {
        Mat pt0 = (Mat_<double>(3,1) << corners0[i].x,corners0[i].y,1); //points in 1st image change to x,y,z
        Mat pt1 = H1*pt0;                                               //transform it to 2nd image    
        pt1 /= pt1.at<double>(2);                                       //divide by scaling
        Point end((int)(grey0.cols +pt1.at<double>(0)), (int) pt1.at<double>(1)); //end point, col 1st(x-axis)
        Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255),CV_8UC3); //random color
        line(img_draw_matches1, corners0[i],end,color,2); //draw line from corners0 to end point, thick=2
    }
    imshow("Draw matches",img_draw_matches1);
    std::vector<Point3f> objectPoints;                                              //getting the objectPoints
    float squareSize = 24.41;
    for( int i = 0; i < patternSize.height; i++ )
       for( int j = 0; j < patternSize.width; j++ )
           objectPoints.push_back(Point3f(float(j*squareSize),float(i*squareSize), 0));
    Mat rvec0,tvec0, rvec1,tvec1;                     //pnp solver, no idea how object does anything tbh
    solvePnP(objectPoints,corners0, cameraMatrix,distCoeffs, rvec0,tvec0);
    solvePnP(objectPoints,corners1, cameraMatrix,distCoeffs, rvec1,tvec1);
    cout << "tvec0: " << tvec0 <<endl;
    //compute-poses
    Mat grey0_copy_pose, grey1_copy_pose;
    cvtColor(grey0, grey0_copy_pose, COLOR_GRAY2BGR);
    cvtColor(grey1, grey1_copy_pose, COLOR_GRAY2BGR);    
    Mat img_draw_poses;
    cv::drawFrameAxes(grey0_copy_pose, cameraMatrix, distCoeffs, rvec0, tvec0, 2*squareSize);
    cv::drawFrameAxes(grey1_copy_pose, cameraMatrix, distCoeffs, rvec1, tvec1, 2*squareSize);
    hconcat(grey0_copy_pose, grey1_copy_pose, img_draw_poses);
    imshow("Chessboard poses", img_draw_poses);
    //convert vector of rvec0 and rvec1 to matrix form.
    Mat R0,R1;
    Rodrigues(rvec0, R0);
    Rodrigues(rvec1, R1);
    //rotate and translate from frame 0 to 1
    Mat R_0to1, T_0to1;
    R_0to1 = R1*R0.t();
    T_0to1 = R1 * (-R0.t()*tvec0) + tvec1;
    //normal unit vector distance to camera from frame 0 to camera 0
    Mat normal = (Mat_<double>(3,1) << 0,0,1);
    Mat normal0 = R0*normal;
    //distance
    Mat origin(3,1,CV_64F, Scalar(0));
    Mat origin0 = R0*origin +tvec0;
    cout << "normal0: " << normal0 <<endl; 
    double d_inv0 = 1.0/normal0.dot(origin0);   //normal unit vector dot product distance
    Mat homography_euclidean = R_0to1 + d_inv0* T_0to1*normal.t();
    Mat homography = cameraMatrix * homography_euclidean * cameraMatrix.inv();
    homography /= homography.at<double>(2,2);
    homography_euclidean /= homography_euclidean.at<double>(2,2);
    std::vector<Mat> Rs_decomp_cam, ts_decomp_cam, normals_decomp_cam;
    int solutions_cam = decomposeHomographyMat(homography, cameraMatrix,Rs_decomp_cam,ts_decomp_cam,normals_decomp_cam);        //u lose scale 
    cout <<"Decompose homography matrix of camera transformation" <<endl;   //just decomposing back itself, hence similar answer, just directional issues
    for(int i=0;i<solutions_cam;i++)
    {
        double factor_d1 = 1/d_inv0;
        cout <<"Solution " <<i << ":" << endl;
        cout <<"r from decomposition: " <<Rs_decomp_cam[i] <<endl;
        cout <<"r from camera displacement: " <<R_0to1 <<endl;
        cout <<"t from decomposition: " << ts_decomp_cam[i] * factor_d1 << endl;
        cout <<"t from camera displacement: " << T_0to1 << endl;
        
    }
    
    

    
    //decompose homography transformation needs Homography and cameraMatrix
    std::vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions = decomposeHomographyMat(H,cameraMatrix,Rs_decomp,ts_decomp,normals_decomp);
    cout<< "Decompose homography transformation of homography through feature matching"<<endl;
    for (int i=0;i< solutions;i++)
    {
        //let scale d be 1 for now
        double factor_d1 = 1/d_inv0;
        cout << "Solution " << i<< ":"<<endl;
        cout << "rvec from homography decomposition: " << Rs_decomp[i] << endl;
        cout << "tvec from homography decomposition: " <<ts_decomp[i].t() <<"and scaled by d(1): " <<factor_d1*ts_decomp[i].t()<<endl;
    }
        
            
    imshow("img match for testfiles-orb ",img_matches);
    imwrite("img match for testfiles-orb.png ",img_matches);

    //-- Show detected (drawn) keypoints
    imshow("homographytestfiles", src0t);
    imwrite("homographytestfiles.png", src0t);

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
