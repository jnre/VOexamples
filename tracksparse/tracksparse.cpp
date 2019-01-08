#include <iostream>
#include <list>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/photo.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <boost/thread/thread.hpp>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;

const double nn_match_ratio = 0.8f;
const double ransac_thresh = 2.5f;
//camera parameters
Mat M1,M2,D1,D2;
Mat R,R1,R2,T,P1,P2,Q;
//remapping
Mat rmap[2][2];
int wid,hei;
int indicator_count = 0;

class Tracker
{
    private:
        Ptr<Feature2D> detector;
        Ptr<DescriptorMatcher> matcher;
        
        vector<Mat> left_frames, left_descriptors, right_frames,right_descriptors;
        vector<vector<KeyPoint>> left_keypoints, right_keypoints;


        //vector of point cloud, why need aligned_allocator
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr , Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr>> cloud_vector;
        int size_cloud_vector;
    
    public:
        Tracker(Ptr<Feature2D> detector, Ptr<DescriptorMatcher> matcher){
            this->detector = detector;
            this->matcher = matcher;
        }

        void setLeftFrame(const Mat frame)
        {
            Mat raw_left_frame, raw_left_descriptors;
            vector<KeyPoint> raw_left_keypoints;
            remap(frame,raw_left_frame,rmap[0][0],rmap[0][1],INTER_LINEAR);
            detector->detectAndCompute(raw_left_frame,noArray(),raw_left_keypoints,raw_left_descriptors);
            left_frames.push_back(raw_left_frame);
            left_descriptors.push_back(raw_left_descriptors);
            left_keypoints.push_back(raw_left_keypoints);
            cout<< raw_left_descriptors.size() <<endl;
            
        }
        
        void setRightFrame(const Mat frame)
        {
            Mat raw_right_frame, raw_right_descriptors;
            vector<KeyPoint> raw_right_keypoints;
            remap(frame,raw_right_frame,rmap[1][0],rmap[1][1],INTER_LINEAR);
            detector->detectAndCompute(raw_right_frame,noArray(),raw_right_keypoints,raw_right_descriptors);
            right_frames.push_back(raw_right_frame);
            right_descriptors.push_back(raw_right_descriptors);
            right_keypoints.push_back(raw_right_keypoints);
            Point2f p = right_keypoints[0][0].pt;
            cout << "p: "<< p <<endl;
        }

        void frameMatchingLeft()
        {
            // Mat current_frame;
            // remap(frame,current_frame,rmap[0][0],rmap[0][1],INTER_LINEAR);
            
            
            // vector<KeyPoint> keypoints;
            // Mat descriptors;
            // detector->detectAndCompute(current_frame,noArray(),keypoints,descriptors);
            // int keypoints_size = keypoints.size();
            cout <<"left_keypoints size of 0: " << left_keypoints[0].size() <<endl;
            cout <<"left_keypoints size of 1: " << left_keypoints[1].size() <<endl;
            //match between previous Lframe and current Lframe
            vector<vector<DMatch>> matches;
            matcher->knnMatch(left_descriptors[0],left_descriptors[1],matches,2);
            vector<KeyPoint> matched0,matched1;
            vector<DMatch> inlier_matches;
            for(int i =0; i<matches.size();i++){
                if(matches[i][0].distance<nn_match_ratio*matches[i][1].distance){
                    //keypoint in first frame stored in matched 0, passing distance thres
                    matched0.push_back(left_keypoints[0][matches[i][0].queryIdx]);
                    //keypoint in current frame stored in matched 1, passing distance thres
                    matched1.push_back(left_keypoints[1][matches[i][0].trainIdx]);
                    
                }
            }
            cout << "matches between left frames" << matched0.size() <<endl;
            
            //ransac 
            Mat inlier_mask, homography;
            vector<KeyPoint> inlier0,inlier1;
            vector<Point2f> converted_matched0, converted_matched1;
            KeyPoint::convert(matched0,converted_matched0);
            KeyPoint::convert(matched1,converted_matched1);
            //must have more than 4 points
            if(matched0.size()>=4){
                homography = findHomography(converted_matched0, converted_matched1,RANSAC,ransac_thresh,inlier_mask);
            }
            else
            {
                vector<KeyPoint> emptykeypoints;
                cout << "lost transformation" <<endl;
                
            }  
            for(int i =0; i <matched0.size();i++){
                //see if mask gives a value of one, where one means it is correct matched
                if(inlier_mask.at<uchar>(i)){
                    int new_i = static_cast<int>(inlier0.size());
                    inlier0.push_back(matched0[i]);
                    inlier1.push_back(matched1[i]);
                    //store matches of correct RANSAC
                    inlier_matches.push_back(DMatch(new_i,new_i,0));
                }
            }
            cout<< "inlier_size after RANSAC between left images: " << inlier_matches.size() <<endl;

            Mat res;
            drawMatches(left_frames[0],inlier0,left_frames[1],inlier1,inlier_matches,res,Scalar(255,0,0),Scalar(255,0,0));
            cv::resize(res,res,cv::Size(wid*2,hei),0,0,CV_INTER_LINEAR);
            imshow("leftside is prev, right side is now", res);
            waitKey();

            // give left_keypoints[0] the good matches in current time
            left_keypoints.erase(left_keypoints.begin());
            left_keypoints.erase(left_keypoints.begin());
            left_keypoints.push_back(inlier1);
            // keep frame[0] and descriptors [0] of current 
            left_frames.erase(left_frames.begin());
            left_descriptors.erase(left_descriptors.begin());
            left_descriptors.erase(left_descriptors.begin());
            Mat revamp_left_descriptors;
            detector->compute(left_frames[0],left_keypoints[0],revamp_left_descriptors);
            left_descriptors.push_back(revamp_left_descriptors);
             
            
        }

        // void increaseOrbFeatures(){
            
        //     Mat temp_descriptors;
        //     vector<KeyPoint> temp_keypoints;
        //     cout<< " feature points before adding" << left_keypoints[0].size() <<endl;
        //     detector->detectAndCompute(left_frames[0],noArray(),temp_keypoints,temp_descriptors);

        // }

        void leftRightCurrentMatching(){
                       
            // Mat new_left_frame, new_right_frame;
            // remap(left_frame,new_left_frame,rmap[0][0],rmap[0][1],INTER_LINEAR);
            // // Mat canvasPart0 = canvas(Rect(0,0,wid,hei)); //drawing box for img0
            // // resize(newgrey0, canvasPart0,canvasPart0.size(),0,0,INTER_AREA);
            // remap(right_frame,new_right_frame,rmap[1][0],rmap[1][1],INTER_LINEAR);
            // Mat canvasPart1 = canvas(Rect(wid,0,wid,hei));
            // resize(newgrey1, canvasPart1,canvasPart1.size(),0,0,INTER_AREA);
            //imshow("canvas01",canvas);


            //right current image detection
            // vector<KeyPoint> right_keypoints;
            // Mat left_descriptors,right_descriptors;
            // detector->detectAndCompute(new_right_frame,noArray(),right_keypoints,right_descriptors);
            // int right_keypoints_size = right_keypoints.size();
            
            //match between left and right current frame, 0 =left, 1 = right
            
            cout << "size of left_keypoints" << left_keypoints[0].size() <<endl;
            cout << "size of right_keypoints" << right_keypoints[0].size() <<endl;
            cout << "size of left_descriptors" << left_descriptors[0].size() <<endl;
            cout << "size of right_descriptors" << right_descriptors[0].size() <<endl;
            vector<vector<DMatch>> matches;
            matcher->knnMatch(left_descriptors[0],right_descriptors[0],matches,2);
            vector<KeyPoint> matched0,matched1;
            vector<DMatch> inlier_matches;
            for(int i =0; i<matches.size();i++){
                if(matches[i][0].distance<nn_match_ratio*matches[i][1].distance){
                    //keypoint in first frame stored in matched 0, passing distance thres
                    matched0.push_back(left_keypoints[0][matches[i][0].queryIdx]);
                    //keypoint in current frame stored in matched 1, passing distance thres
                    matched1.push_back(right_keypoints[0][matches[i][0].trainIdx]);
                    
                }
            }
            cout << "matches between left/right: " << matched0.size() <<endl;
            
            //ransac 
            Mat inlier_mask, homography;
            vector<KeyPoint> inlier0,inlier1;
            vector<Point2f> converted_matched0, converted_matched1;
            KeyPoint::convert(matched0,converted_matched0);
            KeyPoint::convert(matched1,converted_matched1);
            //must have more than 4 points
            if(matched0.size()>=4){
                homography = findHomography(converted_matched0, converted_matched1,RANSAC,ransac_thresh,inlier_mask);
            }
            else
            {
                vector<KeyPoint> emptykeypoints;
                cout << "lost transformation" <<endl;
                
            }  
            for(int i =0; i <matched0.size();i++){
                //see if mask gives a value of one, where one means it is correct matched
                if(inlier_mask.at<uchar>(i)){
                    int new_i = static_cast<int>(inlier0.size());
                    inlier0.push_back(matched0[i]);
                    inlier1.push_back(matched1[i]);
                    //store matches of correct RANSAC
                    inlier_matches.push_back(DMatch(new_i,new_i,0));
                }
            }
            cout<< "inlier_size of l/r after RANSAC: " << inlier_matches.size() <<endl;
            
            Mat res;
            drawMatches(left_frames[0],inlier0,right_frames[0],inlier1,inlier_matches,res,Scalar(255,0,0),Scalar(255,0,0));
            cv::resize(res,res,cv::Size(wid*2,hei),0,0,CV_INTER_LINEAR);
            imshow("leftside is left now, right side is right now", res);

            // reset right to empty vectors
            right_frames.erase(right_frames.begin());
            right_keypoints.erase(right_keypoints.begin());
            right_descriptors.erase(right_descriptors.begin());
            cout<<"size of right:" <<right_keypoints.size() <<endl;
            //tomatchR_keypoints.erase(tomatchR_keypoints.begin());
            //tomatchR_descriptors.erase(tomatchR_descriptors.begin());
            waitKey();            
            
            //triangulation
            std::vector<Point2f> goodmatchespoints0, goodmatchespoints1;
            KeyPoint::convert(inlier0,goodmatchespoints0);
            KeyPoint::convert(inlier1,goodmatchespoints1);
            Mat pnts3D(1,goodmatchespoints0.size(),CV_64FC4);
            cv::triangulatePoints(P1,P2,goodmatchespoints0,goodmatchespoints1,pnts3D);
            pnts3D = pnts3D.t();
            vector<Point3f> worldpnts;
            Mat worldmat;
            convertPointsFromHomogeneous(pnts3D,worldmat);
            Mat newworldmat = worldmat.reshape(1);

            //pcl
            //pcl::visualization::PCLVisualizer viewer("Viewer");
            //viewer.setBackgroundColor (255, 255, 255);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            cloud->width = newworldmat.rows;
            cloud->height = 1;
            cloud->is_dense = false;
            cloud->points.resize(newworldmat.rows);
            
            for(int i = 0; i <newworldmat.rows; i++)
            {
                pcl::PointXYZ &point = cloud->points[i];
                point.x = newworldmat.at<float>(i,0);
                point.y = -newworldmat.at<float>(i,1);
                point.z = -newworldmat.at<float>(i,2);
                //point.r = 0;
                //point.g = 0;
                //point.b = 255;
                cout<<"pointcloud: " <<cloud->points[i] <<endl;
            }
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 0, 255);
            //viewer.addPointCloud(cloud,single_color,"Triangulated Point Cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"Triangulated Point Cloud");
            //viewer.addCoordinateSystem(300.0);
            char save_cloud[100];
            sprintf(save_cloud,"./newpointsmove/test1_%02d.pcd",indicator_count);
            pcl::io::savePCDFileASCII(save_cloud,*cloud);
            indicator_count++;
            // while (!viewer.wasStopped ()) {
            //     viewer.spinOnce(100);
            // }
            cloud_vector.push_back(cloud);
            // newworldmat.release();
            // worldmat.release();
            // pnts3D.release();
        }
   
        void getOdometryMatch(){
            pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
            icp.setInputSource(cloud_vector[0]);
            icp.setInputTarget(cloud_vector[1]);
            icp.setMaximumIterations(300);
            icp.setTransformationEpsilon(1e-9);
            icp.setMaxCorrespondenceDistance(150);
            icp.setEuclideanFitnessEpsilon(1);
            icp.setRANSACOutlierRejectionThreshold(1.5);
            pcl::PointCloud<pcl::PointXYZ> Final;
            icp.align(Final);    
            std::cout<<"hasconverged:" << icp.hasConverged() << std::endl;
            std::cout<<"score: " << icp.getFitnessScore() << std::endl;
            std::cout<< "transform between" <<indicator_count <<"and"<< indicator_count+1 <<std::endl;
            std::cout<< icp.getFinalTransformation() << std::endl;        

        
        }

        int getSizeCloudVector(){
            return size_cloud_vector = cloud_vector.size();
        }
};

int main(int argc, char **argv)
{
    //string of name of files
    std::vector<cv::String> left_img, right_img;
    for(int i=0;i<20;i++){
        char get_img_left[100];
        char get_img_right[100];
        sprintf(get_img_left,"./newpoints/left%02d.jpg",i);
        sprintf(get_img_right,"./newpoints/right%02d.jpg",i);
        left_img.push_back(get_img_left);
        right_img.push_back(get_img_right);   
    }

    //camera
    FileStorage fs("../intrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open intrinsics\n");
    }   
    fs["M1"]>> M1;
    fs["M2"]>> M2;
    fs["D1"]>> D1;
    fs["D2"]>> D2;
    fs.open("../extrinsics.yml", FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open extrinsics\n");
    }   
    fs["R"]>>R;
    fs["T"]>>T;
    fs["R1"]>>R1;
    fs["R2"]>>R2;
    fs["P1"]>>P1;
    fs["P2"]>>P2;
    fs["Q"]>>Q;

    //storing files into Mat
    std::vector<Mat> left_mat, right_mat;
    for(int j=0;j<left_img.size();j++){
        left_mat.push_back(imread(left_img[j]));
        right_mat.push_back(imread(right_img[j]));
    }

    //convert all to greyscale-grey0 and grey1
    vector<Mat> grey0, grey1;    
    for(int i =0;i<left_mat.size();i++){

        Mat temp_grey0,temp_grey1;
        cvtColor(left_mat[i], temp_grey0, COLOR_BGR2GRAY);
        grey0.push_back(temp_grey0);
        cvtColor(right_mat[i], temp_grey1, COLOR_BGR2GRAY);
        grey1.push_back(temp_grey1);
    }

    //create rmap for remapping
    double scaling_factor;
    scaling_factor = 800./MAX(grey0[0].size().width,grey0[0].size().height);
    wid = cvRound(grey0[0].size().width*scaling_factor);
    hei = cvRound(grey1[0].size().height*scaling_factor);
    // canvas.create(hei,wid*2,CV_8UC1);
    initUndistortRectifyMap(M1,D1,R1,P1,grey0[0].size(),CV_32FC1,rmap[0][0],rmap[0][1]);
    initUndistortRectifyMap(M2,D2,R2,P2,grey1[0].size(),CV_32FC1,rmap[1][0],rmap[1][1]);

    //ORB detector
    Ptr<ORB> orb_detector = ORB::create();
    Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    //create tracking
    Tracker orb_tracker(orb_detector,orb_matcher);
    orb_tracker.setLeftFrame(grey0[3]);
    orb_tracker.setLeftFrame(grey0[4]);
    orb_tracker.frameMatchingLeft();
    orb_tracker.setRightFrame(grey1[4]);
    orb_tracker.leftRightCurrentMatching();
    
    orb_tracker.setLeftFrame(grey0[5]);
    orb_tracker.frameMatchingLeft();
    orb_tracker.setRightFrame(grey1[5]);
    orb_tracker.leftRightCurrentMatching();
    cout<< "size of cloud vector" << orb_tracker.getSizeCloudVector() <<endl;

    if(orb_tracker.getSizeCloudVector() == 2){
        cout << " 2 cloud for icp matching"<<endl;
        orb_tracker.getOdometryMatch();

    }

    //orb_tracker.leftRightmatching();
    // if(!left_keypoints.size()){
    //     cout << " no good matches" << endl;
    // }
    // else{
    //     cout << "left_currentframe size: " << left_keypoints.size() <<endl;
    //     orb_tracker.leftRightmatching(left_keypoints,left_mat[1],right_mat[1]);
    // }
    
}
#endif