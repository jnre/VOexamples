#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp" 
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/video/tracking.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <boost/thread/thread.hpp>



using namespace cv;
using namespace std;

const double nn_match_ratio = 0.75f;
const double ransac_thresh = 3.0f;
//camera parameters
Mat M1,M2,D1,D2;
Mat R,R1,R2,T,P1,P2,Q;
//remapping
Mat rmap[2][2];
int wid,hei;

int writing =0;
int g_count2 = 0;
int cloud_count =0;

void setLeftFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors);
void setRightFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors);
vector<KeyPoint> compareInlinePts(vector<vector<KeyPoint>> inline_new,vector<KeyPoint> final_keypoint);
//void calOpticalFlow();
vector<KeyPoint> matchingFunc04(Mat old_descriptors,Mat newest_descriptors,vector<KeyPoint> old_keypoints,vector<KeyPoint> newest_keypoints,Mat old_img,Mat newest_img);
Mat matchingFuncTriangulation(vector<KeyPoint> left_keypoints,vector<KeyPoint> right_keypoints,Mat left_img,Mat right_img);
pcl::PointCloud<pcl::PointXYZ>::Ptr PclDisplay(const Mat &triangulated);
vector<KeyPoint> calOpticalFlow(Mat old_img,Mat new_img,vector<KeyPoint> old_keypoints);
void getOdometryMatch(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud0,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1);
Mat getKeypointDescriptors(Mat img,vector<KeyPoint> keypoints);

int main(int argc, char **argv)
{
    //Left camera
    VideoCapture cap0(0);
    if(!cap0.isOpened())  // check if we succeeded
        return -1;

    //Right camera
    VideoCapture cap1(1);
    if(!cap1.isOpened())
        return -1;

    vector<Mat>  left_descriptors0(2),right_descriptors0(2);
    vector<vector<KeyPoint>> left_keypoints0(2),right_keypoints0(2);
    Mat frame,frame1; 
    vector<Mat> img0(2),img1(2);
    vector<vector<KeyPoint>> inline_newL(2), inline_newR(2);
    //vector<KeyPoint> left_keypoints_formatch, right_keypoints_formatch;
    Mat triangulated_points0, triangulated_points1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0, cloud1;
    int g_count = 0;
    


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

    for(;;){

        cap0 >> frame;
        img0[0]=frame.clone();
        //frame.copyTo(img0[0]);     
        //imshow("imgL",img0[g_count]);   
        cap1 >>frame1;
        img1[0]=frame1.clone();
        //frame.copyTo(img1[0]);

        cvtColor(img0[0],img0[0],COLOR_BGR2GRAY);
        cvtColor(img1[0],img1[0],COLOR_BGR2GRAY);
        //create rmap for remapping
        double scaling_factor;
        scaling_factor = 800./MAX(img0[0].size().width,img0[0].size().height);
        wid = cvRound(img0[0].size().width*scaling_factor);
        hei = cvRound(img0[0].size().height*scaling_factor);

        initUndistortRectifyMap(M1,D1,R1,P1,img0[0].size(),CV_32FC1,rmap[0][0],rmap[0][1]);
        initUndistortRectifyMap(M2,D2,R2,P2,img0[0].size(),CV_32FC1,rmap[1][0],rmap[1][1]);
        remap(img0[0],img0[0],rmap[0][0],rmap[0][1],INTER_LINEAR);
        remap(img1[0],img1[0],rmap[1][0],rmap[1][1],INTER_LINEAR);
        cv::resize(img0[0],img0[0],cv::Size(wid,hei),0,0,CV_INTER_LINEAR);
        cv::resize(img1[0],img1[0],cv::Size(wid,hei),0,0,CV_INTER_LINEAR);
        
        //keypoints and descriptors pass by reference
        //optical
        // if(g_count == 0){
            setLeftFrame(img0[0],left_keypoints0[0],left_descriptors0[0]);
            setRightFrame(img1[0],right_keypoints0[0],right_descriptors0[0]);
            cout<<"keypoints size greater than 0.001 for left"<< left_keypoints0[0].size()<<endl;
            cout<<"keypoints size greater than 0.001 for right"<< right_keypoints0[0].size()<<endl;
        // }
        g_count++;
        cout<< "g_count"<< g_count<<endl;

        //g_cout >1 for optical
        if(g_count >2){

            cout<<"hello"<<endl;
            //optical
            // inline_newL[0]=calOpticalFlow(img0[1],img0[0],left_keypoints0[1]);
            // inline_newR[0]=calOpticalFlow(img1[1],img1[0],right_keypoints0[1]);
            // cout<<"newest img keypoint size left"<< inline_newL[0].size()<<endl;
            // cout<<"newest img keypoint size right"<< inline_newR[0].size()<<endl;
            // if(inline_newL[0].size()<10 || inline_newR[0].size()<10){
            //     Ptr<ORB> orb_detector = ORB::create();
            //     orb_detector->detect(img0[0],left_keypoints0[0]);
            //     orb_detector->detect(img1[0],right_keypoints0[0]);
            //     inline_newL[0]=left_keypoints0[0];
            //     inline_newR[0]=right_keypoints0[0];
            // }
            
            //using matching instead of optical flow
            inline_newL[0]=matchingFunc04(left_descriptors0[1],left_descriptors0[0],left_keypoints0[1],left_keypoints0[0], 
            img0[1], img0[0]);
            inline_newR[0]=matchingFunc04(right_descriptors0[1],right_descriptors0[0],right_keypoints0[1],right_keypoints0[0], 
            img1[1], img1[0]);
            
            cout<< "size of inline_newL[0]"<<inline_newL[0].size() <<endl;
            
            //scoring method over last 10 imgs
            // left_keypoints_formatch = compareInlinePts(inline_newL,left_keypoints0[9]);
            // for(int i=0;i<left_keypoints_formatch.size();i++){
            //     cout<<"score of left keypoints:"<<left_keypoints_formatch[i].response<<endl;
            // }
            // right_keypoints_formatch = compareInlinePts(inline_newR,right_keypoints0[9]);
            // for(int i=0;i<right_keypoints_formatch.size();i++){
            //     cout<<"score of right keypoints:"<<right_keypoints_formatch[i].response<<endl;
            // }

            drawKeypoints(img0[1],inline_newL[0],img0[1],Scalar(0,0,255));
            drawKeypoints(img1[1],inline_newR[0],img1[1],Scalar(0,0,255));
            imshow("final image with best score left",img0[1]);
            imshow("final image with best score right",img1[1]);

            
            //left_keypoints_formatch_descriptors = getKeypointDescriptors(img0[9],left_keypoints_formatch);
            //right_keypoints_formatch_descriptors = getKeypointDescriptors(img1[9],right_keypoints_formatch);
            triangulated_points0=matchingFuncTriangulation(inline_newL[0],inline_newR[0],img0[1],img1[1]);
            cloud0 = PclDisplay(triangulated_points0);
            cout << "triangulated" << triangulated_points0<<endl;

            if(!triangulated_points1.empty() && cloud1->size()>0){

                getOdometryMatch(cloud0,cloud1);
                //compare triangulated 0 and 1
                g_count2++;

                cout<<" g_count2!"<<g_count2 <<endl;
                
                //std::swap(inline_newL[0],inline_newL[1]);
                //std::swap(inline_newR[0],inline_newR[1]);


            }
            std::swap(triangulated_points0,triangulated_points1);
            triangulated_points0.release();
            std::swap(cloud0,cloud1);
            
            //g_count =0;
            //calOpticalFlow();
            //matchingFunc();
            //DO THE MATCH FUNC
            waitKey(500);
            if(writing==2){
                waitKey();
            }

            //optical
            // left_keypoints0[0]= inline_newL[0];
            // right_keypoints0[0]= inline_newR[0];
            
        }
        
        cout<< "before swap keypoints 1:"<<left_keypoints0[1].size()<<endl;
        std::swap(img0[0],img0[1]);
        std::swap(img1[0],img1[1]);
        std::swap(left_keypoints0[0],left_keypoints0[1]);
        std::swap(right_keypoints0[0],right_keypoints0[1]);
        std::swap(left_descriptors0[0],left_descriptors0[1]);
        std::swap(right_descriptors0[0],right_descriptors0[1]);
        cout<< "end of loop keypoints 0:"<<left_keypoints0[0].size()<<endl;
        cout<< "end of loop keypoints 1:"<<left_keypoints0[1].size()<<endl;
        left_keypoints0[0].clear();
        right_keypoints0[0].clear();
        left_descriptors0[0].release();
        right_descriptors0[0].release();
        img0[0].release();
        img1[0].release();        
        // 0 would be new inputs, 1 would be old
        //img0.copyTo(previmg0);
        //std::swap(left_keypoints0[0],left_keypoints0[1]);
        //std::swap(left_descriptors0[0],left_descriptors0[1]);



        waitKey(500);
    }




}

void getOdometryMatch(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud0,const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1){
    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setInputSource(cloud0);
    icp.setInputTarget(cloud1);
    icp.setMaximumIterations(300);
    icp.setTransformationEpsilon(1e-9);
    icp.setMaxCorrespondenceDistance(150);
    icp.setEuclideanFitnessEpsilon(1);
    icp.setRANSACOutlierRejectionThreshold(1.5);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);    
    cout<<"hasconverged:" << icp.hasConverged() << endl;
    cout<<"score: " << icp.getFitnessScore() << endl;
    cout<< "transform between" <<g_count2 <<"and"<< g_count2+1 <<endl;
    cout<< icp.getFinalTransformation() << endl;        


}

pcl::PointCloud<pcl::PointXYZ>::Ptr PclDisplay(const Mat &triangulated){
    // pcl::visualization::PCLVisualizer viewer("Viewer");
    // viewer.setBackgroundColor (255,255,255);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = triangulated.rows;
    cloud->height =1;
    cloud->is_dense = false;
    cloud->points.resize(triangulated.rows);

    for(int i = 0; i <triangulated.rows; i++)
    {
        pcl::PointXYZ &point = cloud->points[i];
        point.x = triangulated.at<float>(i,0);
        point.y = -triangulated.at<float>(i,1);
        point.z = -triangulated.at<float>(i,2);
        //point.r = 0;
        //point.g = 0;
        //point.b = 255;
        cout<<"pointcloud: " <<cloud->points[i] <<endl;
    }
//     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 0, 255);
//     viewer.addPointCloud(cloud,single_color,"Triangulated Point Cloud");
//     viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"Triangulated Point Cloud");
//     viewer.addCoordinateSystem(300.0);
//     while (!viewer.wasStopped())
//    {
//         viewer.spinOnce(100);
//    }
    char save_cloud[100];
    sprintf(save_cloud,"./newpointsmove/test1_%02d.pcd",cloud_count);
    pcl::io::savePCDFileASCII(save_cloud,*cloud);
    cloud_count++;
    return cloud;
}

Mat matchingFuncTriangulation(vector<KeyPoint> left_keypoints,vector<KeyPoint> right_keypoints,Mat left_img,Mat right_img){


    Mat left_descriptors, right_descriptors;
    //match between 4 and 0, 4 is newest
    Ptr<ORB> orb_detector = ORB::create();
    

    Ptr<BFMatcher> orb_matcher = BFMatcher::create(NORM_HAMMING,true);
    orb_detector->compute(left_img,left_keypoints,left_descriptors);
    orb_detector->compute(right_img,right_keypoints,right_descriptors);
    for(int i = 0;i <left_keypoints.size();i++){
        cout<<"left keypoint: "<<left_keypoints[i].pt<<"with descriptors"<<left_descriptors.row(i)<<endl;
        
    }

    for(int i = 0;i <right_keypoints.size();i++){
       cout<<"right keypoint: "<<right_keypoints[i].pt<<"with descriptors"<<right_descriptors.row(i)<<endl; 
        
    }

    vector<DMatch> matches;
    
    vector<DMatch> inter_matches;
    //match along stereo lines
    for(int i =0;i<left_keypoints.size();i++){

        double pre_matches_distance;
        cv::DMatch bestscore_matches;
        double distance_score= 1000;
        
        for(int j=0;j<right_keypoints.size();j++){
            if(fabs(left_keypoints[i].pt.y-right_keypoints[j].pt.y)<=0.001){
                pre_matches_distance = cv::norm(left_descriptors.row(i),right_descriptors.row(j),cv::NORM_HAMMING);
                if(pre_matches_distance<distance_score){
                    distance_score = pre_matches_distance;
                    bestscore_matches=cv::DMatch(i,j,pre_matches_distance);
                }
            }
        }
        if(bestscore_matches.distance<100){
            matches.push_back(bestscore_matches);
        }
        
    }
    cout<< "matches size"<<matches.size()<<endl;
    //orb_matcher->DescriptorMatcher::match(left_descriptors,right_descriptors,matches);
    for(int i =0;i<matches.size();i++){
         cout<<"before sort"<< matches[i].distance<<"with keypoints"<< left_keypoints[matches[i].queryIdx].pt<<"&"<<right_keypoints[matches[i].trainIdx].pt<<endl;
    }    
    std::sort(matches.begin(),matches.end(),[](DMatch const &a, DMatch const &b){
        return a.distance <b.distance;
    });

    vector<KeyPoint> matched_left,matched_right;
    vector<DMatch> inline_matches;

    //take only the first 50 best matches distance, or distance below 30
    
    for(int i = 0;i<matches.size();i++){
        //if(matches[i].distance<=30){
        //fabs((keypoint0[good_matches[i].queryIdx].pt.y)-(keypoint1[good_matches[i].trainIdx].pt.y))
            
        if((left_keypoints[matches[i].queryIdx].pt.x)>(right_keypoints[matches[i].trainIdx].pt.x)){
            inter_matches.push_back(matches[i]);
        }
        //}
    }

    for(int i = 0;i<inter_matches.size();i++){
            matched_left.push_back(left_keypoints[inter_matches[i].queryIdx]);
            matched_right.push_back(right_keypoints[inter_matches[i].trainIdx]);
    }
    for(int i =0;i<matched_left.size();i++){
        cout<<"after sort"<< inter_matches[i].distance<<"with keypoints" <<matched_left[i].pt<<"&" <<matched_right[i].pt<<endl;
    }   
    
    // vector<vector<DMatch>> matches;
    // Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // orb_detector->compute(left_img,left_keypoints,left_descriptors);
    // orb_detector->compute(right_img,right_keypoints,right_descriptors);
    // orb_matcher->knnMatch(left_descriptors,right_descriptors,matches,2);
    // vector<KeyPoint> matched_left,matched_right;
    // vector<DMatch> inline_matches;
    // for(int i =0; i<matches.size();i++){
    //     if(matches[i][0].distance<nn_match_ratio*matches[i][1].distance){
    //         //keypoint in first frame stored in matched 0, passing distance thres
    //         matched_left.push_back(left_keypoints[matches[i][0].queryIdx]);
    //         //keypoint in current frame stored in matched 1, passing distance thres
    //         matched_right.push_back(right_keypoints[matches[i][0].trainIdx]);
    //         //inline_matches.push_back(matches[i][0]);
    //     }
    // }
    // cout<<"matched0 size: "<< matched_left.size() <<endl;

    //ransac 
    // Mat inline_mask, homography;
    // vector<KeyPoint> inline_left,inline_right;
    // vector<Point2f> converted_matched_old, converted_matched_new;
    // KeyPoint::convert(matched_left,converted_matched_old);
    // KeyPoint::convert(matched_right,converted_matched_new);
    // //must have more than 4 points
    // if(matched_left.size()>=4){
    //     homography = findHomography(converted_matched_old, converted_matched_new,RANSAC,ransac_thresh,inline_mask);
    // }
    // else
    // {
    //     throw "too little points"; 
    // // vector<KeyPoint> emptykeypoints;
    // // cout << "lost transformation" <<endl;
    
    // }  
    // for(int i =0; i <matched_left.size();i++){
    //     //see if mask gives a value of one, where one means it is correct matched
    //     if(inline_mask.at<uchar>(i)){
    //         int new_i = static_cast<int>(inline_left.size());
    //         inline_left.push_back(matched_left[i]);
    //         inline_right.push_back(matched_right[i]);
    //         //store matches of correct RANSAC
    //         //inline_matches.push_back(DMatch(new_i,new_i,0));
    //         inline_matches.push_back(DMatch(new_i,new_i,matches[i].distance));
    //     }
    // }
    // cout<< "inline_matches after RANSAC between left and right images: " << inline_matches.size() <<endl;
    // for(int i =0;i<inline_matches.size();i++){
    //     cout<<"inline_matches"<<inline_matches[i].distance <<"L"<<inline_left[i].pt<<"R"<<inline_right[i].pt<<endl;
    // } 


    Mat res;
    
    //old on the left
    //ransac is inline_left,inline_right,inline_matches
    drawMatches(left_img,left_keypoints,right_img,right_keypoints,inter_matches,res,Scalar(255,0,0),Scalar(255,0,0),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //cout<< "size of matched0:" << matched0.size()<<endl;
    //cout<< "size of matched1:" << matched1.size()<<endl;

    imshow(std::to_string(writing)+"matches",res);
    
    writing++;
    //triangulation
    std::vector<Point2f> goodmatches_left,goodmatches_right;
    //ransac inline_left,inline_right
    KeyPoint::convert(matched_left,goodmatches_left);
    KeyPoint::convert(matched_right,goodmatches_right);
    Mat pnts3D(1,goodmatches_left.size(),CV_64FC4);
    cv::triangulatePoints(P1,P2,goodmatches_left,goodmatches_right,pnts3D);
    pnts3D = pnts3D.t();
    vector<Point3f> worldpnts;
    Mat worldmat;
    convertPointsFromHomogeneous(pnts3D,worldmat);
    Mat newworldmat = worldmat.reshape(1);
    return newworldmat; 
    
}

Mat getKeypointDescriptors(Mat img,vector<KeyPoint> keypoints){

    Mat descriptors;
    Ptr<ORB> orb_detector = ORB::create();
    //orb_detector->setMaxFeatures(50);
    orb_detector->compute(img,keypoints,descriptors);
    return descriptors;
}
    

void setLeftFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors){
    //ORB detector
    
    vector<KeyPoint> pointy;
    Ptr<ORB> orb_detector = ORB::create();
    orb_detector->setMaxFeatures(200);
    orb_detector->detect(img,pointy,noArray());
    for(int i=0;i<pointy.size();i++){
        if(pointy[i].response>0.0015){
            keypoints.push_back(pointy[i]);
        }
    }
    for(int i =0;i<keypoints.size();i++){
        cout<<"Lkeypoints"<<keypoints[i].pt<<"and its score"<< keypoints[i].response<<endl;    
    }
    
    orb_detector->compute(img,keypoints,descriptors);
    //cout<< "left_keypoints0" <<left_keypoints0[g_count][0].pt <<endl;
    //cout<<"left_descriptors"<<left_descriptors0[g_count].row(0)<<endl;
    //cout<< "size"<< left_descriptors0[g_count].size()<<endl;
    drawKeypoints(img,keypoints,img, Scalar(255,0,0));
    imshow("keypointsdrawLeft",img);
    

}

void setRightFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors){
    //ORB detector
    
    vector<KeyPoint> pointy;
    Ptr<ORB> orb_detector = ORB::create();
    orb_detector->setMaxFeatures(200);
    orb_detector->detect(img,pointy,noArray());
    for(int i=0;i<pointy.size();i++){
        if(pointy[i].response>0.0015){
            keypoints.push_back(pointy[i]);
        }
    }
    for(int i =0;i<keypoints.size();i++){
        cout<<"Rkeypoints"<<keypoints[i].pt<<"and its score"<< keypoints[i].response<<endl;    
    }
    orb_detector->compute(img,keypoints,descriptors);
    //cout<< "left_keypoints0" <<left_keypoints0[g_count][0].pt <<endl;
    //cout<<"left_descriptors"<<left_descriptors0[g_count].row(0)<<endl;
    //cout<< "size"<< left_descriptors0[g_count].size()<<endl;
    drawKeypoints(img,keypoints,img, Scalar(255,0,0));
    imshow("keypointsdrawRight",img);
}

vector<KeyPoint> matchingFunc04(Mat old_descriptors,Mat newest_descriptors,vector<KeyPoint> old_keypoints,vector<KeyPoint> newest_keypoints, Mat old_img, Mat new_img){
    
    //match between 4 and 0, 4 is newest
    Ptr<ORB> orb_detector = ORB::create();
    vector<vector<DMatch>> matches;
    Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    orb_matcher->knnMatch(old_descriptors,newest_descriptors,matches,2);
    // for(int i =0; i <matches.size();i++){
    //     cout<<"distance matches"<< matches[i][0].distance<<endl;
    // }
    // for(int i =0; i <matches.size();i++){
    //     cout<<"distance matches"<< matches[i][1].distance<<endl;
    // }
    vector<KeyPoint> matched_old,matched_newest;
    vector<DMatch> inline_matches;
    for(int i =0; i<matches.size();i++){
        if(matches[i][0].distance<nn_match_ratio*matches[i][1].distance){
            //keypoint in first frame stored in matched 0, passing distance thres
            matched_old.push_back(old_keypoints[matches[i][0].queryIdx]);
            //keypoint in current frame stored in matched 1, passing distance thres
            matched_newest.push_back(newest_keypoints[matches[i][0].trainIdx]);
            
        }
    }
    
    cout<<"matched0 size: "<< matched_old.size() <<endl; 
    //ransac 
    Mat inline_mask, homography;
    vector<KeyPoint> inline_old,inline_new;
    vector<Point2f> converted_matched_old, converted_matched_new;
    KeyPoint::convert(matched_old,converted_matched_old);
    KeyPoint::convert(matched_newest,converted_matched_new);
    //must have more than 4 points
    if(matched_newest.size()>=4){
        homography = findHomography(converted_matched_old, converted_matched_new,RANSAC,ransac_thresh,inline_mask);
    }
    else
    {
        throw "too little points"; 
    // vector<KeyPoint> emptykeypoints;
    // cout << "lost transformation" <<endl;
    
    }  
    for(int i =0; i <matched_newest.size();i++){
        //see if mask gives a value of one, where one means it is correct matched
        if(inline_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inline_old.size());
            inline_old.push_back(matched_old[i]);
            inline_new.push_back(matched_newest[i]);
            //store matches of correct RANSAC
            inline_matches.push_back(DMatch(new_i,new_i,0));
        }
    }
    cout<< "inline_size after RANSAC between images: " << inline_matches.size() <<endl;


    // for(int i=0;i<inline_matches.size();i++)
    // {
    //     //keypoint in first frame stored in matched 0, passing distance thres
    //     matched0.push_back(left_keypoints0[0][matches[i][0].queryIdx].pt);
    //     //keypoint in current frame stored in matched 1, passing distance thres
    //     matched1.push_back(left_keypoints0[1][matches[i][0].trainIdx].pt);
    // }
        Mat res;
        //old on the left
        drawMatches(old_img,inline_old,new_img,inline_new,inline_matches,res,Scalar(255,0,0),Scalar(255,0,0),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //cout<< "size of matched0:" << matched0.size()<<endl;
        //cout<< "size of matched1:" << matched1.size()<<endl;
        imshow("1st matches",res);
        
        //inline_new is the correct matches in 4.

        //left_keypoints0[0]=inline_new;
        //left_keypoints0[1]=inline1;
        //orb_detector->compute(img0[g_count],inline_new,left_descriptors0[0]);
        //orb_detector->compute(img0[g_count],inline1,left_descriptors0[1]);
        return inline_new;
}

vector<KeyPoint> compareInlinePts(vector<vector<KeyPoint>> inline_new,vector<KeyPoint> final_keypoint){
    
    //vector<KeyPoint> combined_inline;
    vector<KeyPoint> scored_keypoints;
    vector<int> score;
    score.resize(final_keypoint.size());
    cout <<"inline_new size" << inline_new.size() <<endl;
    
    // for(int i=0; i<inline_new[0].size();i++){
    //     cout<< " inline_new[0]: " << inline_new[0][i].pt <<endl;
    // }
    // for(int i=0; i<final_keypoint.size();i++){
    //     cout<< " left_keypoints: " << final_keypoint[i].pt <<endl;
    // }

    // cout<< " inline_new[1] size: " << inline_new[1] <<endl;
    // cout<< " inline_new[2] size: " << inline_new[2] <<endl;
    // cout<< " inline_new[3] size: " << inline_new[3] <<endl;

    for(int k=0;k<9;k++){
        for(int j=0;j<final_keypoint.size();j++){
            
            for(int i=0;i<inline_new[k].size();i++){
                if((inline_new[k][i].pt.x ==final_keypoint[j].pt.x) &&(inline_new[k][i].pt.y ==final_keypoint[j].pt.y)){
                    score[j]= score[j]+1;
                    break;
                }
            }      
        }
    }

                           
    for(int i=0;i<score.size();i++){
        cout<<"score: "<< score[i] <<"\t";
    }
    // left_keypoints_formatch.push_back();
    //left_keypoints_formatch.push_back();
    //left_keypoints_formatch.resize(g_count2);
    for(int i=0;i<score.size();i++){
        if(score[i]>7)
        {
            scored_keypoints.push_back(final_keypoint[i]);
        }
    }



    //cout<<"left_keypoints:" << left_keypoints_formatch[g_count2][0].pt<<endl;
    cout<< "size of keypoints to match"<< scored_keypoints.size() <<endl;
    return scored_keypoints;
}

vector<KeyPoint> calOpticalFlow(Mat old_img,Mat new_img,vector<KeyPoint> old_keypoints){

    vector<Point2f> old_keypointsP2f,newest_keypointsP2f;
    vector<KeyPoint> newest_keypoints;
    vector<uchar> status;
    vector<float> err;
    Size winSize(31,31);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

    KeyPoint::convert(old_keypoints,old_keypointsP2f);
    //KeyPoint::convert(newest_keypoints,newest_keypointsP2f);
    
    //newest_keypointsP2f.resize(old_keypointsP2f.size());
    

    // use current keypoints and prev keypoints
    calcOpticalFlowPyrLK(old_img,new_img,old_keypointsP2f,newest_keypointsP2f,status, err, winSize,3,termcrit,0,0.001);
    
    // for(int i=0;i<status.size();i++){
    //     cout<<"status" << (int)status[i]<<"\t";
    // }
    
    for(int i =0; i <newest_keypointsP2f.size();i++){
        circle(new_img,newest_keypointsP2f[i],3,Scalar(0,255,0),1,8);
    }
    imshow("new_img after optical",new_img);
    cout<<"size of newest_keypointsP2f: "<<newest_keypointsP2f.size()<<endl;
    waitKey();

    KeyPoint::convert(newest_keypointsP2f,newest_keypoints);

    return newest_keypoints;
}