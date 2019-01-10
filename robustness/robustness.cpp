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
#include <deque>

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
int g_count = 0;
int g_count2 = 1;


vector<Mat>  left_descriptors0(10),right_descriptors0(10);
vector<vector<KeyPoint>> left_keypoints0(10),right_keypoints0(10);
Mat frame; 
vector<Mat> img0(10),img1(10);
vector<vector<KeyPoint>> inline_new(10);
vector<KeyPoint> left_keypoints_formatch;


void setLeftFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors);
void matchingFunc04();
void matchingFunc14();
void matchingFunc24();
void matchingFunc34();
void compareInlinePts();
void calOpticalFlow();
vector<KeyPoint> matchingFunc04(Mat old_descriptors,Mat newest_descriptors,vector<KeyPoint> old_keypoints,vector<KeyPoint> newest_keypoints,Mat old_img,Mat newest_img);


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
        frame.copyTo(img0[g_count]);     
        imshow("imgL",img0[g_count]);   
        cap1 >>frame;
        frame.copyTo(img1[g_count]);

        cvtColor(img0[g_count],img0[g_count],COLOR_BGR2GRAY);
        cvtColor(img1[g_count],img1[g_count],COLOR_BGR2GRAY);
        //create rmap for remapping
        double scaling_factor;
        scaling_factor = 800./MAX(img0[g_count].size().width,img0[g_count].size().height);
        wid = cvRound(img0[g_count].size().width*scaling_factor);
        hei = cvRound(img0[g_count].size().height*scaling_factor);

        initUndistortRectifyMap(M1,D1,R1,P1,img0[g_count].size(),CV_32FC1,rmap[0][0],rmap[0][1]);
        initUndistortRectifyMap(M2,D2,R2,P2,img0[g_count].size(),CV_32FC1,rmap[1][0],rmap[1][1]);
        remap(img0[g_count],img0[g_count],rmap[0][0],rmap[0][1],INTER_LINEAR);
        remap(img1[g_count],img1[g_count],rmap[1][0],rmap[1][1],INTER_LINEAR);
        cv::resize(img0[g_count],img0[g_count],cv::Size(wid,hei),0,0,CV_INTER_LINEAR);
        cv::resize(img1[g_count],img1[g_count],cv::Size(wid,hei),0,0,CV_INTER_LINEAR);
        
        //keypoints and descriptors pass by reference
        setLeftFrame(img0[g_count],left_keypoints0[g_count],left_descriptors0[g_count]);
        setRightFrame(img1[g_count],right_keypoints0[g_count],right_descriptors0[g_count]);
        g_count++;

        if(!img0[9].empty()){

            cout<<"hello"<<endl;
            
            inline_new[0]=matchingFunc04(left_descriptors0[0],left_descriptors0[4],left_keypoints0[0],left_keypoints0[4], img0[0], img0[4]);
            inline_new[1]=matchingFunc04(left_descriptors0[1],left_descriptors0[4],left_keypoints0[1],left_keypoints0[4], img0[1], img0[4]);
            inline_new[2]=matchingFunc04(left_descriptors0[2],left_descriptors0[4],left_keypoints0[2],left_keypoints0[4], img0[2], img0[4]);
            inline_new[3]=matchingFunc04(left_descriptors0[3],left_descriptors0[4],left_keypoints0[3],left_keypoints0[4], img0[3], img0[4]);
            inline_new[4]=matchingFunc04(left_descriptors0[4],left_descriptors0[4],left_keypoints0[4],left_keypoints0[4], img0[4], img0[4]);
            inline_new[5]=matchingFunc04(left_descriptors0[5],left_descriptors0[4],left_keypoints0[5],left_keypoints0[4], img0[5], img0[4]);
            inline_new[6]=matchingFunc04(left_descriptors0[6],left_descriptors0[4],left_keypoints0[6],left_keypoints0[4], img0[6], img0[4]);
            inline_new[7]=matchingFunc04(left_descriptors0[7],left_descriptors0[4],left_keypoints0[7],left_keypoints0[4], img0[7], img0[4]);
            inline_new[8]=matchingFunc04(left_descriptors0[8],left_descriptors0[4],left_keypoints0[8],left_keypoints0[4], img0[8], img0[4]);
            inline_new[9]=matchingFunc04(left_descriptors0[9],left_descriptors0[4],left_keypoints0[9],left_keypoints0[4], img0[9], img0[4]);
            cout<<"test121241"<<endl;

            
            
            // matchingFunc14();
            // matchingFunc24();
            // matchingFunc34();
            compareInlinePts();

            
            //calOpticalFlow();
            //matchingFunc();
            //DO THE MATCH FUNC
            g_count2++;
            return 0;
        }

        
        // 0 would be new inputs, 1 would be old
        
        //img0.copyTo(previmg0);
        //std::swap(left_keypoints0[0],left_keypoints0[1]);
        //std::swap(left_descriptors0[0],left_descriptors0[1]);
        cout<<"once"<<endl;



        waitKey(1000);
    }




}

    

void setLeftFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors){
    //ORB detector
    
    cout<< "g_count: "<< g_count <<endl;
    Ptr<ORB> orb_detector = ORB::create();
    orb_detector->detectAndCompute(img,noArray(),keypoints,descriptors);
    //cout<< "left_keypoints0" <<left_keypoints0[g_count][0].pt <<endl;
    //cout<<"left_descriptors"<<left_descriptors0[g_count].row(0)<<endl;
    //cout<< "size"<< left_descriptors0[g_count].size()<<endl;
    drawKeypoints(img,keypoints,img, Scalar(255,0,0));
    imshow("keypointsdrawLeft",img);
    

}

void setRightFrame(Mat img,vector<KeyPoint> &keypoints,Mat &descriptors){
    //ORB detector
    
    cout<< "g_count: "<< g_count <<endl;
    Ptr<ORB> orb_detector = ORB::create();
    orb_detector->detectAndCompute(img,noArray(),keypoints,descriptors);
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
    vector<KeyPoint> emptykeypoints;
    cout << "lost transformation" <<endl;
    
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
    cout<< "inline_size after RANSAC between left images: " << inline_matches.size() <<endl;


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

void matchingFunc14(){
    //match between 4 and 0, 4 is newest
    Ptr<ORB> orb_detector = ORB::create();
    vector<vector<DMatch>> matches;
    Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    orb_matcher->knnMatch(left_descriptors0[1],left_descriptors0[4],matches,2);
    
    vector<KeyPoint> matched_old,matched_new;
    vector<DMatch> inline_matches;
    for(int i =0; i<matches.size();i++){
        if(matches[i][0].distance<0.8*matches[i][1].distance){
            //keypoint in first frame stored in matched 0, passing distance thres
            matched_old.push_back(left_keypoints0[1][matches[i][0].queryIdx]);
            //keypoint in current frame stored in matched 1, passing distance thres
            matched_new.push_back(left_keypoints0[4][matches[i][0].trainIdx]);
            
        }
    }
    cout<<"matched0 size: "<< matched_old.size() <<endl; 
    //ransac 
    Mat inline_mask, homography;
    vector<KeyPoint> inline_old;
    vector<Point2f> converted_matched_old, converted_matched_new;
    KeyPoint::convert(matched_old,converted_matched_old);
    KeyPoint::convert(matched_new,converted_matched_new);
    //must have more than 4 points
    if(matched_new.size()>=4){
        homography = findHomography(converted_matched_old, converted_matched_new,RANSAC,ransac_thresh,inline_mask);
    }
    else
    {
    vector<KeyPoint> emptykeypoints;
    cout << "lost transformation" <<endl;
    
    }  
    for(int i =0; i <matched_new.size();i++){
        //see if mask gives a value of one, where one means it is correct matched
        if(inline_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inline_old.size());
            inline_old.push_back(matched_old[i]);
            inline_new[1].push_back(matched_new[i]);
            //store matches of correct RANSAC
            inline_matches.push_back(DMatch(new_i,new_i,0));
        }
    }
    cout<< "inline_size after RANSAC between left images: " << inline_matches.size() <<endl;


    // for(int i=0;i<inline_matches.size();i++)
    // {
    //     //keypoint in first frame stored in matched 0, passing distance thres
    //     matched0.push_back(left_keypoints0[0][matches[i][0].queryIdx].pt);
    //     //keypoint in current frame stored in matched 1, passing distance thres
    //     matched1.push_back(left_keypoints0[1][matches[i][0].trainIdx].pt);
    // }
        Mat res;
        //old on the left
        drawMatches(img0[0],inline_old,img0[4],inline_new[1],inline_matches,res,Scalar(255,0,0),Scalar(255,0,0),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //cout<< "size of matched0:" << matched0.size()<<endl;
        //cout<< "size of matched1:" << matched1.size()<<endl;
        imshow("1st matches",res);

        //inline_new is the correct matches in 4.

        //left_keypoints0[0]=inline_new;
        //left_keypoints0[1]=inline1;
        //orb_detector->compute(img0[g_count],inline_new,left_descriptors0[0]);
        //orb_detector->compute(img0[g_count],inline1,left_descriptors0[1]);
   
}
void matchingFunc24(){
    //match between 4 and 0, 4 is newest
    Ptr<ORB> orb_detector = ORB::create();
    vector<vector<DMatch>> matches;
    Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    orb_matcher->knnMatch(left_descriptors0[2],left_descriptors0[4],matches,2);
    
    vector<KeyPoint> matched_old,matched_new;
    vector<DMatch> inline_matches;
    for(int i =0; i<matches.size();i++){
        if(matches[i][0].distance<0.8*matches[i][1].distance){
            //keypoint in first frame stored in matched 0, passing distance thres
            matched_old.push_back(left_keypoints0[2][matches[i][0].queryIdx]);
            //keypoint in current frame stored in matched 1, passing distance thres
            matched_new.push_back(left_keypoints0[4][matches[i][0].trainIdx]);
            
        }
    }
    cout<<"matched0 size: "<< matched_old.size() <<endl; 
    //ransac 
    Mat inline_mask, homography;
    vector<KeyPoint> inline_old;
    vector<Point2f> converted_matched_old, converted_matched_new;
    KeyPoint::convert(matched_old,converted_matched_old);
    KeyPoint::convert(matched_new,converted_matched_new);
    //must have more than 4 points
    if(matched_new.size()>=4){
        homography = findHomography(converted_matched_old, converted_matched_new,RANSAC,ransac_thresh,inline_mask);
    }
    else
    {
    vector<KeyPoint> emptykeypoints;
    cout << "lost transformation" <<endl;
    
    }  
    for(int i =0; i <matched_new.size();i++){
        //see if mask gives a value of one, where one means it is correct matched
        if(inline_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inline_old.size());
            inline_old.push_back(matched_old[i]);
            inline_new[2].push_back(matched_new[i]);
            //store matches of correct RANSAC
            inline_matches.push_back(DMatch(new_i,new_i,0));
        }
    }
    cout<< "inline_size after RANSAC between left images: " << inline_matches.size() <<endl;


    // for(int i=0;i<inline_matches.size();i++)
    // {
    //     //keypoint in first frame stored in matched 0, passing distance thres
    //     matched0.push_back(left_keypoints0[0][matches[i][0].queryIdx].pt);
    //     //keypoint in current frame stored in matched 1, passing distance thres
    //     matched1.push_back(left_keypoints0[1][matches[i][0].trainIdx].pt);
    // }
        Mat res;
        //old on the left
        drawMatches(img0[0],inline_old,img0[4],inline_new[2],inline_matches,res,Scalar(255,0,0),Scalar(255,0,0),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //cout<< "size of matched0:" << matched0.size()<<endl;
        //cout<< "size of matched1:" << matched1.size()<<endl;
        imshow("1st matches",res);

        //inline_new is the correct matches in 4.

        //left_keypoints0[0]=inline_new;
        //left_keypoints0[1]=inline1;
        //orb_detector->compute(img0[g_count],inline_new,left_descriptors0[0]);
        //orb_detector->compute(img0[g_count],inline1,left_descriptors0[1]);
   
}

void matchingFunc34(){
    //match between 4 and 0, 4 is newest
    Ptr<ORB> orb_detector = ORB::create();
    vector<vector<DMatch>> matches;
    Ptr<DescriptorMatcher> orb_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    orb_matcher->knnMatch(left_descriptors0[3],left_descriptors0[4],matches,2);
    
    vector<KeyPoint> matched_old,matched_new;
    vector<DMatch> inline_matches;
    for(int i =0; i<matches.size();i++){
        if(matches[i][0].distance<0.8*matches[i][1].distance){
            //keypoint in first frame stored in matched 0, passing distance thres
            matched_old.push_back(left_keypoints0[3][matches[i][0].queryIdx]);
            //keypoint in current frame stored in matched 1, passing distance thres
            matched_new.push_back(left_keypoints0[4][matches[i][0].trainIdx]);
            
        }
    }
    cout<<"matched0 size: "<< matched_old.size() <<endl; 
    //ransac 
    Mat inline_mask, homography;
    vector<KeyPoint> inline_old;
    vector<Point2f> converted_matched_old, converted_matched_new;
    KeyPoint::convert(matched_old,converted_matched_old);
    KeyPoint::convert(matched_new,converted_matched_new);
    //must have more than 4 points
    if(matched_new.size()>=4){
        homography = findHomography(converted_matched_old, converted_matched_new,RANSAC,ransac_thresh,inline_mask);
    }
    else
    {
    vector<KeyPoint> emptykeypoints;
    cout << "lost transformation" <<endl;
    
    }  
    for(int i =0; i <matched_new.size();i++){
        //see if mask gives a value of one, where one means it is correct matched
        if(inline_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inline_old.size());
            inline_old.push_back(matched_old[i]);
            inline_new[3].push_back(matched_new[i]);
            //store matches of correct RANSAC
            inline_matches.push_back(DMatch(new_i,new_i,0));
        }
    }
    cout<< "inline_size after RANSAC between left images: " << inline_matches.size() <<endl;


    // for(int i=0;i<inline_matches.size();i++)
    // {
    //     //keypoint in first frame stored in matched 0, passing distance thres
    //     matched0.push_back(left_keypoints0[0][matches[i][0].queryIdx].pt);
    //     //keypoint in current frame stored in matched 1, passing distance thres
    //     matched1.push_back(left_keypoints0[1][matches[i][0].trainIdx].pt);
    // }
        Mat res;
        //old on the left
        drawMatches(img0[0],inline_old,img0[4],inline_new[3],inline_matches,res,Scalar(255,0,0),Scalar(255,0,0),std::vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //cout<< "size of matched0:" << matched0.size()<<endl;
        //cout<< "size of matched1:" << matched1.size()<<endl;
        imshow("1st matches",res);

        //inline_new is the correct matches in 4.

        //left_keypoints0[0]=inline_new;
        //left_keypoints0[1]=inline1;
        //orb_detector->compute(img0[g_count],inline_new,left_descriptors0[0]);
        //orb_detector->compute(img0[g_count],inline1,left_descriptors0[1]);
   
}

void compareInlinePts(){
    
    //vector<KeyPoint> combined_inline;
    vector<int> score(500,0);
    
    for(int i=0; i<inline_new[0].size();i++){
        cout<< " inline_new[0]: " << inline_new[0][i].pt <<endl;
    }
    for(int i=0; i<left_keypoints0[4].size();i++){
        cout<< " left_keypoints: " << left_keypoints0[4][i].pt <<endl;
    }

    // cout<< " inline_new[1] size: " << inline_new[1] <<endl;
    // cout<< " inline_new[2] size: " << inline_new[2] <<endl;
    // cout<< " inline_new[3] size: " << inline_new[3] <<endl;
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[0].size();i++){
            if((inline_new[0][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[0][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;
            }
        }      
    }
    
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[1].size();i++){
            if((inline_new[1][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[1][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[2].size();i++){
            if((inline_new[2][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[2][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;  
                break;
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[3].size();i++){
            if((inline_new[3][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[3][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[4].size();i++){
            if((inline_new[4][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[4][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[5].size();i++){
            if((inline_new[5][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[5][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[6].size();i++){
            if((inline_new[6][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[6][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[7].size();i++){
            if((inline_new[7][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[7][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[8].size();i++){
            if((inline_new[8][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[8][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }
    for(int j=0;j<left_keypoints0[4].size();j++){
        
        for(int i=0;i<inline_new[9].size();i++){
            if((inline_new[9][i].pt.x ==left_keypoints0[4][j].pt.x) &&(inline_new[9][i].pt.y ==left_keypoints0[4][j].pt.y)){
                score[j]= score[j]+1;
                break;    
            }
        }      
    }                            
    // for(int i=0;i<score.size();i++){
    //     cout<<"score: "<< score[i] <<"\t";
    // }
    // left_keypoints_formatch.push_back();
    //left_keypoints_formatch.push_back();
    //left_keypoints_formatch.resize(g_count2);
    for(int i=0;i<score.size();i++){
        if(score[i]>9)
        {
            left_keypoints_formatch.push_back(left_keypoints0[4][i]);
        }
    }

    //cout<<"left_keypoints:" << left_keypoints_formatch[g_count2][0].pt<<endl;
    cout<< "size of keypoints to match"<< left_keypoints_formatch.size() <<endl;
}

void calOpticalFlow(){

    vector<vector<Point2f>> left_keypoints0P2f(2);
    vector<uchar> status;
    vector<float> err;
    Size winSize(31,31);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

    KeyPoint::convert(left_keypoints0[0],left_keypoints0P2f[0]);
    KeyPoint::convert(left_keypoints0[1],left_keypoints0P2f[1]);
    
    // use current keypoints and prev keypoints
    calcOpticalFlowPyrLK(img0[g_count-1],img0[g_count],left_keypoints0P2f[1],left_keypoints0P2f[0],status, err, winSize,3,termcrit,0,0.001);
    
    for(int i=0;i<status.size();i++){
        cout<<"status" << (int)status[i]<<"\t";
    }
    cout<< endl;
    for(int i =0; i <left_keypoints0P2f[0].size();i++){
        circle(img0[g_count],left_keypoints0P2f[0][i],3,Scalar(0,255,0),-1,8);
    }
    imshow("img0[g_count] after optical",img0[g_count]);

}