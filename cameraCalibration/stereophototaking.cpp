#include <iostream>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap0(0); // open the default camera
    VideoCapture cap1(1);
    if(!cap0.isOpened())  // check if we succeeded
        return -1;
    if(!cap1.isOpened())
        return -1;

    //Size winSize(50,50);
    int i = 1;

    for(;;)
    {
        Mat frame0,frame1,image0,image1;
                
        int *pt_i = &i;
        cap0 >> frame0; // get a new frame from camera
        frame0.copyTo(image0);
        cap1 >> frame1;
        frame1.copyTo(image1);
        //cvtColor(image, gray, COLOR_BGR2GRAY); //convert to gray
        imshow("left camera", image0); //show the image or gray image
        imshow("right camera", image1);
        char c =(char)waitKey(30); //wait 30ms for next frame
        if( c == 's'){        //keypress s to take photos
            
            char filename0[100];
            char filename1[100];
            sprintf(filename0,"/home/joseph/VOexamples/cameraCalibration/data/left%02d.jpg",*pt_i);
            sprintf(filename1,"/home/joseph/VOexamples/cameraCalibration/data/right%02d.jpg",*pt_i);

            printf("write to file %d \n",*pt_i);
            imwrite(filename0, image0); //write gray or image to filename
            imwrite(filename1, image1);
            

            i++;
        }    
           
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
