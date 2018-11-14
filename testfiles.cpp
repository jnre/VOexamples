#include <iostream>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    //Size winSize(50,50);
    string model = "test files";
    namedWindow(model, 0);
    int i = 0;

    for(;;)
    {
        Mat frame, gray, image;
                
        int *pt_i = &i;
        cap >> frame; // get a new frame from camera
        frame.copyTo(image);
        //cvtColor(image, gray, COLOR_BGR2GRAY); //convert to gray
        imshow(model, image); //show the image or gray image

        char c =(char)waitKey(30); //wait 30ms for next frame
        if( c == 's'){        //keypress s to take photos
            
            char filename[100];
            sprintf(filename,"/home/joseph/VOexamples/testfilesdataset/2%d.png",*pt_i);
            printf("write to file %d \n",*pt_i);
            imwrite(filename, image); //write gray or image to filename
            i++;
        }    
           
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
