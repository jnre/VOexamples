#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template <typename T1, typename T2>
    static void drawEpipolarLines(const std::string& , const cv::Matx<T1,3,3> ,
                const cv::Mat& , const cv::Mat& ,
                const std::vector<cv::Point_<T2> > ,
                const std::vector<cv::Point_<T2> > ,
                const float = -1);

template <typename T>
static float distancePointLine(const cv::Point_<T> , const cv::Vec<T,3>& );
