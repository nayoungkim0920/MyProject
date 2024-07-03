#ifndef IMAGEPROCESSING_CUH
#define IMAGEPROCESSING_CUH

#include <opencv2/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif

	void callRotateImageCUDA(cv::Mat& image);

#ifdef __cplusplus
}
#endif

#endif // IMAGEPROCESSING_CUH
