#ifndef POLAR_UTILS_HPP
#define POLAR_UTILS_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#define MAX_PIX_COUNT 128

#define PI 3.14159265359


void linspace( float y0, float ymax, float steps, std::vector<float> & vec);

void create_map(cv::Mat & to_polar_map, int N_s, int N_r, float r_n, float blind, int x_0, int y_0, int src_width);

void get_polar_pixel(int32_t * coords, int x_0, int y_0, float r_min, float r_max, float thet_min, float thet_max, int src_width );

void to_polar_c( uchar* input, int32_t* to_polar_map_x, int32_t* to_polar_map_y, uchar* output);







#endif // POLAR_UTILS_HPP
