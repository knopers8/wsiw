
#include "polar_utils.hpp"


void linspace( float y0, float ymax, float steps, std::vector<float> & vec)
{
    float a = (ymax - y0)/(steps-1);

    vec.clear();
    for (int i = 0; i < steps; i++)
    {
        vec.push_back( a*i + y0 );
    }
}

void create_maps(cv::Mat & to_polar_map, cv::Mat & to_cart_map, int N_s, int N_r, float r_n, float blind, int x_0, int y_0, int src_width, int src_height )
{
// Function is responsible for creating cordinates map from cartesian space
// to polar space.
// img is input image with size n x m
// for other inputs check main, im to lazy
//  to_polar_map_r n x m array with r index in corespongind N_r x N_s polar
//  image. Inedx, not value!
//  to_polar_map_theta - as above but theta
// polar_img - ready polar image, displaying and debugging mostly
// show_polar_pixels - polar image displayed in cartesian space
    float thet_0 = 0; //beggining of theta ragne
    float thet_max = 2*PI-0.001; //end of theta ragne

    std::vector<float> theta;
    linspace( thet_0, thet_max, N_s+1, theta);

    std::vector<float> r;
    for( int i = 0; i <= N_r; i++ )
    {
        r.push_back( i*r_n + blind);
    }

    to_polar_map = cv::Mat( N_s*N_r, MAX_PIX_COUNT, CV_32S, double(0));
    to_cart_map = cv::Mat(src_height, src_width, CV_32S, double(0));

    for(int i = 0; i < N_r; i++)
    {
        for(int j = 0; j < N_s; j++)
        {
            float in_R = r[i];
            float out_R = r[i+1]; //check if doesnt try to access not its memory

            get_polar_pixel( (int32_t *)&to_polar_map.data[(N_s*i+j)*MAX_PIX_COUNT*4],
                             (int32_t *)to_cart_map.data,
                             x_0, y_0,
                             in_R, out_R,
                             theta[j], theta[j+1],
                             src_width, N_s, i, j );

        }
    }

}

void get_polar_pixel(int32_t * polar_coords, int32_t * cart_coords, int x_0, int y_0, float r_min, float r_max, float thet_min, float thet_max, int src_width, int N_s, int i, int j )
{
    int x_corners[4] = { r_min*cos(thet_min) + x_0, r_min*cos(thet_max) + x_0, r_max*cos(thet_min) + x_0, r_max*cos(thet_max) + x_0};
    int y_corners[4] = { r_min*sin(thet_min) + y_0, r_min*sin(thet_max) + y_0, r_max*sin(thet_min) + y_0, r_max*sin(thet_max) + y_0};

    int x_max = *std::max_element( x_corners, x_corners+4);
    int x_min = *std::min_element( x_corners, x_corners+4);
    int y_max = *std::max_element( y_corners, y_corners+4);
    int y_min = *std::min_element( y_corners, y_corners+4);


//    std::vector<int> x_span;
//    std::vector<int> y_span;
    int error_margin = 15;

//    for( int i = x_min - error_margin; i <= x_max + error_margin; i++)
//    {
//        x_span.push_back( i );
//    }
//    for( int i = y_min - error_margin; i <= y_max + error_margin; i++)
//    {
//        y_span.push_back( i );
//    }

    for(int x_el = x_min - error_margin; x_el <= x_max + error_margin; x_el++)// auto&& x_el : x_span)
    {
        for(int y_el = y_min - error_margin; y_el <= y_max + error_margin; y_el++  )// auto&& y_el : y_span)
        {
            int x = x_el - x_0;
            int y = y_el - y_0;

            float r = sqrt(x*x+y*y);
            float thet = fmod( atan2(y,x) + 2*PI, (2*PI));

            if ( r >= r_min && r <= r_max && thet >= thet_min && thet <= thet_max)
            {
                *polar_coords++ = (int32_t) x_el*src_width + y_el;
//                std::cout << "x_el*src_width + y_el " << x_el*src_width + y_el << std::endl;
                cart_coords[x_el*src_width + y_el] = (int32_t) i*N_s + j;

            }
        }
    }
}
