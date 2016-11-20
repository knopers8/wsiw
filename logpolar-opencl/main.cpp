#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#define PI 3.14159265359

void to_logpolar_c(
    uchar* input,
    float * thet_vals_cos,
    float * thet_vals_sin,
    float * p_vals,
//    __constant int center_h,
//    __constant int center_w,
    uchar* output);


//delete 'x' to activate
#define xC_MODEL

int main(int argc, const char** argv)
{
    //-----------------------------------------------------
    // device init
    //-----------------------------------------------------

    cv::ocl::DevicesInfo devInfo;
    int res = cv::ocl::getOpenCLDevices(devInfo);
    if(res == 0)
    {
        std::cerr << "There is no OPENCL Here !" << std::endl;
        return -1;
    }
    else
    {
        for(unsigned int i = 0 ; i < devInfo.size() ; ++i)
        {
            std::cout << "Device : " << devInfo[i]->deviceName << " is present" << std::endl;
        }
    }

    cv::ocl::setDevice(devInfo[1]);        // select device to use
    std::cout << CV_VERSION_EPOCH << "." << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << std::endl;


    //-----------------------------------------------------
    //image source init
    //-----------------------------------------------------
    cv::Mat mat_src = cv::imread("cameraman.tif", cv::IMREAD_GRAYSCALE);

    if(mat_src.empty())
    {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }
    unsigned int channels = mat_src.channels();
    unsigned int depth    = mat_src.depth();

    //-----------------------------------------------------
    //log polar transformation init
    //-----------------------------------------------------


    int src_height = mat_src.rows;
    int src_width = mat_src.cols;

    int center_h = src_height/2;
    int center_w = src_width/2;

    int N_fov = 50; // total number of parts to divide each ring, each 2pi/N_fov wide
    int N_circ = 39; //numbers of rings

    float b = (N_fov+PI)/(N_fov-PI); // base of logarithm
    float p_fov = N_fov/PI; // size of circle at (x,y)=(x_0,y_0), blind spot


    std::vector<float> p_vals;
    std::vector<float> sample_radius;
    for(int i = 0; i < N_circ; i++)
    {
        p_vals.push_back( pow(b, i) ); // R values  b^i  i: [0, N_circ-1]
        sample_radius.push_back( PI * p_vals.back() / N_fov ); // radius of single wheel on 'i' ring (single log-polar pixel)
    }

    // theta values
    std::vector<float> thet_vals;
    std::vector<float> thet_vals_cos;
    std::vector<float> thet_vals_sin;

    for(int i = 0; i < N_fov; i++)
    {
        thet_vals.push_back( i * ( (2*PI) / N_fov ));
        thet_vals_cos.push_back( cos( thet_vals.back() ) );
        thet_vals_sin.push_back( sin( thet_vals.back() ) );
    }


    //check accuracy with matlab model

    std::cout << "p_vals, sample_radius: \n";
    for( int i = 0; i < p_vals.size(); i++ )
        std::cout << std::setw(15) << p_vals[i] << std::setw(15) << sample_radius[i] << std::endl;
    std::cout << std::endl;

    std::cout << "thet_vals, cos, sin: \n";
    std::cout.width(2);
    for( int i = 0; i < thet_vals.size(); i++ )
        std::cout << std::setw(15) << thet_vals[i] << std::setw(15) <<  thet_vals_cos[i] << std::setw(15) << thet_vals_sin[i] << std::endl;

#ifdef C_MODEL
    cv::Mat mat_dst = cv::Mat( N_circ, N_fov, mat_src.type(), double(0));

    to_logpolar_c( mat_src.data, thet_vals_cos.data(), thet_vals_sin.data(), p_vals.data(), mat_dst.data);


#else
    //-----------------------------------------------------
    // OpenCL matrices init
    //-----------------------------------------------------

    cv::Mat mat_dst = cv::Mat( N_circ, N_fov, mat_src.type(), double(0));

    cv::ocl::oclMat ocl_src(mat_src);
//    cv::ocl::oclMat ocl_dst( mat_src.size(), mat_src.type());
    cv::ocl::oclMat ocl_dst( mat_dst);//N_circ, N_fov, mat_src.type());

    cv::ocl::oclMat ocl_thet_vals_cos( {thet_vals_cos.size(), 1}, CV_32F, (void *) thet_vals_cos.data() );
    cv::ocl::oclMat ocl_thet_vals_sin( {thet_vals_sin.size(), 1}, CV_32F, (void *) thet_vals_sin.data() );
    cv::ocl::oclMat ocl_p_vals( {p_vals.size(), 1}, CV_32F, (void *) p_vals.data() );

    //-----------------------------------------------------
    //load kernel source code and init program
    //-----------------------------------------------------

    std::ifstream in("to_logpolar.cl");
    std::string contents((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program("to_logpolar", contents.c_str());
    printf("mat step: %d total: %d\n", mat_dst.step1(), mat_dst.total());
    std::size_t globalThreads[3]={ mat_dst.step1(), N_circ, 1};
    std::vector<std::pair<size_t , const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_src.data ));

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_thet_vals_cos.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_thet_vals_sin.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_p_vals.data ));
//    args.push_back( std::make_pair( sizeof(cl_mem), (void *) center_h ));
//    args.push_back( std::make_pair( sizeof(cl_mem), (void *) center_w ));

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_dst.data ));

    //-----------------------------------------------------
    //execute kernel
    //-----------------------------------------------------

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program, "to_logpolar", globalThreads, NULL, args, channels, depth, NULL);
    ocl_dst.download(mat_dst);
#endif


    //-----------------------------------------------------
    //show results
    //-----------------------------------------------------
//    cv::resize(mat_dst, mat_dst, {N_fov*4, N_circ*4}, 0, 0, CV_INTER_NN);


    cv::namedWindow("mat_src");
    cv::namedWindow("mat_dst");
    cv::imshow("mat_src", mat_src);
    cv::imshow("mat_dst", mat_dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}



void to_logpolar_c(
    uchar* input,
    float * thet_vals_cos,
    float * thet_vals_sin,
    float * p_vals,
//    __constant int center_h,
//    __constant int center_w,
    uchar* output)
{
    int N = 50;
    int i; //kolumna
    int j; //wiersz

    int center_w = 128;
    int center_h = 128;

    std::cout << p_vals[0] << " " << p_vals[38] << std::endl;
    std::cout << thet_vals_sin[0] << " " << thet_vals_sin[49] << std::endl;

    int read_pos;

    for( i = 0; i < 50; i++){
        for( j = 0; j < 39; j++){
            read_pos = p_vals[j] * thet_vals_sin[i] + center_w;
            read_pos += 256 * ( p_vals[j] * thet_vals_cos[i] + center_h);
            output[50*j+i] = input[read_pos];
//            output[50*j+i] = input[256*j+i];

        }
    }

}
