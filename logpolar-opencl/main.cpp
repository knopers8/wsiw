#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <iomanip>


#define PI 3.14159265359
#define MAX_PIX_COUNT 128

#include "polar_utils.hpp"


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
    int blind = 10; // radius of blind spot, can be 0
    int N_r = 40;   //number of rings
    int r_max = 100; // outer raius of last ring
    float r_n = (r_max-blind)/(float)N_r;   // radius of n-th ring = n*r_n n=0:N_r-1;
    int N_s = 40; // number of slices, just like pizza. Find better name and let me know. Number of part to divide every ring.

    int src_height = mat_src.rows;
    int src_width = mat_src.cols;

    int center_h = src_height/2;
    int center_w = src_width/2;

    cv::Mat to_polar_map_x;
    cv::Mat to_polar_map_y;


    create_map(to_polar_map_x, to_polar_map_y, N_s, N_r, r_n, blind, center_h, center_w);

//    int a = MAX_PIX_COUNT*2*(40*39+39);
//
//    for( int i=a; i< a+280; i += 2)
//    {
//        std::cout << (((int16_t)to_polar_map_x.data[i+1] << 8)) + (int16_t)to_polar_map_x.data[i] << " ";
//        std::cout << (((int16_t)to_polar_map_y.data[i+1] << 8)) + (int16_t)to_polar_map_y.data[i]  << std::endl;
//    }



#ifdef C_MODEL
    cv::Mat mat_dst = cv::Mat( N_r, N_s, mat_src.type(), double(0));

    to_polar_c( mat_src.data, (int32_t *)to_polar_map_x.data, (int32_t *)to_polar_map_y.data, mat_dst.data);


#else
    //-----------------------------------------------------
    // OpenCL matrices init
    //-----------------------------------------------------

    cv::Mat mat_dst = cv::Mat( N_r, N_s, mat_src.type(), double(0));

    cv::ocl::oclMat ocl_src(mat_src);
//    cv::ocl::oclMat ocl_dst( mat_src.size(), mat_src.type());
    cv::ocl::oclMat ocl_dst( mat_dst);//N_circ, N_fov, mat_src.type());

    cv::ocl::oclMat ocl_to_polar_map_x( to_polar_map_x );
    cv::ocl::oclMat ocl_to_polar_map_y( to_polar_map_y );

//    std::vector<int> vec_x = std::vector<int>( N_r*N_s*MAX_PIX_COUNT, 1);
//    std::vector<int> vec_y = std::vector<int>( N_r*N_s*MAX_PIX_COUNT, 1);
//
//    cv::ocl::oclMat ocl_to_polar_map_x( {vec_x.size(), 1}, CV_32S, (void *) vec_x.data() );
//    cv::ocl::oclMat ocl_to_polar_map_y( {vec_y.size(), 1}, CV_32S, (void *) vec_y.data() );



    //-----------------------------------------------------
    //load kernel source code and init program
    //-----------------------------------------------------

    std::ifstream in("to_polar.cl");
    std::string contents((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program("to_polar", contents.c_str());
    in.close();

    printf("mat step: %d total: %d sizeof(cl_mem): %d\n", mat_dst.step1(), mat_dst.total(), sizeof(cl_mem));
    std::size_t globalThreads[3]={ mat_dst.step1(), N_r, 1};
    std::vector<std::pair<size_t , const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_src.data ));


    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_to_polar_map_x.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_to_polar_map_y.data ));

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_dst.data ));

    //-----------------------------------------------------
    //execute kernel
    //-----------------------------------------------------

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program, "to_polar", globalThreads, NULL, args, channels, depth, NULL);
    ocl_dst.download(mat_dst);
#endif


    //-----------------------------------------------------
    //show results
    //-----------------------------------------------------
    cv::resize(mat_dst, mat_dst, {N_r*4, N_s*4}, 0, 0, CV_INTER_NN);


    cv::namedWindow("mat_src");
    cv::namedWindow("mat_dst");
    cv::imshow("mat_src", mat_src);
    cv::imshow("mat_dst", mat_dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}






