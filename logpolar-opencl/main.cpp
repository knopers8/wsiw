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
#include "util.hpp"


//delete 'x' to activate
#define xC_MODEL
#define WEBCAM



int main(int argc, const char** argv)
{
    //-----------------------------------------------------
    // timing init
    //-----------------------------------------------------
    double loop_start_time = 0;
    double cl_start_time = 0;
    double loop_run_time = 0;
    double cl_run_time = 0;
    util::Timer timer;

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

#ifdef WEBCAM
    cv::VideoCapture cap(0);
    cv::Mat mat_src;

	if (!cap.isOpened())  // check if we succeeded
		return 10;

    if (!cap.read(mat_src))
		return 11;

    cv::cvtColor(mat_src, mat_src, CV_BGR2GRAY);
    cv::resize(mat_src, mat_src, {256, 256}, 0, 0, CV_INTER_NN);

#else

    cv::Mat mat_src = cv::imread("cameraman.tif", cv::IMREAD_GRAYSCALE);

    if(mat_src.empty())
    {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }
#endif // WEBCAM

    unsigned int channels = mat_src.channels();
    unsigned int depth    = mat_src.depth();

    //-----------------------------------------------------
    // polar transformation init
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

    //create params vector
    int step;
    for( step = 0; step < N_s; step += 64);
    std::vector<int> params = { MAX_PIX_COUNT, N_s, N_r, src_height, src_width, step};

    cv::Mat mat_dst = cv::Mat( N_r, N_s, mat_src.type(), double(0));
    cv::Mat result_display;

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
    // OpenCL init
    //-----------------------------------------------------

    cv::ocl::oclMat ocl_to_polar_map_x( to_polar_map_x );
    cv::ocl::oclMat ocl_to_polar_map_y( to_polar_map_y );
    cv::ocl::oclMat ocl_params( {params.size(), 1}, CV_32S, (void *) params.data() );

   //-----------------------------------------------------
    //load kernel source code and init program
    //-----------------------------------------------------

    std::ifstream in("to_polar.cl");
    std::string contents((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program("to_polar", contents.c_str());
    in.close();

    printf("mat step: %d total: %d sizeof(cl_mem): %d\n", mat_dst.step1(), mat_dst.total(), sizeof(cl_mem));
    std::size_t globalThreads[3] = { 1, N_r, 1}; //mat_dst.step1(), N_r, 1};
    //std::size_t localThreads[3] = {};
    std::vector<std::pair<size_t , const void *> > args(5);



    args[1] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_to_polar_map_x.data );
    args[2] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_to_polar_map_y.data );
    args[3] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_params.data );

    cv::ocl::oclMat ocl_src;
    cv::ocl::oclMat ocl_dst;

#ifdef WEBCAM
    while(cap.read(mat_src))
    {

//    cv::Rect crop(0, 0, 255, 255);
//    mat_src = mat_src(crop);

    cv::cvtColor(mat_src, mat_src, CV_BGR2GRAY);
    cv::resize(mat_src, mat_src, {256, 256}, 0, 0, CV_INTER_NN);
#endif //WEBCAM

    cl_start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

    ocl_src = mat_src;
    ocl_dst = mat_dst;

    args[0] = std::make_pair( sizeof(cl_mem), (void *) &ocl_src.data );
    args[4] = std::make_pair( sizeof(cl_mem), (void *) &ocl_dst.data );

    //-----------------------------------------------------
    //execute kernel
    //-----------------------------------------------------

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program, "to_polar", globalThreads, NULL, args, channels, depth, NULL);
    ocl_dst.download(mat_dst);

    cl_run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - cl_start_time;



#endif //C_MODEL


    //-----------------------------------------------------
    //show results
    //-----------------------------------------------------
    cv::resize(mat_dst, result_display, {N_r*4, N_s*4}, 0, 0, CV_INTER_NN);


    cv::namedWindow("Source");
    cv::namedWindow("Polar");
    cv::imshow("Source", mat_src);
    cv::imshow("Polar", result_display);

#ifdef WEBCAM

    loop_run_time = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - loop_start_time;
    printf("\bLoop: %.3f s, OpenCL: %.3f s - %.0f\% \n", loop_run_time, cl_run_time, 100*cl_run_time/loop_run_time);
    loop_start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    cv::waitKey(1); //ms
    }
#else

    printf("OpenCL runtime %.3f seconds\n",  cl_run_time);

#endif // WEBCAM

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}






