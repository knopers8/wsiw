#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <iomanip>


#define PI (3.14159265359)

#include "polar_utils.hpp"
#include "util.hpp"

//delete 'x' to activate
#define WEBCAM
#define MOVIEx
#define GRAYSCALEx
#define WRITE_PERFORMANCE_TO_FILEx
#define SHOW_OUTPUTS

#define WEBCAM_IMG_SIZE (512)
#define POLAR_SIZE (256)
#define MAX_RADIUS (250)

#define MOVIE_NAME "test512.mp4"
#define IMAGE_NAME "cameraman.tif"

int main(int argc, const char** argv)
{
    //-----------------------------------------------------
    // timing init
    //-----------------------------------------------------
    double loop_start_time = 0;
    double cl_start_time = 0;
    double loop_run_time = 0;
    double cl_run_time = 0;
    double cl_to_polar_time = 0;
    double cl_processing_time = 0;
    double cl_to_cart_time = 0;
    util::Timer timer;


    //-----------------------------------------------------
    // performance results file init
    //-----------------------------------------------------
#ifdef WRITE_PERFORMANCE_TO_FILE

    std::ofstream performance_file;
    performance_file.open("performance.txt");

#endif //WRITE_PERFORMANCE_TO_FILE
    //-----------------------------------------------------
    // device init
    //-----------------------------------------------------

    cv::ocl::DevicesInfo devInfo;
    int res = cv::ocl::getOpenCLDevices(devInfo, cv::ocl::CVCL_DEVICE_TYPE_ALL);
    if(res == 0)
    {
        std::cerr << "There is no OPENCL Here !" << std::endl;
        return -1;
    }
    else
    {
        for(unsigned int i = 0 ; i < devInfo.size() ; ++i)
        {
            std::cout << "Device " << i << ": " << devInfo[i]->deviceName << " is present" << std::endl;
        }
    }

    std::cout << "Choose device number." << std::endl;

    int devNumber;
    std::cin >> devNumber;

    if(devNumber < 0 || devNumber >= devInfo.size())
    {
        std::cout << "Incorrect device number, closing..." << std::endl;
        return -1;
    }

    cv::ocl::setDevice(devInfo[devNumber]);        // select device to use
    std::cout << "Open CV version: " << CV_VERSION_EPOCH << "." << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << std::endl;


    //-----------------------------------------------------
    //image source init
    //-----------------------------------------------------

#if defined(WEBCAM) || defined(MOVIE)
#ifdef WEBCAM
    cv::VideoCapture cap(0);// + CV_CAP_DSHOW);
#else //MOVIE
    cv::VideoCapture cap(MOVIE_NAME);
#endif // WEBCAM

    cv::Mat mat_src;

	if (!cap.isOpened())  // check if we succeeded
		return 10;

    if (!cap.read(mat_src))
		return 11;
#ifdef GRAYSCALE
    cv::cvtColor(mat_src, mat_src, CV_BGR2GRAY);
#endif
#ifdef WEBCAM
    cv::resize(mat_src, mat_src, {WEBCAM_IMG_SIZE, WEBCAM_IMG_SIZE}, 0, 0, CV_INTER_NN);
#else //MOVIE
//    cv::resize(mat_src, mat_src, {WEBCAM_IMG_SIZE, WEBCAM_IMG_SIZE}, 0, 0, CV_INTER_NN);
//    cv::Rect crop((mat_src.cols - mat_src.rows)/2, 0, mat_src.rows, mat_src.rows);
//    cv::Mat(mat_src(crop)).copyTo(mat_src);
#endif
    std::cout << "Source size: " << mat_src.cols << " " << mat_src.rows << " step: " << mat_src.step << std::endl;
#else

    cv::Mat mat_src = cv::imread(IMAGE_NAME, cv::IMREAD_GRAYSCALE);
    cv::resize(mat_src, mat_src, {WEBCAM_IMG_SIZE, WEBCAM_IMG_SIZE}, 0, 0, CV_INTER_NN);
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
    int N_r = POLAR_SIZE;   //number of rings
    int r_max = MAX_RADIUS; // outer raius of last ring
    float r_n = (r_max-blind)/(float)N_r;   // radius of n-th ring = n*r_n n=0:N_r-1;
    int N_s = POLAR_SIZE; // number of slices, just like pizza. Find better name and let me know. Number of part to divide every ring.

    int src_height = mat_src.rows;
    int src_width = mat_src.cols;

    int center_h = src_height/2;
    int center_w = src_width/2;

    cv::Mat to_polar_map;
    cv::Mat to_cart_map;
    std::cout << "Creating maps" << std::endl;
    create_maps(to_polar_map, to_cart_map, N_s, N_r, r_n, blind, center_h, center_w, src_width, src_height);
    std::cout << "Maps created" << std::endl;

//    int32_t * cart_coords = (int32_t *)to_polar_map.data;// + src_width*128;
//    std::ofstream myfile;
//    myfile.open ("to_polar_map.txt");
//
//    for(int i = 0; i < src_width*src_height; i++)
//    {
//        for( int j = 0 ; j < MAX_PIX_COUNT; j++)
//        {
//            myfile << std::setw(10) << *(cart_coords++) << " ";
//        }
//        myfile << std::endl;
//    }
//    myfile.close();
//    return 0;

    //create params vector
    int polar_step;
    for( polar_step = 0; polar_step < N_s; polar_step += 32);
    int cart_step;
    for( cart_step = 0; cart_step < src_width; cart_step += 32);
    std::vector<int> params = { MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step};

    cv::Mat mat_polar = cv::Mat( N_r, N_s, mat_src.type(), double(0));
    cv::Mat mat_processed = cv::Mat( N_r, N_s, mat_src.type(), double(0));
    cv::Mat mat_cart = cv::Mat( src_height, src_width, mat_src.type(), double(0));
    cv::Mat polar_result_display;

    //-----------------------------------------------------
    // OpenCL init
    //-----------------------------------------------------

    cv::ocl::oclMat ocl_to_polar_map( to_polar_map );
    cv::ocl::oclMat ocl_params( {params.size(), 1}, CV_32S, (void *) params.data() );

    cv::ocl::oclMat ocl_to_cart_map( to_cart_map );

   //-----------------------------------------------------
    //load kernel source code and init program
    //-----------------------------------------------------

    std::ifstream in("to_polar.cl");
    std::string contents((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program("to_polar", contents.c_str());
    cv::ocl::ProgramSource program_processing("processing", contents.c_str());
    cv::ocl::ProgramSource program_to_cart("to_cart", contents.c_str());
    in.close();

    //sprintf("mat step: %d total: %d sizeof(cl_mem): %d\n", mat_dst.step1(), mat_dst.total(), sizeof(cl_mem));
    std::size_t polarGlobalThreads[3] = { N_s, N_r, 1}; //mat_dst.step1(), N_r, 1};
    std::size_t cartGlobalThreads[3] = { src_height, src_width, 1};
    //std::size_t localThreads[3] = {};
    std::vector<std::pair<size_t , const void *> > to_polar_args(4);
    std::vector<std::pair<size_t , const void *> > processing_args(3);
    std::vector<std::pair<size_t , const void *> > to_cart_args(4);



    to_polar_args[1] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_to_polar_map.data );
    to_polar_args[2] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_params.data );

    processing_args[1] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_params.data );

    to_cart_args[1] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_to_cart_map.data );
    to_cart_args[2] =  std::make_pair( sizeof(cl_mem), (void *) &ocl_params.data );

    cv::ocl::oclMat ocl_src;
    cv::ocl::oclMat ocl_polar;
    cv::ocl::oclMat ocl_processed;
    cv::ocl::oclMat ocl_cart;

    std::cout << "Start\n";

#if defined(WEBCAM) || defined(MOVIE)
    while(cap.read(mat_src))
    {

//    cv::Rect crop(0, 0, 255, 255);
//    mat_src = mat_src(crop);
#ifdef GRAYSCALE
    cv::cvtColor(mat_src, mat_src, CV_BGR2GRAY);
#endif

#ifdef WEBCAM
    cv::resize(mat_src, mat_src, {WEBCAM_IMG_SIZE, WEBCAM_IMG_SIZE}, 0, 0, CV_INTER_NN);
#else
//    cv::resize(mat_src, mat_src, {WEBCAM_IMG_SIZE, WEBCAM_IMG_SIZE}, 0, 0, CV_INTER_NN);
//    cv::Rect crop(0, 0, 1024, 1024);
//    cv::Mat(mat_src(crop)).copyTo(mat_src);
#endif

#endif //WEBCAM or MOVIE

    cl_start_time = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;


    ocl_src = mat_src;
    ocl_polar = mat_polar;

    to_polar_args[0] = std::make_pair( sizeof(cl_mem), (void *) &ocl_src.data );
    to_polar_args[3] = std::make_pair( sizeof(cl_mem), (void *) &ocl_polar.data );

    //-----------------------------------------------------
    //execute kernel
    //-----------------------------------------------------
    //to polar
    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program, "to_polar", polarGlobalThreads, NULL, to_polar_args, channels, depth, NULL);
    ocl_polar.download(mat_polar);

    cl_to_polar_time = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0 - cl_start_time;


    //image processing
    ocl_processed = mat_processed;
    processing_args[0] = std::make_pair( sizeof(cl_mem), (void *) &ocl_polar.data );
    processing_args[2] = std::make_pair( sizeof(cl_mem), (void *) &ocl_processed.data );
    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program_processing, "processing", polarGlobalThreads, NULL, processing_args, channels, depth, NULL);
    ocl_processed.download(mat_processed);

    cl_processing_time = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0 - cl_start_time - cl_to_polar_time;

    //to cart
    ocl_cart = mat_cart;
    to_cart_args[0] = std::make_pair( sizeof(cl_mem), (void *) &ocl_processed.data );
    to_cart_args[3] = std::make_pair( sizeof(cl_mem), (void *) &ocl_cart.data );
    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program_to_cart, "to_cart", cartGlobalThreads, NULL, to_cart_args, channels, depth, NULL);
    ocl_cart.download(mat_cart);

    cl_run_time  = (static_cast<double>(timer.getTimeMicroseconds()) / 1000.0) - cl_start_time;
    cl_to_cart_time = cl_run_time - cl_processing_time - cl_to_polar_time;

    //-----------------------------------------------------
    //show results
    //-----------------------------------------------------
    cv::resize(mat_polar, polar_result_display, {N_s*4, N_r*4}, 0, 0, CV_INTER_NN);
//    polar_result_display = mat_dst;
#ifdef SHOW_OUTPUTS
    cv::namedWindow("Source");
    cv::namedWindow("Polar");
    cv::namedWindow("Processing");
    cv::namedWindow("Cart");
    cv::imshow("Source", mat_src);
    cv::imshow("Processing", mat_processed);
    cv::imshow("Polar", polar_result_display);
    cv::imshow("Cart", mat_cart);
#endif

#if defined(WEBCAM) || defined(MOVIE)

    loop_run_time = (static_cast<double>(timer.getTimeMicroseconds()) / 1000.0) - loop_start_time;
#ifdef SHOW_OUTPUTS
    printf("\bLoop: %.1fms, OpenCL: %.2fms - %.0f\%, to polar: %.2fms, imp proc: %.2fms to cart: %.2fms \n",
            loop_run_time, cl_run_time, 100*cl_run_time/loop_run_time, cl_to_polar_time, cl_processing_time, cl_to_cart_time);
#endif
    loop_start_time = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;

#ifdef WRITE_PERFORMANCE_TO_FILE

    performance_file << loop_run_time << " " << cl_run_time << " " << cl_to_polar_time << " " << cl_processing_time << " " << cl_to_cart_time << std::endl;

#endif // WRITE_PERFORMANCE_TO_FILE

    cv::waitKey(1); //ms
    }
#else

    printf("OpenCL runtime %.3f ms\n",  cl_run_time);

#endif // WEBCAM

#ifdef WRITE_PERFORMANCE_TO_FILE
    performance_file.close();
#endif // WRITE_PERFORMANCE_TO_FILE


    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}






