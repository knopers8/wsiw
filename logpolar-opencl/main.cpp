
#define __CL_ENABLE_EXCEPTIONS

#include "opencv2\opencv.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <opencv/cv.h>

#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"




#define ORDER 64
#define COUNT 1

//--------------------------------------------------------------
//
// In order to compile, set environment variable CUDA_PATH
// to your OpenCL installation path, for example:
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
//
//--------------------------------------------------------------






//------------------------------------------------------------------------------
//
//  Function to analyze and output results
//
//------------------------------------------------------------------------------
void results(int N, std::vector<int>& C, double run_time)
{

    float mflops;
    //float errsq;

    mflops = 2.0 * N * N * N/(1000000.0f * run_time);
    printf(" %.2f seconds at %.1f MFLOPS \n",  run_time,mflops);

}



int main(int argc, char *argv[])
{

    int N;                  // A[N][N], B[N][N], C[N][N]
    int mat_size;               // Number of elements in each matrix


    double start_time;      // Starting time
    double run_time;        // Timing
    util::Timer timer;      // Timing

    N    = ORDER;
    mat_size = N * N;

    std::vector<int> h_In(mat_size); // Host memory for Matrix A
    std::vector<int> h_Out(mat_size); // Host memory for Matrix B

    cl::Buffer d_In, d_Out;   // Matrices in device memory

    //
    // OpenCV init
    //

    cv::VideoCapture cap(0);

	if (!cap.isOpened())  // check if we succeeded
		return -1;

    cv::Mat frame;
    cap.read(frame);

    cv::imshow("frame", frame);

    cv::waitKey(10);
//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {

        cl_uint deviceIndex = 2;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

//--------------------------------------------------------------------------------
// Run sequential matmul
//--------------------------------------------------------------------------------

        for (int i = 0; i < mat_size; i++)
            h_In[i] = rand();
//
//
//        timer.reset();
//
//        printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",N);
//        for(int i = 0; i < COUNT; i++)
//        {
//            for( int j = 0; j < mat_size; j++)
//                h_Out[j] = 0;
//
//            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
//
//            for (int j = 0; j < mat_size; j++)
//                h_Out[j] = h_In[j];
//
//
//            run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
//            results(N, h_C, run_time);
//        }

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

        d_In = cl::Buffer(context, h_In.begin(), h_In.end(), true);

        d_Out = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * mat_size);

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------

        timer.reset();

        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("to_logpolar.cl"), true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer> to_logpolar(program, "to_logpolar");

        printf("\n===== OpenCL, to_logpolar ======\n");

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            for( int j = 0; j < mat_size; j++)
                h_Out[j] = 0;

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            cl::NDRange global(N, N);
            to_logpolar(cl::EnqueueArgs(queue, global),
                    N, d_In, d_Out);

            queue.finish();

            run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;

            cl::copy(queue, d_Out, h_Out.begin(), h_Out.end());

            results(N, h_Out, run_time);

            int acc=0;
            for( int j = 0; j < mat_size; j++)
                acc += abs(h_In[j]-h_Out[j]);

            printf("example: %d %d error: %d", h_In[31], h_Out[31], acc);

        } // end for loop

    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << cl_err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
