#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <CL/cl.h>
#include <iostream>
#include <fstream>

int main(int argc, const char** argv)
{
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

    cv::Mat mat_src = cv::imread("lena.png", cv::IMREAD_GRAYSCALE);
    cv::Mat mat_dst;
    if(mat_src.empty())
    {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }
    unsigned int channels = mat_src.channels();
    unsigned int depth    = mat_src.depth();

    cv::ocl::oclMat ocl_src(mat_src);
    cv::ocl::oclMat ocl_dst(mat_src.size(), mat_src.type());


    //load kernel source code
    std::ifstream in("to_logpolar.cl");
    std::string contents((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

    cv::ocl::ProgramSource program("to_logpolar", contents.c_str());


    std::size_t globalThreads[3]={ocl_src.rows, ocl_src.step, 1};
    std::vector<std::pair<size_t , const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_src.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &ocl_dst.data ));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        program, "to_logpolar", globalThreads, NULL, args, channels, depth, NULL);
    ocl_dst.download(mat_dst);

    cv::namedWindow("mat_src");
    cv::namedWindow("mat_dst");
    cv::imshow("mat_src", mat_src);
    cv::imshow("mat_dst", mat_dst);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
