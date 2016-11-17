//__kernel void to_logpolar(
//    __global int* In,
//    __global int* Out)
//{
//    int i = get_global_id(0);
//    int j = get_global_id(1);
//
//    Out[i*N+j] = In[i*N+j];
//}


__kernel void to_logpolar_C1_D0(
   __read_only __global uchar* input,
   __write_only __global uchar* output)
{
    int N = 512;
   int i = get_global_id(0);
   int j = get_global_id(1);
   output[N*j+i] = 255 - input[N*j+i];

}
