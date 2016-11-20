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
    __constant float * thet_vals_cos,
    __constant float * thet_vals_sin,
    __constant float * p_vals,
//    __constant int center_h,
//    __constant int center_w,
    __write_only __global uchar* output)
{
    int N = 50;
    int i = get_global_id(0); //kolumna
    int j = get_global_id(1); //wiersz

    int center_w = 128;
    int center_h = 128;


    int read_pos;

    read_pos = p_vals[j] * thet_vals_sin[i] + center_w;
    read_pos += 256 * ( p_vals[j] * thet_vals_cos[i] + center_h);

    output[64*j+i] = input[read_pos];
}

