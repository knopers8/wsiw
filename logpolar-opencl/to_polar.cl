//__kernel void to_logpolar(
//    __global int* In,
//    __global int* Out)
//{
//    int i = get_global_id(0);
//    int j = get_global_id(1);
//
//    Out[i*N+j] = In[i*N+j];
//}


__kernel void to_polar_C1_D0(
   __read_only __global uchar* input,
   __read_only __global int* to_polar_map_x,
   __read_only __global int* to_polar_map_y,
    __write_only __global uchar* output)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int read_pos;
    int acc = 0;

//    int16_t to_polar_map[128];
    int k;

    for( k = 0; k < 128; k++)
    {
        read_pos = to_polar_map_y[k + (40*j+i)*128];
        if ( read_pos > 0 )
        {
            read_pos += to_polar_map_x[k + (40*j+i)*128]*256;
            acc += input[read_pos];
        }
        else
        {
            break;
        }
    }
    output[64*j+i] = k>0 ? acc/k : 0;
}

