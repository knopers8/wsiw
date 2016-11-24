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
   __read_only __global int* params,
    __write_only __global uchar* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    int read_pos;
    int acc = 0;

    //params: { MAX_PIX_COUNT, N_s, N_r, src_height, src_width, step}
    int k;
    int a = (params[1]*j+i)*params[0];
    int step = params[5];
    int src_width = params[4];

    for( k = 0; k < params[0]; k++)
    {
        read_pos = to_polar_map_y[k + a];
        if ( read_pos > 0 )
        {
            read_pos += to_polar_map_x[k + a]*src_width;
            acc += input[read_pos];
        }
        else
        {
            break;
        }
    }

    output[step*j+i] = k>0 ? acc/k : 0;
}

