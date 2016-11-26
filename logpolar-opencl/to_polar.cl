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
   __read_only __constant int* params,
    __write_only __global uchar* output)
{

//    int i = get_global_id(0);
    int j = get_global_id(1);

    int read_pos;
    int acc = 0;

    //params: { MAX_PIX_COUNT, N_s, N_r, src_height, src_width, step}
    int k;
    int a;
    int max_pix_count = params[0];
    int N_s = params[1];
    int step = params[5];
    int src_width = params[4];
    int map_local_x[128];
    int map_local_y[128];

    a = (N_s*j)*max_pix_count;
    for( int i = 0; i < N_s; i++)
    {
        a += max_pix_count;

        //barrier( CLK_GLOBAL_MEM_FENCE );

        for( k = 0; k < max_pix_count; k++)
        {
            map_local_x[k] = to_polar_map_x[k + a];
            map_local_y[k] = to_polar_map_y[k + a];
        }
        map_local_x[k] = 0;
        map_local_y[k] = 0;

        //barrier( CLK_GLOBAL_MEM_FENCE );

        acc = 0;
        for( k = 0; k < max_pix_count; k++)
        {
            read_pos = map_local_y[k];
            if ( read_pos > 0 )
            {
                read_pos += map_local_x[k]*src_width;
                acc += input[read_pos];
            }
            else
            {
                break;
            }
        }

        //barrier( CLK_GLOBAL_MEM_FENCE );

        output[step*j+i] = k>0 ? acc/k : 0;
    }

}

