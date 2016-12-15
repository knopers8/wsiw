
__kernel void to_polar_C1_D0(
   __read_only __global uchar* input,
   __read_only __global int* to_polar_map,
   __read_only __constant int* params,
    __write_only __global uchar* output)
{

    int i = get_global_id(0);
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
    int map_local[128];


    a = (N_s*j+i)*max_pix_count;

    acc = 0;
    for( k = 0; k < max_pix_count; k++)
    {
        read_pos = to_polar_map[k + a];
        if ( read_pos > 0 )
        {
            acc += input[read_pos];
        }
        else
        {
            break;
        }
    }
    output[step*j+i] = k>0 ? acc/k : 0;
//    if (i == 127 && j ==127)
//        printf("topolar pix: %d\n", acc/k);

}


__kernel void to_cart_C1_D0(
   __read_only __global uchar* input,
   __read_only __global int* to_cart_map,
//   __read_only __constant int* params,
    __write_only __global uchar* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    int addr = to_cart_map[i+(j*512)];
    output[i+j*512] = addr ? input[ addr ] : 0;

}
