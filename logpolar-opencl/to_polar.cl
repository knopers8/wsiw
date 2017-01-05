
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
    int k;

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int max_pix_count = params[0];
    int N_s = params[1];
    int step = params[5];
    int src_width = params[4];
//    int map_local[128];

    int a = (N_s*j+i)*max_pix_count;

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
}

__kernel void processing_C1_D0(
   __read_only __global uchar* input,
   __read_only __constant int* params,
    __write_only __global uchar* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int step = params[5];
    int N_s = params[1];

    int shift = N_s/6;

    if ( i + shift > N_s)
        output[step*j+i+shift-N_s] = 255 - input[step*j+i];
    else
        output[step*j+i+shift] = 255 - input[step*j+i];
}

__kernel void to_cart_C1_D0(
   __read_only __global uchar* input,
   __read_only __global int* to_cart_map,
   __read_only __constant int* params,
    __write_only __global uchar* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int step = params[6];
    int addr = to_cart_map[i+(j*step)];

    output[i+j*step] = addr ? input[ addr ] : 0;
}



__kernel void to_polar_C3_D0(
   __read_only __global uint* input,
   __read_only __global int* to_polar_map,
   __read_only __constant int* params,
    __write_only __global uint* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    int read_pos;
    uint3 acc = {0,0,0};
    union{ uint ui; uchar4 uch; } pix;// = 0;
    int k;

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int max_pix_count = params[0];
    int N_s = params[1];
    int step = params[5];
    int src_width = params[4];


    int a = (N_s*j+i)*max_pix_count;

    for( k = 0; k < max_pix_count; k++)
    {
        read_pos = to_polar_map[k + a];
        if ( read_pos > 0 )
        {
            pix.ui = input[read_pos];
            acc += (uint3)((uint)pix.uch.s0, (uint)pix.uch.s1, (uint)pix.uch.s2);
        }
        else
        {
            break;
        }
    }
    if( k > 0 )
    {
        acc = acc / (uint3)(k,k,k);
        pix.uch = (uchar4)((uchar)acc.s0, (uchar)acc.s1, (uchar)acc.s2, 0);

        output[step*j+i] = pix.ui;
    }
    else
    {
        output[step*j+i] = 0;
    }

}

__kernel void processing_C3_D0(
   __read_only __global uchar4* input,
   __read_only __constant int* params,
    __write_only __global uchar4* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int step = params[5];
    int N_s = params[1];
    int shift = N_s/6;

    if ( i + shift > N_s)
        output[step*j+i+shift-N_s] = (uchar4)(255,255,255,0) - input[step*j+i];
    else
        output[step*j+i+shift] = (uchar4)(255,255,255,0) - input[step*j+i];
}

__kernel void to_cart_C3_D0(
   __read_only __global uint* input,
   __read_only __global int* to_cart_map,
   __read_only __constant int* params,
    __write_only __global uint* output)
{

    int i = get_global_id(0);
    int j = get_global_id(1);

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int step = params[6];
    int addr = to_cart_map[i+(j*step)];

    output[i+j*step] = addr ? input[ addr ] : 0;
}
