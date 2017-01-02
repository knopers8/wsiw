
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
    int acc_r = 0;
    int acc_g = 0;
    int acc_b = 0;
    union{ uint ui; uchar4 uch; } pix;// = 0;
    int k;

    //params: MAX_PIX_COUNT, N_s, N_r, src_height, src_width, polar_step, cart_step
    int max_pix_count = params[0];
    int N_s = params[1];
    int step = params[5];
    int src_width = params[4];
//    int map_local[128];


    int a = (N_s*j+i)*max_pix_count;
//blue = RGB & 0xff; green = (RGB >> 8) & 0xff; red = (RGB >> 16) & 0xff;
    for( k = 0; k < max_pix_count; k++)
    {
        read_pos = to_polar_map[k + a];
        if ( read_pos > 0 )
        {
            pix.ui = input[read_pos];
            acc_b += pix.uch.s2;//pix & 0xff;
            acc_g += pix.uch.s1;//(pix >> 8) & 0xff;
            acc_r += pix.uch.s0;//(pix >> 16) & 0xff;
        }
        else
        {
            break;
        }
    }
    if( k > 0 )
    {
        pix.uch.s0 = acc_r/k;
        pix.uch.s1 = acc_g/k;//(pix << 8) | (acc_g/k);
        pix.uch.s2 = acc_b/k;//(pix << 8) | (acc_b/k);
        output[step*j+i] = pix.ui;
    }
    else
    {
        output[step*j+i] = 0;//k>0 ? acc/k : 0;
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
