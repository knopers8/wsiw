__kernel void to_logpolar(
    const int N,
    __global int* In,
    __global int* Out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    Out[i*N+j] = In[i*N+j];
}
