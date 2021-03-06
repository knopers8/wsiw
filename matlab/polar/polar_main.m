img=imread('cameraman.tif');
close all
[n,m,~]=size(img);
x_0=n/2; 
y_0=m/2;

blind=10; % radius of blind spot, can be 0
N_r=40;   %number of rings
r_max=100; % outer radius of last ring
r_n=(r_max-blind)/N_r;   % radius of n-th ring = n*r_n n=0:N_r-1;
N_s= 40; % number of slices. Number of part to divide every ring.
thet_0=0; %beggining of theta range
thet_max=2*pi-0.001; %end of theta ragne
theta=linspace(thet_0,thet_max,N_s+1);

r=(0:N_r)*r_n +blind; 

[to_polar_map_r, to_polar_map_theta,show_polar_pixels,polar_img, to_polar_map_x, to_polar_map_y] = to_polar_map( img, x_0,y_0,r,theta, N_r,N_s );

[polar_img_mapped] = remap_to_polar( img, to_polar_map_x, to_polar_map_y, N_r, N_s);

[to_cart_map_r,to_cart_map_theta] = to_cart_map( img, n,m, x_0,y_0,r,theta, N_r,N_s );
 
polar_img2=rotate_lp_image( polar_img_mapped, 10 );

[ img2] = remap_to_cart( polar_img2 , to_cart_map_r,  to_cart_map_theta);

subplot(2,2,1)
imshow(uint8(img))
title('Input image')

subplot(2,2,2)
imshow(uint8(show_polar_pixels))
title('Polar pixels map verryfication')

subplot(2,2,3)
imshow(uint8(polar_img_mapped))
title('Polar image')

subplot(2,2,4)
imshow(uint8(img2))
title('Inverse polar transform')
