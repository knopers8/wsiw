img=imread('cameraman.tif');
close all
[n,m,~]=size(img);
x_0=n/2; 
y_0=m/2;

blind=10; % radius of blind spot, can be 0
N_r=40;   %number of rings
r_max=100; % outer raius of last ring
r_n=(r_max-blind)/N_r;   % radius of n-th ring = n*r_n n=0:N_r-1;
N_s= 40; % number of slices, just like pizza. Find better name and let me know. Number of part to divide every ring.
thet_0=0; %beggining of theta ragne
thet_max=2*pi-0.001; %end of theta ragne
theta=linspace(thet_0,thet_max,N_s+1);
% theta=theta(1:end-1) removes overlaping when theta [0, 2*pi] not sure if
% necessary
r=(0:N_r)*r_n +blind; 

[to_polar_map_r, to_polar_map_theta,show_polar_pixels,polar_img ] = to_polar_map( img, x_0,y_0,r,theta, N_r,N_s );

[ polar_img_mapped ] = remap_to_polar( img, to_polar_map_r, to_polar_map_theta, N_r, N_s);
subplot(2,2,1)

imshow(uint8(show_polar_pixels))

subplot(2,2,2)

imshow(uint8(polar_img))

subplot(2,2,3)

imshow(uint8(to_polar_map_r ))
imshow(uint8(polar_img_mapped))
subplot(2,2,4)

imshow(uint8(to_polar_map_theta ))