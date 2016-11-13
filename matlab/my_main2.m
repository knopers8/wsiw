img=imread('cameraman.tif');
close all
[n,m,~]=size(img);
x_0=n/2; 
y_0=m/2;
N_fov=50; % 24,50% total number of parts to divide each ring, each 2pi/N_fov wide
N_circ=39; % 18,39   % numbers of rings 

%asdasd

b=(N_fov+pi)/(N_fov-pi);  % base of logarithm
p_fov=(N_fov/pi); % size of circle at (x,y)=(x_0,y_0), blind spot


p_vals=0:N_circ-1;
p_vals=b.^p_vals; % R values  b^i  i: [0, N_circ-1]

thet_vals=(0:N_fov-1)*((2*pi)/N_fov); % theta values 

sample_radius=(pi.*p_vals)./N_fov; % radius of single wheel on 'i' ring (single log-polar pixel)

figure;
subplot(2,2,1)
draw_log_polar_circle(img,x_0,y_0, p_vals, thet_vals, sample_radius);




[lp_img,lp_map,show_map]=to_logpolar(img,x_0,y_0, N_circ,N_fov,p_vals, thet_vals,sample_radius);

subplot(2,2,2)
imshow(uint8(show_map))
subplot(2,2,3)
imshow(uint8(lp_map))
subplot(2,2,4)
imshow(uint8(lp_img))


% 
% %%Cordinates map
% map_p=zeros(n,m);
% map_thet=zeros(n,m);
% for i=1:n
%     for j=1:m
%         R=sqrt((i-x_0)^2+(j-y_0)^2);
%         map_p(i,j)=log_n(R/r_0,b);
%         map_thet(i,j)=(N_fov/(2*pi))*atan2((i-x_0),(j-y_0));
%     end
% end
% p_vals=[4:N_fov];
% thet_vals=linspace(0,2*pi,N_fov);
% lp_map=zeros(N_fov,p_0,2);
% 
% for i=1:N_fov
%     for j=1:p_0
%        
%         lp_map(i,j,1)=(b^j)*r_0*cos((2*pi*i)/N_fov)+x_0;
%         lp_map(i,j,2)=(b^j)*r_0*sin((2*pi*i)/N_fov)+y_0;
%     end
% end
% 
% %%Fovea
% N_fov=round(N_fov);
% 
% %%Periphery
% 
