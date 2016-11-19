function [to_polar_map_r, to_polar_map_theta,show_polar_pixels,polar_img ] = to_polar_map( img, x_0,y_0,r,theta, N_r,N_s )
% Function is responsible for creating cordinates map from cartesian space
% to polar space. 
% img is input image with size n x m
% for other inputs check main, im to lazy
%  to_polar_map_r n x m array with r index in corespongind N_r x N_s polar
%  image. Inedx, not value!
%  to_polar_map_theta - as above but theta
% polar_img - ready polar image, displaying and debugging mostly
% show_polar_pixels - polar image displayed in cartesian space


[n,m,~]=size(img);

to_polar_map_r=zeros(n,m);
to_polar_map_theta=zeros(n,m);

show_polar_pixels=zeros(n,m);
polar_img=zeros(N_r,N_s);

for i=1:N_r
    for j=1:N_s
        in_R=r(i);
        out_R=r(i+1);
        [ value,cord,x_cordinates, y_cordinates ] = get_polar_pixel( img,x_0, y_0, in_R, out_R, theta(j), theta(j+1) );
        if ~isnan(value) %when get_polar_pixel finds no matching pixels cord.. arrays are empty and value=NaN
%          show_polar_pixels(x_cordinates,y_cordinates)=value;
            [ show_polar_pixels ] = write_value( show_polar_pixels, x_cordinates, y_cordinates, value); %work arond of the above line
            polar_img(i,j)=value;
            [ to_polar_map_r ] = write_value( to_polar_map_r, x_cordinates, y_cordinates, i);
            [ to_polar_map_theta] = write_value( to_polar_map_theta, x_cordinates, y_cordinates, j);
                     
           %             hold on; % ploting dots
%             plot(x_cordinates, y_cordinates,'.');
        end
    end
end
end

