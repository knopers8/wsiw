function [ polar_img2] = rotate_lp_image( polar_img, rotation )
% Function rotates image shifting colums by rotation parameter. 

    [~,N_s]=size(polar_img);
    polar_img2=zeros( size(polar_img));
    

    for i=1:N_s
        i2=mod(N_s+i+rotation,N_s)+1;
        polar_img2(:,i)=polar_img(:,i2);

    end

end

