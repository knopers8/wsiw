function [ img] = remap_to_cart( polar_img, to_cart_map_r,  to_cart_map_theta)

    [n,m]=size(to_cart_map_r);
    img=zeros(n,m);
     
    
    for i=1:n
       for j=1:m
           r_cord=to_cart_map_r(i,j);
           thet_cord=to_cart_map_theta(i,j);
           if and(r_cord>0, thet_cord>0)
               img(i,j)=polar_img(r_cord,thet_cord);
           end
       end
    end

end

