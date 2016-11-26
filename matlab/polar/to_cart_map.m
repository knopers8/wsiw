function [ to_cart_map_r,  to_cart_map_theta ] = to_cart_map( img, n,m, x_0,y_0,r,theta, N_r,N_s )

    to_cart_map_r=zeros(n,m);
    to_cart_map_theta=zeros(n,m);

    for i=1:N_r
        for j=1:N_s
            in_R=r(i);
            out_R=r(i+1);
            [ value,~,x_cordinates, y_cordinates ] = get_polar_pixel( img,x_0, y_0, in_R, out_R, theta(j), theta(j+1) );
            if ~isnan(value) %when get_polar_pixel finds no matching pixels cord.. arrays are empty and value=NaN       
                [ to_cart_map_r ] = write_value( to_cart_map_r , x_cordinates, y_cordinates, i);
                [ to_cart_map_theta] = write_value( to_cart_map_theta, x_cordinates, y_cordinates, j);
            end
        end
    end
end

