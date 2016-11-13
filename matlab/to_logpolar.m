function [lp_img,lp_map,show_map]=to_logpolar(img,x_0,y_0, N_circ,N_fov,p_vals, thet_vals,sample_radius)
    [n,m,~]=size(img);
    lp_img=zeros(N_circ,N_fov,1); % Log polar image
    lp_map=zeros(n,m,1); % Mean value for each log polar pixel show in cartesian space
    show_map=zeros(n,m); % original image with removed pixel not used for lop polar image. lp_map without mean

    for i=1:N_circ
        for j=1:N_fov
            cos1=cos((thet_vals(j)));
            sin1=sin((thet_vals(j)));
            x_tmp=p_vals(i)*cos1+x_0;
            y_tmp=p_vals(i)*sin1+y_0;
            [ points] = get_wheel( x_tmp, y_tmp, sample_radius(i),n, m  );
            lp_img(i,j)= mean(mean(img(points(:,1),points(:,2))));
            lp_map(points(:,1),points(:,2))=lp_img(i,j);
            show_map(points(:,1),points(:,2))=img(points(:,1),points(:,2));

        end
    end
    
end