function [ polar_img,cnt ] = remap_to_polar( img, to_polar_map_r, to_polar_map_theta, N_r, N_s)

    polar_img=zeros(N_r, N_s);
    cnt=zeros(N_r, N_s); % stores number of cartesian pixels corresponding to log polar pixels
    [n,m,~]=size(img);
    
    for i =1:n
        for j=1:m
            r=to_polar_map_r(i,j);
            theta=to_polar_map_theta(i,j);
            if and(r>0,theta>0) % skip not mapped pixels
                polar_img(r,theta)=polar_img(r,theta)+double(img(i,j)); % sum all cartesian pixels
                cnt(r,theta)= cnt(r,theta)+1;
                1;
            end
        end
    end
    
    for i =1:N_r
        for j=1:N_s
            polar_img(i,j)=polar_img(i,j)./cnt(i,j); % divide / amount of cartesia pixels
        end
    end

end

