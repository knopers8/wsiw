function [ points] = get_wheel( x_0, y_0, r,x_ma, y_max  )
% Function calulate and returns cordinates of pixels inside circle with
% radius r in miidle in point (x_0,y_0)
% points has 2 colums 1st one stores x cordinates and 2nd y cordinates 
    points=[];
    if r <0
        error('Troche maly ten promien, co nie?')
    elseif round(r) <=1
        points=[round(x_0) round(y_0)];
    else
        square_x=[floor(x_0-r):ceil(x_0+r)]; 
        square_y=[floor(y_0-r):ceil(y_0+r)];
        for i=1:length(square_x)
            for j=1:length(square_y)
                r_tmp=sqrt((square_x(i)-x_0)^2+(square_y(j)-y_0)^2);
                if r_tmp<=r
                    points=[points; square_x(i) square_y(j)];
                end
            end  
        end
    end


end

