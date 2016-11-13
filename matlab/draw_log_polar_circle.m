function [] = draw_log_polar_circle(img,x_0,y_0, p_vals, thet_vals, sample_radius)
% Function draws circles on image img. For visualisation and debuggin only.
imshow(img)
hold on
for i=1:length(p_vals)
        for j=1:length(thet_vals)
%              pos = [p_vals(i)-sample_radius(i) -sample_radius(i) 2*sample_radius(i) 2*sample_radius(i)];
            cos1=cos((thet_vals(j)));
            sin1=sin((thet_vals(j)));
            x=p_vals(i)*cos1-abs(sample_radius(i)*cos1)+x_0;
            y=p_vals(i)*sin1+y_0-abs(sample_radius(i)*sin1);
            pos=[x y  2*sample_radius(i) 2*sample_radius(i)];
            rectangle('Position',pos,'Curvature',[1 1])
            axis equal
        end
end