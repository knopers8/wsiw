function [ out_array ] = write_value( array, x_cordinates, y_cordinates, value)
%Matlab vector notatnion decided to show middle finger so here it goes,
%should also help in C
    out_array=array;
    for i=1:length(x_cordinates)
%         for j=1:length(y_cordinates)
            out_array(x_cordinates(i),y_cordinates(i))=value;
%         end
    end


end

