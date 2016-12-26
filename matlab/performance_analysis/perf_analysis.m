clear all
close all

filename = 'movie_color_512_128_nooutputs.txt';
[frame_time,cl_runtime,cl_to_polar,cl_to_cart] = importfile(filename, 1, inf);

min_times = min([frame_time,cl_runtime,cl_to_polar,cl_to_cart]);
max_times = max([frame_time,cl_runtime,cl_to_polar,cl_to_cart]);
avg_times = mean([frame_time,cl_runtime,cl_to_polar,cl_to_cart]);
median_times = median([frame_time,cl_runtime,cl_to_polar,cl_to_cart]);
std_times = std([frame_time,cl_runtime,cl_to_polar,cl_to_cart]);

figure(1);
xlabel('Frame number');
ylabel('Time [ms]');
hold on;
plot(frame_time);
plot(cl_runtime);
plot(cl_to_polar);
plot(cl_to_cart);
axis([0 length(frame_time) 0 avg_times(1)*1.5]);
legend('Frame time', 'GPU time', 'GPU - to polar', 'GPU - to cart');