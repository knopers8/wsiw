clear all
close all

filename = 'movie_gs_512_128.txt';
[frame_time,cl_runtime,cl_to_polar,cl_imgproc,cl_to_cart] = importfile(filename, 1, inf);

a = [frame_time,cl_runtime,cl_to_polar,cl_imgproc,cl_to_cart];
b = a(2:end,:);


min_times = min(b);
max_times = max(b);
avg_times = mean(b);
median_times = median(b);
std_times = std(b);

figure(1);
xlabel('Frame number');
ylabel('Time [ms]');
hold on;
plot(frame_time);
plot(cl_runtime);
plot(cl_to_polar);
plot(cl_imgproc);
plot(cl_to_cart);
axis([0 length(frame_time) 0 avg_times(1)*1.8]);
legend('Frame time', 'GPU time', 'GPU - to polar', 'GPU - image proc.', 'GPU - to cart');

%[min_times; max_times; median_times; avg_times; std_times]'
median_times