clear all;
close all;

Fs = 10e6; %sampling frequency
test_file_factor = [1 0]; % test / test_test
problem_factor = [0 1]; % 1,2


if test_file_factor(1) == 1
    test_path = 'test_data';
elseif test_file_factor(2) == 1
    test_path = 'test_test_data';
end

if problem_factor(1) == 1
    problem_num = '1';
elseif problem_factor(2) == 1
    problem_num = '2';
end

for i = 1:10
    eval(['load("./',problem_num,'/',test_path,'/radar/radar_test',num2str(i),'.mat")'])
    eval(['load("./',problem_num,'/',test_path,'/dvb/dvb_test',num2str(i),'.mat")'])
    eval(['load("./',problem_num,'/',test_path,'/wifi/wifi_test',num2str(i),'.mat")'])
end

radar = [radar_test1 radar_test2 radar_test3 radar_test4 radar_test5 radar_test6 radar_test7 radar_test8 radar_test9 radar_test10];
dvb = [dvb_test1 dvb_test2 dvb_test3 dvb_test4 dvb_test5 dvb_test6 dvb_test7 dvb_test8 dvb_test9 dvb_test10];
wifi = [wifi_test1 wifi_test2 wifi_test3 wifi_test4 wifi_test5 wifi_test6 wifi_test7 wifi_test8 wifi_test9 wifi_test10];
final_data = radar + dvb + wifi;

start_num = 1;
end_num = 30000000;
subplot(2,1,1);
plot(abs(final_data(start_num:end_num)));
FFT = fft(final_data(start_num:end_num),[],2);
Y = fftshift(FFT);
L = end_num - start_num; % signal length
fshift = (-L/2:L/2)*(Fs/L);
Amshift = abs(Y)/L;
subplot(2,1,2)
plot(fshift,Amshift);

if test_file_factor(1) == 1
    IQ_2_test = reshape_IQ(final_data);
    eval(['save ./',problem_num,'/',test_path,'/IQ_2_test.mat IQ_2_test']);
elseif test_file_factor(2) == 1
    IQ_2_valid = reshape_IQ(final_data);
    eval(['save ./',problem_num,'/',test_path,'/IQ_2_valid.mat IQ_2_valid']);
end
%% test
plot(abs(IQ_2_test(1,1:30000)))