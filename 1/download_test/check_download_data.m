%% load
clear all;
close all;

test_file_factor = [1 0]; % test / test_test
original_signal_factor = [1 0 0]; % radar / DVB-T2 / WIFI
signal_factor = [0 0 1]; % radar / DVB-T2 / WIFI

time_slot = 315;

start_num =1+30000*(time_slot-1);
end_num = 30000*time_slot;
L = end_num -start_num;
Fs = 10e6;

load("./original_data/Radar.mat");
load("./original_data/WIFI.mat");
load("./original_data/DVB_T2.mat");

if test_file_factor(1) == 1
    test_path = 'test_data';
    eval(['load ./',test_path,'/IQ_2_test.mat']);
elseif test_file_factor(2) == 1
    test_path = 'test_test_data';
    eval(['load ./',test_path,'/IQ_2_valid.mat']);
end

%% check time-frequency data

FFT = fft(IQ_2_test(start_num:end_num),[],2);
Y = fftshift(FFT);
fshift = (-L/2:L/2)*(Fs/L);
Amshift = abs(Y/L);

subplot(2,1,1);
plot(abs(IQ_2_test(start_num:end_num))) 

subplot(2,1,2);
plot(fshift,Amshift) 

eval(['fileID=py.open("./',test_path,'/ANS_2_0.dat",''rb'');'])
test_ans = double(py.pickle.load(fileID));