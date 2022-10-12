%% load
clear all;
close all;

test_file_factor = [1 0]; % test / test_test
original_signal_factor = [0 0 1]; % radar / DVB-T2 / WIFI
signal_factor = [0 0 0 1]; % radar / DVB-T2 / WIFI / all

time_slot = 4;
start_num =1;
end_num = 30000;
L = end_num -start_num;
Fs_0 = 1e6;
Fs = 10e6;
L0 = 2999;

load("./original_data/Radar.mat");
load("./original_data/WIFI.mat");
load("./original_data/DVB_T2.mat");

if test_file_factor(1) == 1
    test_path = 'test_data';
    test_valid = 'test';
elseif test_file_factor(2) == 1
    test_path = 'test_test_data';
    test_valid = 'valid';
end

eval(['load ./',test_path,'/IQ_2_',test_valid,'.mat']);
% eval(['load("./',test_path,'/radar/radar_test1.mat")']);
% eval(['load("./',test_path,'/dvb/dvb_test1.mat")']);
% eval(['load("./',test_path,'/wifi/wifi_test1.mat")']);

if original_signal_factor(1) == 1
    signal = Radar;
    zero_vector0 = zeros(1,600);
    signal = [signal zero_vector0];
    L0 = 2399;
elseif original_signal_factor(2) == 1
    signal = DVB_T2;
    zero_vector1 = zeros(1,952);
    signal = [signal zero_vector1];
    L0 = 2047;
elseif original_signal_factor(3) == 1
    signal = WIFI;
    zero_vector2 = zeros(1,1960);
    signal = [signal zero_vector2];
    L0 = 1039;
end

FFT = fft(signal,[],2);
Y0 = fftshift(FFT);
fshift1 = (-L0/2:L0/2)*(Fs_0/L0);
Amshift1 = abs(Y0/L0);

if signal_factor(1) == 1
    signal1 = radar_test1(time_slot,start_num:end_num);
%     check_start_num = 13500;
%     check_end_num = 16500;
elseif signal_factor(2) == 1
    signal1 = dvb_test1(time_slot,start_num:end_num);
%     check_start_num = 1500;
%     check_end_num = 4500;
elseif signal_factor(3) == 1
    signal1 = wifi_test1(time_slot,start_num:end_num);
%     check_start_num = 7500;
%     check_end_num = 10500;
elseif signal_factor(4) == 1
    if test_file_factor(1) == 1
        signal1 = IQ_2_test(time_slot,start_num:end_num); 
    elseif test_file_factor(2) == 1 
        signal1 = IQ_2_valid(time_slot,start_num:end_num);
    end
end

FFT = fft(signal1,[],2);
Y = fftshift(FFT);
fshift2 = (-L/2:L/2)*(Fs/L);
Amshift2 = abs(Y/L);

zero_vector = zeros(1,27000);

nor_Amshift1 = Amshift1/max(Amshift1);
nor_Amshift1 = [nor_Amshift1 zero_vector];
nor_Amshift2 = Amshift2/max(Amshift2);

signal = signal/max(signal);
signal1 = signal1/max(signal1);

% subplot(2,1,1)
% plot(Amshift1/max(Amshift1))
% subplot(2,1,2)
% plot(Amshift2(check_start_num:check_end_num)/max(Amshift2(check_start_num:check_end_num)))

% subplot(2,1,1)
% plot(abs(signal))
% subplot(2,1,2)
% plot(abs(signal1))
%% frequency
[C,lag] = xcorr(nor_Amshift1,nor_Amshift2,'coeff');
stem(lag,C)
%% time
% x = 1:1:3000;
% xq = 1.1:0.1:3001;
% sampled_signal = interp1(x,signal,xq,'spline');
% plot(abs(sampled_signal))
% [C,lag] = xcorr(abs(sampled_signal),abs(signal1),'coeff');
% stem(lag,C)