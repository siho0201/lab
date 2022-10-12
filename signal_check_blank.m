%% read file   
clear all;
close all;

Fs = 10e6; %sampling frequency
on_off_factor = [0 1 0]; % radar dvb wifi
test_file_factor = [1 0]; % test / test_test
problem_factor = [0 1]; % 1,2
slot_number = 10;

if test_file_factor(1) == 1
    test_path = 'test_data';
elseif test_file_factor(2) == 1
    test_path = 'test_test_data';
else
    test_path = 'original';
end

if problem_factor(1) == 1
    problem_num = '1';
elseif problem_factor(2) == 1
    problem_num = '2';
end

for i=slot_number:slot_number
    if on_off_factor(1) == 1
        eval(['fid=fopen("./',problem_num,'/',test_path,'/radar/radar_test',num2str(i),'.bin","r");']);
        eval(['fileID=py.open("./',problem_num,'/',test_path,'/radar/radar_list',num2str(i),'.bin",''rb'');'])
    elseif on_off_factor(2) == 1
        eval(['fid=fopen("./',problem_num,'/',test_path,'/dvb/dvb_test',num2str(i),'.bin","r");']);
        eval(['fileID=py.open("./',problem_num,'/',test_path,'/dvb/dvb_list',num2str(i),'.bin",''rb'');'])
    else
        eval(['fid=fopen("./',problem_num,'/',test_path,'/wifi/wifi_test',num2str(i),'.bin","r");']);
        eval(['fileID=py.open("./',problem_num,'/',test_path,'/wifi/wifi_list',num2str(i),'.bin",''rb'');'])
    end
    data = fread(fid,'*float');
    test_ans = double(py.pickle.load(fileID));
    test_ans = reshape(test_ans,[10,10]);
    test_ans = test_ans';
    fclose(fid);
    fileID.close();
    data_re = reshape(data,2,[]);
    I_data = data_re(1,1:end);
    Q_data = data_re(2,1:end);
    real_data = I_data + 1i*Q_data;
    
    energy = zeros(10,10);
    for check_index = 1:10
        for sample_index = 1:10
            start_num = 300000*(check_index-1) + 30000*(sample_index-1)+1;
            end_num = 300000*(check_index-1) + 30000 * sample_index;
            energy(check_index,sample_index) = mean(abs(real_data(start_num:end_num)));
            FFT = fft(real_data(start_num:end_num),[],2);
            Y = fftshift(FFT);
            L = end_num - start_num; % signal length
            fshift = (-L/2:L/2)*(Fs/L);
            Amshift = abs(Y)/L;
        end
    end

%% find blank ver1
%     threshold = mean(energy,"all");
%     abnormal_idx = energy < threshold*0.7;
%     [x,y] = find(abnormal_idx);
%     [normal_x, normal_y] = find(~abnormal_idx);
%     
%     for index = 1:length(x)
%         start_num = 300000*(x(index)-1) + 30000*(y(index)-1)+1;
%         end_num = 300000*(x(index)-1) + 30000*y(index);
%         real_data(start_num:end_num);
%         if mod(y(index),2) == 0 && ~(x(index) == 10 && y(index) == 10) % even_num
%             for normal_index=1:length(normal_x)
%                 if mod(normal_y(normal_index),2) == 0
%                     break;
%                 end
%             end
%             normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
%             normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
%             real_data(start_num:end_num) = real_data(normal_start:normal_end);
%         elseif mod(y(index),2) == 1 
%             for normal_index=1:length(normal_x)
%                 if mod(normal_y(normal_index),2) == 1
%                     break;
%                 end
%             end
%             normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
%             normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
%             real_data(start_num:end_num) = real_data(normal_start:normal_end);
%         
%         end
%     end
    
%% find blank ver2
    threshold = mean(energy,"all");
    abnormal_idx = energy < threshold*0.7;
    [x,y] = find(abnormal_idx);
    [normal_x, normal_y] = find(~abnormal_idx);
    
    for index = 1:length(x)
        start_num = 300000*(x(index)-1) + 30000*(y(index)-1)+1;
        end_num = 300000*(x(index)-1) + 30000*y(index);
        real_data(start_num:end_num);
        if mod(y(index),5) == 0 && ~(x(index) == 10 && y(index) == 10) % even_num
            for normal_index=1:length(normal_x)
                if mod(normal_y(normal_index),5) == 0
                    break;
                end
            end
            normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
            normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
            real_data(start_num:end_num) = real_data(normal_start:normal_end);
        elseif mod(y(index),5) == 1 
            for normal_index=1:length(normal_x)
                if mod(normal_y(normal_index),5) == 1
                    break;
                end
            end
            normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
            normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
            real_data(start_num:end_num) = real_data(normal_start:normal_end);
        elseif mod(y(index),5) == 2 
            for normal_index=1:length(normal_x)
                if mod(normal_y(normal_index),5) == 2
                    break;
                end
            end
            normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
            normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
            real_data(start_num:end_num) = real_data(normal_start:normal_end);     
        elseif mod(y(index),5) == 3 
            for normal_index=1:length(normal_x)
                if mod(normal_y(normal_index),5) == 3
                    break;
                end
            end
            normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
            normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
            real_data(start_num:end_num) = real_data(normal_start:normal_end);
        elseif mod(y(index),5) == 4 
            for normal_index=1:length(normal_x)
                if mod(normal_y(normal_index),5) == 4
                    break;
                end
            end
            normal_start = 300000*(normal_x(normal_index)-1) + 30000*(normal_y(normal_index)-1)+1;
            normal_end = 300000*(normal_x(normal_index)-1) + 30000*normal_y(normal_index);
            real_data(start_num:end_num) = real_data(normal_start:normal_end);
        end
    end
%% check result
    figure('name','result','NumberTitle','off')
    plot(abs(real_data(1:3000000)));
    set(gcf,'Position',[1000 1000 2000 1500])
    energy = zeros(10,10);
    for check_index = 1:10
        for sample_index = 1:10
            start_num = 300000*(check_index-1) + 30000*(sample_index-1)+1;
            end_num = 300000*(check_index-1) + 30000 * sample_index;
            energy(check_index,sample_index) = mean(abs(real_data(start_num:end_num)));
        end
    end
    abnormal_idx = energy < threshold;
%% save
    if on_off_factor(1) == 1
        eval(['radar_test',num2str(i),' = real_data;'])
        eval(['save ./',problem_num,'/',test_path,'/radar/radar_test',num2str(i),'.mat radar_test',num2str(i)])
    elseif on_off_factor(2) == 1
        eval(['dvb_test',num2str(i),' = real_data;'])
        eval(['save ./',problem_num,'/',test_path,'/dvb/dvb_test',num2str(i),'.mat dvb_test',num2str(i)])
    else
        eval(['wifi_test',num2str(i),' = real_data;'])
        eval(['save ./',problem_num,'/',test_path,'/wifi/wifi_test',num2str(i),'.mat wifi_test',num2str(i)])
    end
    

end
%% check not_blank
for check_index = 1:10
    figure('name',string(check_index),'NumberTitle','off')
    set(gcf,'Position',[1000 1000 2000 1500])
    for sample_index = 1:10
        start_num = 300000*(check_index-1) + 30000*(sample_index-1)+1;
        end_num = 300000*(check_index-1) + 30000 * sample_index;
        
        subplot(2,10,sample_index);
        plot(abs(real_data(start_num:end_num)));
        ylim([0 0.1])
        title(string(test_ans(check_index,sample_index)))
        energy(check_index,sample_index) = mean(abs(real_data(start_num:end_num)));

        FFT = fft(real_data(start_num:end_num),[],2);
        Y = fftshift(FFT);
        L = end_num - start_num; % signal length
        fshift = (-L/2:L/2)*(Fs/L);
        Amshift = abs(Y)/L;
        subplot(2,10,10+sample_index)
        plot(fshift,Amshift);
        ylim([0 0.0025])
    end
end

figure('name','real','NumberTitle','off')
set(gcf,'Position',[1000 1000 2000 1500])
plot(abs(real_data(1:3000000)));
