% #########################################################################
% #                    Pakistan Navy Engineering College                  #              
% #               National University Of Science & Technology             #
% #########################################################################
%-------------------------------Copyright----------------------------------
% Author: Engr. Muhammad Khizar Abbas          Email: khizar.abass@gmail.com
% Supervisor : Capt. Dr. Syed Sajjad Zaidi - PN
% Department : Electronics & Power Engineering

% Matlab Code : Normal Brain State Detection Part 
clc
clear all 
close all
 eeg_fs=200;
 wavlet_level=4;
% ############## Load Data Seizure ###################
load hseizure1_fs200;load hseizure2_fs200;load hseizure3_fs200;load hseizure4_fs200;
load hseizure5_fs200;load hseizure6_fs200;load hseizure7_fs200;load hseizure8_fs200;
load hseizure9_fs200;load hseizure10_fs200;
load seizure1_fs200;load seizure2_fs200;
%%%%%%%%%%%%%%%%%Noraml Brain Wave Part%%%%%%%%%%%%%%%%%%%%%%%
% Load Data
load normal1_fs200;load normal2_fs200;load hnormal1_fs200;
load hnormal2_fs200;load hnormal3_fs200;load hnormal4_fs200;
load hnormal5_fs200;load hnormal6_fs200;load hnormal7_fs200;load hnormal8_fs200;load hnormal9_fs200;
load hnormal10_fs200;load hnormal11_fs200;load hnormal12_fs200;load hnormal13c1_fs200;
load hnormal12L_fs200;
% Random Length fixing 
n1=normal1_fs200(1:288000); n2=normal2_fs200(1:282000); 
n3=hnormal1_fs200(1:144000);n4=hnormal2_fs200(1:90000);n5=hnormal3_fs200(1:166000);
n6=hnormal4_fs200(1:630000);
% data from 5 spt slice
n7=hnormal5_fs200(1:10000);n8=hnormal6_fs200(1:6000);n9=hnormal7_fs200(1:18000);
n10=hnormal8_fs200(1:1300);n11=hnormal9_fs200(1:24000);n12=hnormal10_fs200(1:4000);
n13=hnormal11_fs200(1:10000);n14=hnormal12_fs200(1:8000);
n12L=hnormal12L_fs200(1:120000);n13c1=hnormal13c1_fs200(1:308000);
% n_s1=seizure1_fs200(69001:95000);

% Seperating seizure for 30 seconds length
%%
% Framing of Random length signal
sg_len=2000;
for fl=1:144
    nfr1(:,fl) = n1((fl-1)*sg_len+1:sg_len*fl);% 10 second length
end
for fl=1:141
    nfr2(:,fl) = n2((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:72
    nfr3(:,fl) = n3((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:45
    nfr4(:,fl) = n4((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:83
    nfr5(:,fl) = n5((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:315
    nfr6(:,fl) = n6((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:5
    nfr7(:,fl) = n7((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:3
    nfr8(:,fl) = n8((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:9
    nfr9(:,fl) = n9((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:12
    nfr11(:,fl) = n11((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:2
    nfr12(:,fl) = n12((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:60
    nfr12L(:,fl) = n12L((fl-1)*sg_len+1:sg_len*fl);% 
end
for fl=1:154
    nfr13c1(:,fl) = n13c1((fl-1)*sg_len+1:sg_len*fl);% 
end
% for fl=1:13
%     ns1fr(:,fl) = n13c1((fl-1)*sg_len+1:sg_len*fl);% 
% end
combine_normal= horzcat(nfr1,nfr2,nfr3,nfr4,nfr5,nfr6,nfr7,nfr8,nfr9,nfr11,nfr12,nfr12L,nfr13c1);
%
% FFT Parameters
eeg_frame_length=length(combine_normal);
frame_NFFT = 2^nextpow2(eeg_frame_length);
frame_f = eeg_fs/2*linspace(0,1,frame_NFFT/2+1);
%% 
%%%%%%%%% Wavelet Decomposition For Normal Samples %%%%%%%%
close all
for n_number=1:1000  %95 % Number Of Nomral samples
[C,L] = wavedec(combine_normal(:,n_number),5,'db4');   
% [C,L] = wavedec(eeg_splitt(:,2),10,'db1');   
% figure(num_frames)
% reconstruction loop
for n_wlevel=4:5 % Wavelet deompocition level
    wavdecomp_normal(:,n_wlevel,n_number)=wrcoef('d',C,L,'db4',n_wlevel);
% Feature calculation

% 1- Relative Average AMPLITUDE - Peak to Peak
    normal_p2p(:,n_wlevel,n_number)= peak2peak(wavdecomp_normal(:,n_wlevel,n_number)); % peak to peak amplitude diff of frame
    normal_mean_p2p(:,n_wlevel,n_number)=mean(normal_p2p(:,n_wlevel,n_number));% mean peak to peak amplitude diff of frame
% 2- Co efficient of variation of Amplitudes
    normal_standard_deviation(:,n_wlevel,n_number) = std(wavdecomp_normal(:,n_wlevel,n_number)); % standard deviation
    normal_cva(:,n_wlevel,n_number)= normal_standard_deviation(:,n_wlevel,n_number)/normal_p2p(:,n_wlevel,n_number); % CVA

% 3- Relative derivative
    normal_fft(:,n_wlevel,n_number)=fft(wavdecomp_normal(:,n_wlevel,n_number),frame_NFFT)/eeg_frame_length; %applying fft
    normal_periodogram(:,n_wlevel,n_number)=periodogram(wavdecomp_normal(:,n_wlevel,n_number));      %Periodogram power spectral density estimate
    nomral_pwelch(:,n_wlevel,n_number)=pwelch(wavdecomp_normal(:,n_wlevel,n_number));                % Welch's spectral estimate
    normal_ARpsde(:,n_wlevel,n_number) = pyulear(wavdecomp_normal(:,n_wlevel,n_number),4);           % Autoregressive power spectral density estimate 
    normal_entropy(:,n_wlevel,n_number)=wentropy(wavdecomp_normal(:,n_wlevel,n_number),'shannon');   % Entropy (wavelet packet)
    
% ############RMS########    
    normal_rms(:,n_wlevel,n_number)=rms(wavdecomp_normal(:,n_wlevel,n_number));             % RMS

% ###############KURTOSIS######
    normal_kurtosis(:,n_wlevel,n_number)=kurtosis(wavdecomp_normal(:,n_wlevel,n_number));   % KURTOSIS

% ###############SKEWNWSS######
    normal_skewness(:,n_wlevel,n_number)=skewness(wavdecomp_normal(:,n_wlevel,n_number));   % KURTOSIS

% ###############MEDIAN FREQUENCY######
    normal_medianfreq(:,n_wlevel,n_number)=medfreq(wavdecomp_normal(:,n_wlevel,n_number));   % KURTOSIS
% ###############MEDIAN FREQUENCY######
    normal_meanfreq(:,n_wlevel,n_number)=meanfreq(wavdecomp_normal(:,n_wlevel,n_number));   % KURTOSIS

    % %     Plotting 10 level decomposition
figure()
%     subplot(4,1,1)
    subplot(3,1,1)

    plot(combine_normal(:,n_number))
    ed_str = sprintf('Non-Seizure Sample Number = %d',n_number);
    title(ed_str);
     xlabel('Time (seconds)')
    ylabel('Amplitude (mV)')
%     subplot(4,1,2)
    subplot(3,1,2)
    plot(wavdecomp_normal(:,n_wlevel,n_number))% plot each level signal
    ed_str = sprintf('Wavelet Decomposition level = %d, Seziure Number = %d',n_wlevel,n_number);
    title(ed_str);
     xlabel('Time (seconds)')
    ylabel('Amplitude')
%   title('EEG Frame Wavelet Decomposition')
%     subplot(4,1,3)
    subplot(3,1,3)
    % Plot single-sided amplitude spectrum.
    eeg_fft=normal_fft(:,n_wlevel,n_number); 
%     e=findpeaks(eeg_fft);
%     plot(e)
    plot(frame_f,2*abs(eeg_fft(1:frame_NFFT/2+1))) % Freq spectrum Plot
    title('Single-Sided Amplitude Spectrum')
    xlabel('Frequency (Hz)')
    ylabel('Amplitude')
    axis([0 100 0 0.03]);

    end
end
