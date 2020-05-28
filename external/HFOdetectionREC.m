function [candidate]=HFOdetectionREC(data,fs,A)
% Ver Nov 06 2017
% High frequency oscillation detector from "A Simple Statistical Method for
% the Automatic Detection of Ripples in Human Intracranial EEG," by Krit
% Charupanit and Beth Lopour (Brain Topography).
%
% Identify and measure peaks on rectified band-pass filtered data, 
% then use iterative method from Grange 2008 to  determine threshold. 
% Define HFOs as a number of consecutive peaks above threshold 
% Input data 
%   data : raw signal (CH,samples)
%   A : alpha value 
%   fs : sampling rate (Hz)
%   
% Output : candidate(ch, event_start_time, event_stop_time, alpha_value, threshold_value)

%% Parameter setting, and mode
% MODE
WIN=0;            % Epoching  the data - 0: using ALL data as single window, 1: seperate to small windows
 
% Parameters - alpha, #peak, number of iteration, inter-event itme gap
nCycles = 6;                   % number of peaks considered as one window ; Default 5 out of 6 peak
npeakth =5;                   % number of qualified peaks above threshold
niter = 16;                      % Number of iterations to fit the distribution
tstep=0.010;                 % minimum inter-event gap time of HFOs (s)

%%
nsam=size(data,2);            % number of samples (time points)

% Epoching  the data - 0: using ALL data as single window, 1: seperate to small windows
 if WIN==1
    winsize=60*fs;                % window size , instead of using the whole data,
                                                % data can spitted to multiple small windows (unit : sec)               
elseif WIN==0                                
    winsize=nsam;                % use whole data as 1 window : default setting
end

nwin=floor(nsam/winsize);       % number of window
nch = size(data,1) ;                       % number of channels      

% Band pass Filter for detecting ripples (default)
n1 = 700;                                              % filter order
f1 = [0 70 80 250 260 fs/2]/(fs/2);   % 80-250 Hz band pass
a1 = [0 0 1 1 0 0];                                % band pass filter
b1 = firls(n1, f1, a1);                            % band pass filter

R = filtfilt(b1, 1, data')';            % Filter the data
tVec = (1:nsam)/fs;                   % Time vector for plotting
Rabs = abs(R);                             % Rectify filtered data
peakValAll = cell(nch,1);           % To Collect Amp of every peaks in all channels 
peakTimesAll = cell(nch,1);      % To Collect Time of every peaks in all channels 

nn = zeros(nch,3000);              % index for Probability distribution function  (x axis)             
count1=1;                                     % to store CANDIDATE events
     
 for ii=1:nch

        % find peak process
          R2 = Rabs(ii,2:(end-1));          
          R3 = Rabs(ii,3:end);               
          R1 = Rabs(ii,1:(end-2));         
          peakInd = ((R2 > R3) & (R2 > R1));     % P2 Peak that higher than P1 and lower than P3 
          peakInd = logical([0 peakInd 0]);       % add zero start & end elements

          % result from find peaks >>>> peak amp, peak time
          peakVal = Rabs(ii, peakInd);              % peak amplitude values
          peakTimes = find(peakInd)/fs;          % time of the peaks (seconds)

          peakValAll{ii} = peakVal;                     % save all peak values (peakVal will be changed in the loop)
          peakTimesAll{ii} = peakTimes;            % save all peak Times (peakTime will be changed in the loop)
          
         for yy=1:size(A,2)         % loop for alpha
              alpha =A(yy) ;            % false detections rate (alpha)
             
             for kk=1:nwin             % loop for multiwindow (defualt: using whole data, nwin =1)
                  clear peakVal peakTimes

                  winstart=((kk-1)*winsize+1)/fs;        % window start time
                  winend=(kk*winsize)/fs;                     % window stop time
                  peakValtemp=peakValAll{ii};               % all peak value in ii channel within the window
                  peakTimestemp=peakTimesAll{ii};    % all peak time in ii channel within the window

                  % peak_target index, those are between window for default setting all recordings are considered as one window
                  Targetind=find(peakTimestemp<=winend & peakTimestemp>=winstart);   
                  peakVal=peakValtemp(Targetind);               % peaks value only in the kk window
                  peakTimes=peakTimestemp(Targetind);    % peaks time only in the kk window

                  detections = zeros(niter,length(peakVal));   % initialize matrix to store detections
                  kai = zeros(1,niter);                                             % threshold value in each iteration
                  nMax = max(peakValtemp)*3;                         % max value for distribution estimation
                  nn(ii,:)= linspace(0,nMax,3000);                     % x-values for distribution estimation     
                
                  P = zeros(niter,size(nn,2));                               % Initialize matrix for distribution probability
                 for jj=1:niter                % loop of iteration
                            % Assume underlying GAMMA distribution
                            [AA2,BB2] =gamfit(peakVal(peakVal>0));      % fit the peak amp distribution with Gamma Dist
                            % CDF of GAMMA Dist
                            P(jj,:) =   gamcdf(nn(ii,:),AA2(1),AA2(2));     % PDF
                            % Cumulative distribution of P
                            cumulative = 1-P(jj,:);     
                        
                            % Set threshold
                            alpha_ind = find(cumulative < alpha);       % CDF index that lower than set alpha
                            kai(jj) = nn(ii,min(alpha_ind));                     % kai >>> actual threshold value
                            % Identify detections
                            detections(jj,:) = (peakVal > kai(jj));           % qualified peaks

                            % Remove detections
                            peakVal(peakVal > kai(jj)) = 0;                    % removed qualified peaks that higher than TH for next iterations

                 end  
                                 
                  tempsum=sum(detections);                          % sum qualified peak from every iterations
                  inarow = conv(tempsum,ones(1,nCycles),'same');          % sum within moving window (number of peak > TH)
                  inarow(inarow<npeakth)=0;                         % keep only elements that higher than threshold
                  inarow2 = conv(inarow,ones(1,nCycles),'same');            
                                % convolution to get actual event duration in case more
                                % than one element that qualify stay together (more than 6 peaks) 
                  inarow2(inarow2~=0)=1;                                            % set all qualified element = 1 
                  flagtemp= bwlabel(inarow2);                  % flag candidated events
                  stcount=count1;
                           
                  for zz=1:max(flagtemp(:))
                      tempfind=find(flagtemp(:)==zz);             
                      temporary(count1,1:5)= [ii peakTimes(tempfind(1)) peakTimes(tempfind(end)) alpha  kai(end)];  
                      % (ch        event_start_time      event_stop_time       alpha_value       threshold_value)
                      count1=count1+1;
                  end
             end
             
             if max(flagtemp(:))<1       % channel with no detection set all parameters as zero
                  temporary(count1,1:5)=[0 0 0 0 0];
            end
            fncount=count1-1;
            
            temporary((stcount+1):fncount,6)= temporary((stcount+1):fncount,2)-temporary(stcount:(fncount-1),3);
                % column 6 is gaptime between two consecutive events
            temporary(stcount,6) =tstep+1;      % set first element more than duration TH
            tempindex=find(temporary(:,6)< tstep & temporary(:,6)>0);   % events that less than gapTH apart are marged
            temporary(:,6)=0;                             % set temporary column 6 to 0 
            temporary(tempindex,6)=1;          % Target event to merge set to 1 
            flag2= bwlabel(temporary(:,6));   % flag the column 6 - in case there are multiple consecutive events joint together 
            nflag=max(flag2);
            
            for zz=1:nflag              % merged all flaged events
                  tempfound2=find(flag2==zz);
                  temporary(tempfound2(1)-1,3)= temporary(tempfound2(end),3);
            end
         
             temporary(:,6)=[];                            % clear old gaptime to replace with new one
             temporary(tempindex,:)=[];          % clear old events after merged
             count1=size(temporary,1)+1;
        end           % Channels
    end                 % Alpha
    temporary(temporary(:,1)==0,:)=[];     % delete channels without detection
    candidate=temporary;                     % Final Result kept in candidate
    
end    
 