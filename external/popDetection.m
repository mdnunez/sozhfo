function ARTrec = popDetection(data,fs)
%Based on universal automated high frequency oscillation detector for real-time, long term EEG
% Stephen V. Gliske, Zachary T. Irwin , Kathryn Davis, Kinshuk Sahaya , Cynthia Chestek ,William C. Stacey
% function  popDetection(data,fs)
%
%
% Artifect rejection of fast transient DC shift event 
% this detector identifies when the line-length (sum of the absolute value of the difference) 
% of a 0.05 s window of 850–990 Hz band pass filtered data.
% The mean and standard deviation are computed using a 5 s window previous to the window being evaluated.
% Band pass Filter for ripples (default)
% input is preprocessed-raw data
%
% Copyright (C) 2020 Michael D. Nunez, <mdnunez1@uci.edu>, Krit Charupanit, Casey Trevino
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Record of Revisions
%   Date           Programmers               Description of change
%   ====        =================            =====================
%  09/03/20      Michael Nunez            Adpated from code by C. Trevino and K. Charupanit

forder = 1500;                           % filter order
f1 = [0 840 850 990 1000 fs/2]/(fs/2);   % 850-990 Hz band pass
a1 = [0 0 1 1 0 0];                     %band pass filter           
b1 = firls(forder, f1, a1);                 %band pass filter
R = filtfilt(b1, 1, data')';           % Filter the data
%[H,f] = freqz(b1,a1,forder);
%plot(f,abs(H));
% fvtool(b1,1)      % plot the shape of the filter

nSD=4;
nsamples=size(R,2);                   % number of samples (time points)
nchannels=size(R,1);                      % number of channels

basewinsize=round(5*fs);              % baseline window size
testwinsize=round(0.1*fs);          % target window size
hfwin=round(basewinsize)/2;        % half of baseline window size 
nwin=floor((nsamples-basewinsize)/testwinsize);     % number of windows
count=1;
nrecalcwin = round(basewinsize/testwinsize); %Number of windows after which the 

for ii=1:nchannels            % CH loops
     for jj=1:nwin         % windows loop 
            SMstartind=basewinsize+(jj-1)*testwinsize+1;                  % test window start index at the baseline window size
            SMendind=SMstartind+testwinsize-1;                    % test window stop index
            SMmLinLeng=mean(abs(diff(R(ii,SMstartind:SMendind))));          % mean LineLength per data point
            if  mod(jj,nrecalcwin)==1                               % Recalculate baseline window every nrecalcwin loops
                BGstartind=SMstartind-basewinsize;                        % Baseline window start index
                BGendind=BGstartind+basewinsize-1;           % Baseline window stop index
                BGmLinLeng=mean(abs(diff(R(ii,BGstartind:BGendind))));            % mean LineLength per data point
                BGsdLinLeng=std(abs(diff(R(ii,BGstartind:BGendind))));                % SD LineLength 
                Thsd=BGmLinLeng+nSD*BGsdLinLeng;                  % threshold Value of Baseline window  MEAN+xSD
            end
            % when small test window have linelength higher than threshold will be considered as an artifact
            if SMmLinLeng>Thsd                                          
                ARTrec(count,1:3)=[ii,SMstartind,SMendind];     % KEEP ART  (channel , time_start, time_stop)
                count=count+1;
            end
      end
end

if ~exist('ARTrec','var'),
    ARTrec = [];
end