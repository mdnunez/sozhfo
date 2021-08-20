function [count, whichchans, percent_removed, countperwin,thesefreqs, power, coherence] = hfo_timecourse_qHFO(varargin)
%HFO_TIMECOURSE_QHFO - Script that extracts the time course of HFO onsets and subtracts concurrent artifacts
%
%
% Copyright (C) 2021 Michael D. Nunez, <m.d.nunez@uva.nl>
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
%  10/28/20      Michael Nunez            Converted from hfo_timecourse1.m
%  10/29/20      Michael Nunez              Rethrow error
%  11/28/20      Michael Nunez             Relative home directory
%  08/20/21      Michael Nunez             Remove patient identifiers


%To do:  
%% Initial
[~,whichsub,whichchans,starttrack,endtrack,plotit,coherencewith]=parsevar(varargin,'whichsub','Patient2','whichchans',{'LTH1', 'LTH2', 'LTH3', 'LTH6'}, ...
	'starttrack',41,'endtrack',58,'plotit',1,'coherencewith',2);

%% Code
csvname = '%s_hfofiltdata%d.csv';
qhfocsv = '%s_falseHFOdata%d.csv';
secs = 0:3600; %3600 seconds per hour


HFOfile = sprintf('data/hfos/%s_qHFOcounts_total.mat',whichsub);
if exist(HFOfile)
	fprintf('Loading summary count data from %s ...\n',HFOfile);
	HFOdat = load(HFOfile);
	for c=1:numel(whichchans),
		count(:,c) = HFOdat.count((1+(starttrack-1)*3600):(3600*endtrack),strcmp(HFOdat.whichchans,whichchans{c})); 
	end
	numfiles = endtrack - starttrack + 1;
else
	tic;
	filetrack = starttrack;
	numfiles = 1;
	percent_removed = zeros(endtrack - starttrack + 1,length(whichchans));
	while (filetrack <= endtrack)
		hfotoload = sprintf(csvname,whichsub,filetrack);
		falsetoload = sprintf(qhfocsv,whichsub,filetrack);
		try
			fprintf('Loading file %s ...\n',hfotoload);
			[Channel_label, Start_time_secs, End_time_secs]  = csvimport(hfotoload, 'columns',{'Channel_label', 'Start_time_secs', 'End_time_secs'},'ignoreWSpace',1);
			fprintf('Loading file %s ...\n',falsetoload);
			[Channel_label_false, Start_time_secs_false, End_time_secs_false]  = csvimport(falsetoload, 'columns',{'Channel_label', 'Start_time_secs', 'End_time_secs'},'ignoreWSpace',1);
			%Remove white spaces in string array
			Channel_label = strrep(Channel_label,' ','');
			Channel_label_false = strrep(Channel_label_false,' ','');
			for c=1:numel(whichchans)
				hfoevents = strcmp(Channel_label, whichchans{c}); %Find HFO events in this channel
				origincount = sum(hfoevents);
				falseevents = strcmp(Channel_label_false, whichchans{c}) | strcmp(Channel_label_false,'AVGREF'); %Find FALSE HFO events (either a pop in this channel or in the average iEEG reference respectively)
				removed = 0;
				for e=1:length(hfoevents),
					if hfoevents(e) == 1,
						%Find where false candidates start after the start of the detected HFO and before the end of the detected HFO
						overlap1 = (Start_time_secs(e) <= Start_time_secs_false(falseevents)) & (End_time_secs(e) >= Start_time_secs_false(falseevents));
						%Find where false candidates end after the start of the detected HFO and before the end of the detected HFO
						overlap2 = (Start_time_secs(e) <= End_time_secs_false(falseevents)) & (End_time_secs(e) >= End_time_secs_false(falseevents));
						overlaps = overlap1 | overlap2;
						if any(overlaps),
							hfoevents(e) == 0;
							removed = removed + 1;
						end
					end
				end
				fprintf('%d false events out of %d detected HFO events removed for channel %s ! \n', removed, origincount, whichchans{c});
				percent_removed(numfiles,c) = removed/origincount; 
				for i=2:length(secs),
					tempcount(i-1) = sum((Start_time_secs(hfoevents) < secs(i)) & (Start_time_secs(hfoevents) > secs(i-1)));
				end
				count((1+(numfiles-1)*3600):(3600*numfiles),c) = tempcount;
				fprintf('Max %d, min %d, mean %.3f ripple-band HFOs found per second for channel %s in file %d \n',max(tempcount),min(tempcount),mean(tempcount),whichchans{c},filetrack);
			end
		catch me
			% fprintf('File %s not found! \n',hfotoload);
			rethrow(me);
			fprintf('Replacing counts with NaNs! \n');
			for c=1:numel(whichchans)
				count((1+(numfiles-1)*3600):(3600*numfiles),c) = NaN;
			end
		end
		filetrack = filetrack + 1;
		numfiles = numfiles + 1;
	end
	numfiles = numfiles -1;

	fprintf('Total loading time was %.3f minutes !\n',toc/60);
end

engineers = .4; %Note that Engineer's Nyquist frequency is 1 cycle per 2.5 windows (.4 cycles per window) 
countperwin = [];
windowsize = 30; %in seconds
fprintf('Calculating the count per 30 seconds of data...\n');
for c=1:size(count,2),
	countperwin(1,c) = sum(count(1:windowsize,c));
	for m=2:floor(size(count,1)/windowsize),
		countperwin(m,c) = sum(count(((1:windowsize) + windowsize*(m-1)),c));
	end
end

if plotit,
	E = size(count,2);
	T = size(count,1);
	allsecs = 1:(3600*numfiles);
	fprintf('Plotting the counts...\n');
	figure;
	cortplotxalt(allsecs,count,'custom',whichchans);
	set(gca,'Fontsize',18);
	xlabel('Seconds','Fontsize',18)
	ylabel('# HFOs per second','Fontsize',18);
	xlim([1 3600*numfiles]);

	fprintf('Plotting the counts per window of data...\n');
	figure;
	cortplotxalt(1:(floor(T/windowsize)),countperwin,'custom',whichchans);
	set(gca,'Fontsize',18);
	xlabel('Windows','Fontsize',18)
	ylabel('# HFOs per window','Fontsize',18);
	xlim([1 floor(T/windowsize)]);

	fprintf('Calculating the amplitude spectrum with 12 minute samples...\n');
	% meansubtract = count - ones(size(count,1),1)*mean(count,1);
	hilbcount = hilbert(count);
	inst_amplitude = abs(hilbcount); %Instantaneous amplitude from Hilbert transform
	meansubtract = hilbcount - ones(size(hilbcount,1),1)*mean(hilbcount,1);
	segmean = cattoseg(meansubtract,720);
	fourier = fft(segmean)/size(segmean,1);
	nsr = 1/size(segmean,1); %Nyquist sampling rate (Hz) 1 observation per second

	freqs = [0 engineers/windowsize]; %.4 cycles per window / (30 seconds per window) = 0.0133 cycles per second (Hz)
	plotfreqs = 0:nsr:freqs(2);
	[~,minindex] = min(abs(freqs(1)-plotfreqs));
	maxindex = length(plotfreqs);
	amplitude = mean(abs(fourier(minindex:maxindex,:,:)),3); %Amplitude spectrum
	power = mean(abs(fourier(minindex:maxindex,:,:)).^2,3); %Amplitude spectrum
	thesefreqs = plotfreqs(minindex:maxindex);

	figure;
	cortplotxalt(thesefreqs,power,'custom',whichchans,'Linewidth',5);
	set(gca,'Fontsize',24);
	xt = get(gca,'XTick');
	set(gca,'XTick', xt, 'XTickLabel', xt*60);
	xlabel('Cycles per minute (Hz*60)','Fontsize',24);
	ylabel('Power (count^2)','Fontsize',24);
	title('');

	coherence = zeros(length(minindex:maxindex),E,E);
	fprintf('Calculating the coherence spectrum with 10 minute windows...\n');
	for k = minindex:maxindex,
	   	tempcorr = corrcoef(transpose(squeeze(fourier(k,:,:))));
	    coherence(k,:,:) = abs(tempcorr).^2;
	end
	figure;
	cortplotxalt(thesefreqs,squeeze(coherence(:,coherencewith,:)),'custom',whichchans);
	set(gca,'Fontsize',18);
	xt = get(gca,'XTick');
	set(gca,'XTick', xt, 'XTickLabel', xt*60);
	xlabel('Cycles per minute (Hz*60)','Fontsize',18);
	ylabel('Coherence','Fontsize',18);
	title(sprintf('Coherence with electrode %d',coherencewith),'Fontsize',18);
else
	thesefreqs = [];
	amplitude = [];
	coherence = [];
end