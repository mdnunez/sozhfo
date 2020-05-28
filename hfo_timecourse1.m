function [count, whichchans, countperwin,thesefreqs, power, coherence] = hfo_timecourse1(varargin)
%HFO_TIMECOURSE1 - Script that extracts the time course of HFO onsets
%
%
% Copyright (C) 2019 Michael D. Nunez, <mdnunez1@uci.edu>
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
%  02/06/18      Michael Nunez                   Original code
%  02/07/18      Michael Nunez                Plotting changes
%  02/13/18      Michael Nunez         Use of cortplox with labels
%  03/01/18      Michael Nunez             Plot and graph unsmoothed
%  03/12/18      Michael Nunez            Add holes to missing data
%  03/13/18      Michael Nunez            Quick fix for Patient2, file 40
%  03/14/18      Michael Nunez                  Fixes
%  04/03/18      Michael Nunez   Use of csvimport() instead of readtable()
%  04/04/18      Michael Nunez             Remove smoothing
%  04/09/18      Michael Nunez              Remove smoothing output, remove Quick fix
%  04/24/18      Michael Nunez            Changed defaults
%  04/26/18      Michael Nunez            Change plotting parameters, load prebuilt data
%  05/17/18      Michael Nunez          Use instantanous amplitude from Hilbert transform,
%                                           Plot power instead of amplitude
%  05/25/18      Michael Nunez        Change defaults
%  01/10/19      Michael Nunez        Update To Do list
%  05/28/20      Michael Nunez             Remove patient identifiers

% To do:
% 1) Recording hour does not necessarily correspond to file number (some files < 60 minutes)
%    This script outputs counts per file number but not real hour

%Notes:
%1) Bumps in power spectrum is an artifact of the smoothing function!!!
%2) Use Hilbert transform to get instantaneous amplitude before count frequency calculation (Thank Ramesh and Joachim)


%To do:  
%% Initial
[~,whichsub,whichchans,starttrack,endtrack,plotit,coherencewith]=parsevar(varargin,'whichsub','Patient2','whichchans',{'LTH1', 'LTH2', 'LTH3', 'LTH6'}, ...
	'starttrack',41,'endtrack',58,'plotit',1,'coherencewith',2);

%% Code
csvname = '%s_hfofiltdata%d.csv';
secs = 0:3600; %3600 seconds per hour

% tic;
% broken = 0;
% filetrack = starttrack;
% numfiles = 1;
% while ~broken & (filetrack <= endtrack)
% 	filetoload = sprintf(csvname,whichsub,filetrack);
% 	try
% 		fprintf('Loading file %s ...\n',filetoload);
% 		hfo_candidate = readtable(filetoload);
% 		for c=1:numel(whichchans)
% 			sozevents = strcmp(hfo_candidate.Channel_label, whichchans{c});
% 			for i=2:length(secs),
% 				tempcount(i-1) = sum((hfo_candidate.Start_time_secs(sozevents) < secs(i)) & (hfo_candidate.Start_time_secs(sozevents) > secs(i-1)));
% 			end
% 			count((1+(numfiles-1)*3600):(3600*numfiles),c) = tempcount;
% 			fprintf('Max %.3f min %.3f mean %.3f ripple-band HFOs found per second for channel %s in file %d \n',max(tempcount),min(tempcount),mean(tempcount),whichchans{c},filetrack);
% 		end
% 		filetrack = filetrack + 1;
% 		numfiles = numfiles + 1;
% 	catch
% 		fprintf('File %s not found! Finished loading! \n',filetoload);
% 		broken = 1;
% 		numfiles = numfiles - 1;
% 	end
% end
% if ~broken
% 	numfiles = numfiles -1;
% end
% fprintf('Total loading time was %.3f minutes !\n',toc/60);

HFOfile = sprintf('data/hfos/%s_HFOcounts_total.mat',whichsub);
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
	while (filetrack <= endtrack)
		filetoload = sprintf(csvname,whichsub,filetrack);
		try
			fprintf('Loading file %s ...\n',filetoload);
			[Channel_label, Start_time_secs]  = csvimport(filetoload, 'columns',{'Channel_label', 'Start_time_secs'},'ignoreWSpace',1);
			%Remove white spaces in string array
			Channel_label = strrep(Channel_label,' ','');
			for c=1:numel(whichchans)
				sozevents = strcmp(Channel_label, whichchans{c});
				for i=2:length(secs),
					tempcount(i-1) = sum((Start_time_secs(sozevents) < secs(i)) & (Start_time_secs(sozevents) > secs(i-1)));
				end
				count((1+(numfiles-1)*3600):(3600*numfiles),c) = tempcount;
				fprintf('Max %d, min %d, mean %.3f ripple-band HFOs found per second for channel %s in file %d \n',max(tempcount),min(tempcount),mean(tempcount),whichchans{c},filetrack);
			end
		catch me
			fprintf('File %s not found! \n',filetoload);
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