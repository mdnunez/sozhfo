function [plotpower, plotfreqs, plotcount, plotseiz, header] = hfo_spectrumHFO(varargin)
%HFO_SPECTRUMHFO - Script that plots power spectrum and HFO counts of full overnight recordings
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
%  03/15/18      Michael Nunez                   Original code
%  03/22/18      Michael Nunez   Fix time course, plot multiple channels
%  04/02/18      Michael Nunez            Fixing hour labels on plots
%  04/04/18      Michael Nunez         Plot seizure data as lines
%  04/10/18      Michael Nunez         Plot multiple HFO counts
%  04/16/18      Michael Nunez             Add legend to HFO count figure
%  05/17/18      Michael Nunez           Use of cortplotxalt
%  05/25/18      Michael Nunez           Remove army time
%  06/21/18      Michael Nunez            Plot slow-wave power versus rate
%  10/04/18      Michael Nunez           Create file start flag
%  12/05/18      Michael Nunez           Change x axis labels
%  01/10/19      Michael Nunez        Remove dependency on argmin()
%                               Remove seizure index on HFO count plots
%  01/18/19      Michael Nunez         Now works with edf files
%  01/23/19      Michael Nunez     Remove missing power data for .edf files
%  05/28/20      Michael Nunez             Remove patient identifiers

%% Notes:
% 1) Recording hour does not necessarily correspond to file number (some files < 60 minutes)

%% To do:
% 1) Plot multiple electrodes at once
% 2) Using FFT transform, might want to use Hanning windows or wavelets in the future


%% Code
[~,whichdir,plotit,spectrumchan,HFOchan,freqs,filetrack]=parsevar(varargin,'whichdir',pwd,'plotit',1,...
	'spectrumchan','LTH2','HFOchan',...
	{'LAM1','LAM2','LAM3','LHH1','LHH2','LHH3','LTH2','LTH3','LTH6','LTH1'},...
	'freqs',[1 20],'filetrack',2);

%% Code
edfflag = 0;
besas = rdir([whichdir,'/**/*.besa']);
csvs = rdir([whichdir,'/**/*.csv']);
if isempty(besas)
	fprintf('No .besa files found! Looking for .edf files... %s\n',whichdir);
	besas = rdir([whichdir,'/**/*.edf']);
	if isempty(besas)
		error('No .besa or .edf files found in directory %s\n',whichdir);
	end
	edfflag = 1;
end
fprintf('')


wheresub = regexp(besas(1).name,'Patient');
subject = besas(1).name((wheresub(1)):(wheresub(1)+8));
wheredate = regexp(besas(1).name,'_');
recordingdate = besas(1).name((wheredate-10):(wheredate-1));
filetrackstr = num2str(filetrack);

if ~edfflag,
	besaname = [recordingdate,'_',repmat('0',1,4-length(filetrackstr)),filetrackstr,'.besa'];
	fprintf('Loading header of first file: %s ...\n',besaname);
	header = read_besa_besa(besaname);
	minutes = header.nSamples/header.Fs/60;
	fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
	fprintf('There are %d electrode recordings in the data \n',header.nChans)
	hourstart = header.orig.file_info.recording_date.start(9:10);
	minstart = header.orig.file_info.recording_date.start(11:12);
	secstart = header.orig.file_info.recording_date.start(13:14);
	armystart = sprintf('%s:%s:%s',hourstart,minstart,minstart);
	fprintf('The start time was %s \n',armystart);
else
	edfname = [recordingdate,'_',repmat('0',1,4-length(filetrackstr)),filetrackstr,'.edf'];
	header = read_edf(edfname);
	header.Fs =header.samples(1)/header.duration;
	header.nChans = header.ns;
	minutes = header.records*header.duration/60;
	fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
	fprintf('There are %d electrode recordings in the data \n',header.ns)
	timenums = regexp(header.starttime,'[^.]','match'); %Remove periods
	gettime = cat(2,timenums{:});
	armystart = sprintf('%s:%s:%s',gettime(1:2),gettime(3:4),gettime(5:6));
	fprintf('The start time was %s \n',armystart);
end

powerspecfile = sprintf('%s_powertimecourse.mat',subject);
fprintf('Loading power spectrum data from file %s...\n',powerspecfile);
powerdat = load(powerspecfile);

chanlabels = header.label;
powerchan = find(strcmp(chanlabels,spectrumchan));
thesefreq = powerdat.plotfreqs;
[~, minfreqindx] = min(abs(thesefreq - freqs(1)));
[~, maxfreqindx] = min(abs(thesefreq - freqs(2)));
freqindx = minfreqindx:maxfreqindx;
plotfreqs = thesefreq(freqindx);
plotpower = squeeze(powerdat.power(:,freqindx,powerchan));
plotseiz = powerdat.seizure;

if edfflag,
    %Remove missing data from the power spectrum
    rmindex = isnan(plotpower(:,1)) | (plotpower(:,1) == 0);
    plotpower(rmindex,:) = [];
    powerdat.armytimes(rmindex) = [];
end

fivemin = 1:size(plotpower,1);
wherexlabel = find(mod(fivemin,36) == 0); %Every 3 hours
wherearmytimes = find(mod(1:length(powerdat.armytimes),60) == 0);
wherehours = 3:3:(3*length(wherexlabel));

onemin = 1:(size(plotpower,1)*5);
wherexlabel1 = find(mod(onemin,180) == 0); %Every 3 hours

fprintf('Plotting power spectrum for channel %s ... \n',spectrumchan);
powerfig = figure;
imagesc(plotpower');
yticks = get(gca,'YTick');
ylabs = plotfreqs(yticks);
set(gca,'YTick',yticks,'YTickLabel',ylabs);
% set(gca,'XTick',wherexlabel,'XTickLabel',powerdat.armytimes(wherearmytimes),...
% 	'Fontsize',16);
set(gca,'XTick',wherexlabel,'XTickLabel',wherehours,'Fontsize',16);
% xlabel('Army time','Fontsize',16);
xlabel('Recording Hour','Fontsize',16);
ylabel('Frequency','Fontsize',16);
h = colorbar;
ylabel(h, sprintf('Power of Channel %s',spectrumchan),'Fontsize',16);
seizindex = find(plotseiz==1);
for k=1:length(seizindex),
	temph = line([seizindex(k) seizindex(k)],get(gca,'YLim'));
	set(temph,'Color','r','LineWidth',3);
end

HFOfile = sprintf('/data/hfos/%s_HFOcounts_total.mat',subject);
fprintf('Loading HFO data from file %s...\n',HFOfile);
HFOdat = load(HFOfile);
%Counts per minute of data
for c=1:length(HFOchan)
	hfoelec = find(strcmp(HFOdat.whichchans,HFOchan{c}));
	for n=1:(size(plotpower,1)*5),
		speccount = HFOdat.count((1:60) + 60*(n-1),hfoelec);
		if all(isnan(speccount(:)))
			plotcount(n,c) = NaN;
		else
			plotcount(n,c) = nansum(speccount);
		end
	end
end
fprintf('Plotting HFO counts per 5 minutes of data ... \n');
countfig = figure;
cortplotxalt(onemin,plotcount,'Linewidth',1,'custom',HFOchan);
% set(gca,'XTick',wherexlabel1,'XTickLabel',powerdat.armytimes(wherearmytimes),...
% 	'Fontsize',16);
set(gca,'XTick',wherexlabel1,'XTickLabel',wherehours,'Fontsize',16,'YAxisLocation','right');
% seizindex = find(plotseiz==1);
% for k=1:length(seizindex),
% 	temph = line([seizindex(k)*5 seizindex(k)*5],get(gca,'YLim'));
% 	set(temph,'Color','r','LineWidth',3);
% end
xlabel('File number','Fontsize',16);
ylim([0 120]);
legend(HFOchan);
% hold on;
% scatter(onemin,plotcount,20,'b','x');
% hold off;


