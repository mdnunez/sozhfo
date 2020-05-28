function matobj = hfo_besaspectrum(varargin)
%HFO_BESASPECTRUM - Script that extracts the power spectrum of full overnight recordings
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
%  03/13/18      Michael Nunez                   Original code
%  03/14/18      Michael Nunez                    Fixes
%  03/19/18      Michael Nunez          Save before every new file
%  03/22/18      Michael Nunez          Remove missing time periods
%  04/04/18      Michael Nunez         Load seizure data
%                                              exact time per chunk
%  04/11/18      Michael Nunez           Fix seizure load
%  05/25/18		 Michael Nunez             Load HFO file based on starttrack parameter
%  06/22/18      Michael Nunez              Fix army time NAN replacement
%  10/01/18      Michael Nunez               Rethrow file load error
%                                            Fix absence of "seiztrack"
%  10/05/18      Michael Nunez            Fix initialization indexing
%  01/17/19      Michael Nunez           Looks for edf files if no besa files found
%                                       Remove loading of HFO data, require localization file
%  05/28/20      Michael Nunez             Remove patient identifiers

%% Notes:
% 1) Note that this power output is not necessarily time sequential,
%    depending upon when the .besa files onset

%% To do:
% 1) Using FFT transform, might want to use Hanning windows or wavelets in the future


%% Initial
[~,whichdir,starttrack,endtrack,plotit,seizloc]=parsevar(varargin,'whichdir',pwd,'starttrack',1,'endtrack',69,'plotit',1,'seizloc',[]);

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


localization = [];
for v=1:length(csvs),
	if ~isempty(regexp(csvs(v).name,sprintf('%s_localization.csv',subject)))
		fprintf('Loading known electrode locations from %s_localization.csv ...\n',subject);
		localization = readtable(csvs(v).name);
	else
		error(sprintf('File %s_localization.csv not found!',subject));
	end
end

if isempty(seizloc)
	seizloc = sprintf('/data/seizure_data.csv');
end
if (exist(seizloc) == 2),
	fprintf('Seizure data found at %s \n',seizloc);
	seizdata = readtable(seizloc);
	seizindex = strcmp(seizdata.Subject,subject);
	seizfile = seizdata.ExactFile_Start(seizindex);
	seiztrack(1) = -1;
	for f=1:length(seizfile)
		if length(seizfile{f} > 0),
			seiztrack(f) = str2num(seizfile{f}(12:15));
		else
			seiztrack(f) = 0;
		end
	end
	seizsamps = seizdata.Seizure_Samp_Start(seizindex);
else
	fprintf('Seizure data NOT found at %s \n',seizloc);
end

savefile = sprintf('%s_powertimecourse.mat',subject);

broken = 0;
chunktrack = 1;
filetrack = starttrack;
while (filetrack <= endtrack)
	filetrackstr = num2str(filetrack);
	try
		%Load data and report time
		if edfflag
			besaname = [recordingdate,'_',repmat('0',1,4-length(filetrackstr)),filetrackstr,'.edf'];
			fprintf('Loading file %s ...\n',besaname);
			header = read_edf(besaname);
			header.Fs =header.samples(1)/header.duration;
			header.nChans = header.ns;
			minutes = header.records*header.duration/60;
			fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
			fprintf('There are %d electrode recordings in the data \n',header.ns)
			timenums = regexp(header.starttime,'[^.]','match'); %Remove periods
			gettime = cat(2,timenums{:});
			armystart = sprintf('%s:%s:%s',gettime(1:2),gettime(3:4),gettime(5:6));
			fprintf('The start time was %s \n',armystart);
		else
			besaname = [recordingdate,'_',repmat('0',1,4-length(filetrackstr)),filetrackstr,'.besa'];
			fprintf('Loading file %s ...\n',besaname);
			header = read_besa_besa(besaname);
			minutes = header.nSamples/header.Fs/60;
			fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
			fprintf('There are %d electrode recordings in the data \n',header.nChans)
			hourstart = header.orig.file_info.recording_date.start(9:10);
			minstart = header.orig.file_info.recording_date.start(11:12);
			secstart = header.orig.file_info.recording_date.start(13:14);
			armystart = sprintf('%s:%s:%s',hourstart,minstart,minstart);
			fprintf('The start time was %s \n',armystart);
		end
		if chunktrack == 1,
			if ~exist(savefile),
				matobj.power = nan(12*(endtrack-(starttrack-1)),139,header.nChans);
				matobj.seizure = zeros(12*(endtrack-(starttrack-1)),1);
			else
				fprintf('Loading file %s ...\n',savefile);
				matobj = load(savefile);
			end

			likelyiEEG = zeros(1,length(header.label));
			changroup = zeros(1,length(header.label));
			suggested_ref = zeros(1,length(header.label));
			chantype = cell(1,length(header.label));
			human = cell(1,length(header.label));
			reference = cell(1,length(header.label));
			[reference{1:length(header.label)}] = deal('Original');
			for c=1:length(header.label),
				wherechan = find(localization.Number == c);
				if ~isempty(wherechan)
					wherechan = wherechan(1);
					likelyiEEG(c) = localization.Lopour_iEEG_or_ECoG(wherechan);
					chantype{c} = strrep(localization.Channel_group{wherechan},',','');
					human{c} = strrep(localization.Knight_localization{wherechan},',','');
					suggested_ref(c) = localization.Lopour_Suggested_Reference(wherechan);
				else
					likelyiEEG(c) = 0;
					chantype{c} = 'NULL';
					human{c} = 'Unknown';
					suggested_ref(c) = 0;
				end
			end	
			[~,~,changroup] = unique(chantype);
			changroup = changroup';
			nchangroups = length(unique(changroup));
		end
		%Mark seizures
		whereseiz = (filetrack == seiztrack);
		seizchunks = floor(seizsamps(whereseiz)/header.Fs/60/5);
		if sum(whereseiz) > 0,
			fprintf('Seizure found in 5 minute chunk %d \n',seizchunks);
			matobj.seizure(seizchunks + (chunktrack-1)) = 1;
		end
		broken = 0;
	catch me
		fprintf([me.message,'\n']);
		% fprintf('File %s not found! \n',besaname);
		broken = 1;
		fprintf('Saving NaNs...\n');
		matobj.power((1:12) + (chunktrack-1),:,:) = NaN;
		chunktrack = chunktrack + 12;
		for m=1:12,
			matobj.armytimes{(m) + (chunktrack-1)} = NaN;
		end
	end
	if ~broken
		for chunk=1:ceil(minutes/5),
			matobj.armytimes((chunktrack-1) + chunk) = toarmy(fromarmy(armystart) + (chunk-1)*5*60);
			startsamp = header.Fs*60*5*(chunk-1) + 1;
			endsamp = header.Fs*60*5*chunk;
			fprintf('Loading successive 5 minute chunk #%d...\n',chunk);
			if edfflag,
				[~,data] = read_edf(besaname,startsamp,endsamp);
			else
				[data] = read_besa_besa(besaname,header,startsamp,endsamp,1:header.nChans);
			end
			
			for g=1:nchangroups
				wheregroup = (changroup == g); %Find group channel locations
				if any(likelyiEEG(wheregroup)) %If group is iEEG'
					reference_elecs = wheregroup & (suggested_ref);
					notreference_elecs = wheregroup & (~suggested_ref);
					grouplabs = chantype(wheregroup);
					grouplab = grouplabs{1};
					if any(reference_elecs)
						fprintf('Using suggested reference for %s electrode group ...\n',grouplab);
						avgref = mean(data(reference_elecs,:),1);
						matref = ones(sum(notreference_elecs),1)*avgref; %matrix of values
						data(notreference_elecs,:) = data(notreference_elecs,:) - matref;
						[reference{notreference_elecs}] = deal(sprintf('%s',header.label{reference_elecs}));
					elseif any(likelyiEEG(wheregroup)) & (sum(wheregroup) > 4)
						fprintf('Using average reference for %s electrode group ...\n',grouplab);
						avgref = mean(data(wheregroup,:),1);
						matref = ones(sum(wheregroup),1)*avgref; %matrix of values
						data(wheregroup,:) = data(wheregroup,:) - matref;
						[reference{wheregroup}] = deal('Average_of_Group');
					else
						fprintf('Not rereferencing electrodes in %s electrode group ...\n',grouplab);
					end
				end
			end

			%Filter the data
			fprintf('Filtering the data 1-70 Hz passband IIR...\n');
			filtdata = filtereeg(data',header.Fs,[1 70],[.25 80]);

			%Time-frequency transform, might want to use Hanning windows or wavelets in the future
			fprintf('Calculating the power spectrum...\n');
			segmean = cattoseg(filtdata,header.Fs*2); %Two second windows
			fourier = fft(segmean)/size(segmean,1);
			nsr = header.Fs/size(segmean,1); %Nyquist sampling rate (Hz), header.Fs observations per second
			
			freqs = [1 70];
			plotfreqs = 0:nsr:freqs(2);
			[~,minindex] = min(abs(freqs(1)-plotfreqs));
			maxindex = length(plotfreqs);
			power = mean(abs(fourier(minindex:maxindex,:,:)).^2,3)*(2/(nsr));; %Power spectrum
			thesefreqs = plotfreqs(minindex:maxindex);

			fprintf('Saving the power spectrum...\n');
			matobj.power((chunktrack-1) + chunk,:,:) = power;
			matobj.plotfreqs = thesefreqs;

			if plotit
				fprintf('Plotting the power spectrum...\n');
				figure(1);
				cortplotxalt(thesefreqs,power,'custom',header.label);
				set(gca,'Fontsize',18);
				xlabel('Frequency (Hz)','Fontsize',18);
				ylabel('Standardized Power (\muV^2/Hz)','Fontsize',18);
				drawnow;
			end
		end
		chunktrack = chunktrack + ceil(minutes/5);
	end
	filetrack = filetrack + 1;
	fprintf('Saving file %s ...\n',savefile);
	save(savefile,'-struct','matobj');
end