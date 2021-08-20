function [fail,subject,edfpath] = hfo_edfsteps_qHFO(varargin)
%HFO_EDFSTEPS_QHFO - Script that follows the steps to extract HFOs from EDF files
%
% Copyright (C) 2020 Michael D. Nunez, <mdnunez1@uci.edu>
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
%  11/16/20      Michael Nunez    Adapted from hfo_edfsteps.m and hfo_besasteps_qHFO.m
%  11/18/20      Michael Nunez             Replace home folder with general folder
%  08/20/21      Michael Nunez             Remove patient identifiers

%% Initial
default_saveloc = '/data/hfos';
default_seizloc = '/data/hfos/seizure_data.csv';
[~,whichdir,saveloc,seizloc,alpha_param]=parsevar(varargin,'whichdir',pwd,'saveloc',default_saveloc,...
	'seizloc',default_seizloc,'alpha',.042);

%% Code
%https://www.mathworks.com/matlabcentral/fileexchange/32226-recursive-directory-listing-enhanced-rdir
edfs = rdir([whichdir,'/**/*.edf']);
csvs = rdir([whichdir,'/**/*.csv']);

if isempty(edfs)
	error('No .edf files found in directory %s\n',whichdir);
end

fprintf('Loading seizure data...\n');
seizdata = readtable(seizloc);
nrows = size(seizdata,1);

% fprintf('File load order:\n');
% for bnum = 1:length(edfs)
%     fprintf('%s \n',edfs(bnum).name);
% end

fail = zeros(1,length(edfs));

tic;
edftrack = 0;
for bnum = 1:length(edfs),
canloadedf = 1;
try
	[header,fullpath,filestr,armytime] = loadedf(edfs,bnum);
	Fs = header.samples(1)/header.duration;
	edftrack = str2num(filestr(14:15)); % Find file number
	currentsamp = 1;
	chunknum = 1;
	buffersamps = round(.2*Fs); %5 minute chunk with a 200 ms buffer
	endsamp = currentsamp + header.Fs*60*5 - 1 + buffersamps;
	read_edf(fullpath,[currentsamp endsamp]);
catch me
	fprintf('Unable to load because:\n');
	fprintf('%s\n',me.message);
	fail(bnum) = 1;
	canloadedf = 0;
end
if canloadedf,
	edfpath = fullpath;
	wheresub = regexp(edfs(bnum).name,'Patient');
	subject = edfs(bnum).name((wheresub(1)):(wheresub(1)+3));
	[splitname] = regexp(edfs(bnum).name,'[.]','split');
	filestr = splitname{1};
	filestr = filestr((end-14):end);

	fprintf('Subject %s \n',subject);

	localization = [];
	for v=1:length(csvs),
		if ~isempty(regexp(csvs(v).name,sprintf('%s_localization.csv',subject)))
			fprintf('Loading known electrode locations from %s_localization.csv ...\n',subject);
			localization = readtable(csvs(v).name);
		else
			error(sprintf('File %s_localization.csv not found!',subject));
		end
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
	nChans = header.ns;
	header.label{nChans+1} = 'AVGREF';
	human{nChans+1} = 'CommonAverage_iEEG';
	reference{nChans+1} = 'Self';
	likelyiEEG(nChans+1) = 1;
	changroup(nChans+1) = max(changroup) + 1;


	while currentsamp < header.nSamples - buffersamps
		fprintf('Loading 5 minute chunk #%d...\n',chunknum);
		endsamp = currentsamp + header.Fs*60*5 - 1 + buffersamps;
		if endsamp >= header.nSamples & (bnum < length(edfs))
			endsamp = header.nSamples;
			fprintf('Trying to load future data from next .edf ...\n');
			try
				[futureheader,futurepath,futurestr,futurearmy] = loadedf(edfs,bnum + 1);
				futureFs = futureheader.Fs;
				futuretrack = str2num(futurestr(14:15)); % Find file number
				if (Fs==futureFs) & (futuretrack == edftrack+1)
					fprintf('Loading future data from next .edf ...\n');
					[~,futuredata] = read_edf(futurepath,1,buffersamps);
				else
					fprintf('Next .edf is not sequential ...\n');
					futuredata = [];
				end
			catch me
				% rethrow(me);
				fprintf('\n');
				fprintf('Unable to load because:\n');
				fprintf('%s\n',me.message);
				fail(bnum+1) = 1;
				futuredata = [];
			end
		else
			futuredata = [];
		end
		[~,data] = read_edf(fullpath,currentsamp,endsamp);
		data = [data futuredata];

		fprintf('Calculating average of all iEEG channels for FALSE HFO artifact correction...\n')
		avgref = mean(data(logical(likelyiEEG(1:nChans)),:),1);

		%Detect false HFOs using Charupanit algorithm
		fprintf('Detecting FALSE HFOs in common average of iEEG data using Charupanit algorithm...\n');
		fprintf('Using an alpha paramter of %.3f...\n', alpha_param);
		false_candidate1=HFOdetectionREC(avgref,header.Fs,alpha_param);
		false_candidate1(:,1) = nChans+1;

		%Detect false HFOs using popDetection()
		fprintf('Detecting pops in 850-990 band pass data...\n');
		false_candidate2=popDetection(data,header.Fs);
		if ~isempty(false_candidate2),
			false_candidate2(:,2:3) = false_candidate2(:,2:3)/header.Fs;
			false_candidate2 = [false_candidate2 zeros(size(false_candidate2,1),2)];
		end

		false_candidate = [false_candidate1; false_candidate2];

		fprintf('Removing detected FALSE HFOs on the boundaries...\n');
		%Also remove data that starts in the end buffer so that it is not double recorded
		keepthese = find((false_candidate(:,2) < (endsamp-buffersamps)) & (false_candidate(:,2) ~= currentsamp));
		false_candidate = false_candidate(keepthese,:);

		false_candidate = sortrows(false_candidate,2); %Sort by onset time (secs)
		false_candidate(:,7:8) = false_candidate(:,2:3) + ...
		(currentsamp - 1)/header.Fs; % Convert to true seconds
		false_candidate(:,6) = false_candidate(:,8) - false_candidate(:,7); %Duration in secs
		false_candidate(:,2:3) = round(false_candidate(:,2:3)*header.Fs) + ...
		(currentsamp - 1); % Convert to true samples

		for n=1:nrows,
			%Fill in missing seizure onsets and offsets with 150 seconds
			if ~isnan(seizdata.Seizure_Samp_Start(n)) & isnan(seizdata.Seizure_Samp_End(n))
				seizdata.Seizure_Samp_End(n) = min([(seizdata.Seizure_Samp_Start(n) + 150*header.Fs) header.nSamples]);
			end
			if strcmp(filestr,seizdata.ExactFile_Start{n})
				fprintf('Removing detected FALSE HFOs during seizures...\n');
				%Remove detected HFOs if they start or stop in a seizure
				rmhfos1 = find((false_candidate(:,2) > seizdata.Seizure_Samp_Start(n)) & (false_candidate(:,2) < seizdata.Seizure_Samp_End(n)));
				rmhfos2 = find((false_candidate(:,3) > seizdata.Seizure_Samp_Start(n)) & (false_candidate(:,3) < seizdata.Seizure_Samp_End(n)));
				rmhfos4 = intersect(rmhfos1,rmhfos2);
				%Remove detected HFOs if they exist in two .edf files
				if ~strcmp(seizdata.ExactFile_Start{n},seizdata.ExactFile_End{n})
					rmhfos3 = find((false_candidate(:,2) > seizdata.Seizure_Samp_Start(n)));
				else
					rmhfos3 = [];
				end
				rmhfos = intersect(rmhfos4,rmhfos3);
				false_candidate(rmhfos,:) = [];
				fprintf('%d detected FALSE HFOs removed during seizures...\n',length(rmhfos));
			end
		end

		cd(saveloc);
		hfocsvloc = sprintf('%s_falseHFOdata%d.csv',subject,edftrack);
		if exist(hfocsvloc) ~= 2,
			fprintf('Creating file %s at %s...\n',hfocsvloc,saveloc);
			fileID = fopen(hfocsvloc,'w');
			fprintf(fileID,'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n',...
				'Subject', ...
				'Besafile', ...
				'Channel_label', ...
				'Human_readable_label', ...
				'Start_armytime', ...
				'End_armytime', ...
				'Reference', ...
				'likely_iEEG', ...
				'Channel_group',...
				'Channel', ...
				'Start_time_secs', ...
				'End_time_secs', ...
				'Alpha_detec_param', ...
				'Threshold', ...
				'Duration_secs',...
				'Start_time_samps', ...
				'End_time_samps');
			fclose(fileID);
		end

		
		if ~isempty(false_candidate),
			%Calculate army time
			armytimestart = toarmy(fromarmy(armytime) + false_candidate(:,7));
			armytimeend = toarmy(fromarmy(armytime) + false_candidate(:,8));
			fprintf('Writing out table of Charupanit detected HFOs...\n');
			fileID = fopen(hfocsvloc,'a'); %Append to file
			for j=1:size(false_candidate,1),
				fprintf(fileID,'%s, %s, %s, %s, %s, %s, %s, %d, %d, %d, %6.4f, %6.4f, %.3f, %.2f, %.4f, %d, %d\n',...
					subject, ...
					filestr, ...
					header.label{false_candidate(j,1)},...
					human{false_candidate(j,1)},...
					armytimestart{j},...
					armytimeend{j},...
					reference{false_candidate(j,1)},...
					likelyiEEG(false_candidate(j,1)),...
					changroup(false_candidate(j,1)),...
					false_candidate(j,1),...
					false_candidate(j,7),...
					false_candidate(j,8),...
					false_candidate(j,4),...
					false_candidate(j,5),...
					false_candidate(j,6),...
					false_candidate(j,2),...
					false_candidate(j,3));
			end
			fclose(fileID);
			% dlmwrite(hfocsvloc, false_candidate,'-append','delimiter',',','precision',5);
		end

		cd(whichdir);

		chunknum = chunknum +1;
		currentsamp = endsamp + 1 - buffersamps; %5 minute chunk with a 200 ms buffer

	end
end

fprintf('Finished! Total extraction took %d minutes!\n', round(toc/60));
end

function [header,fullpath,filestr,armytime] = loadedf(edfs,bnum)
	fullpath = edfs(bnum).name;
	[splitname] = regexp(fullpath,'[.]','split');
	filestr = splitname{1};
	filestr = filestr((end-14):end);
	filestrs{bnum} = filestr;

	fprintf('Loading %s ...\n',fullpath);
	header = read_edf(fullpath);
	header.Fs = header.samples(1)/header.duration;
	header.nSamples = header.records*header.samples(1);
	minutes = header.records*header.duration/60;
	fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
	fprintf('There are %d electrode recordings in the data \n',header.ns)
	timenums = regexp(header.starttime,'[^.]','match'); %Remove periods
	armytime = cat(2,timenums{:});
	fprintf('The start time was %s:%s:%s\n',armytime(1:2),armytime(3:4),armytime(5:6));

function [armytime] = toarmy(secs) %Converts seconds to armytime
	armyhours = mod(floor(secs/60/60), 24); %24 hour
	armymins = mod(floor(secs/60) ,60); %60 minutes
	armysecs = mod(round(secs), 60); %60 seconds (rounded seconds since no milliseconds)
	for j=1:length(secs),
		strhours{j} = num2str(armyhours(j));
		if numel(strhours{j}) < 2,
			strhours{j} = ['0' strhours{j}];
		end
		strmins{j} = num2str(armymins(j));
		if numel(strmins{j}) < 2,
			strmins{j} = ['0' strmins{j}];
		end
		strsecs{j} = num2str(armysecs(j));
		if numel(strsecs{j}) < 2,
			strsecs{j} = ['0' strsecs{j}];
		end
		armytime{j} = [strhours{j} strmins{j} strsecs{j}];
	end

function [secs] = fromarmy(armytime) %Converts armytime to secs
	if iscell(armytime)
		for j=1:length(armytime)
			findarmy = regexp(armytime{j},'[^:]','match'); %Remove colons
			armytime{j} = cat(2,findarmy{:}); %Concatenate findarmy cell strings together
			tempsecs = str2num(armytime{j}(1:2))*60*60 + str2num(armytime{j}(3:4))*60 + str2num(armytime{j}(5:6));
			if isempty(tempsecs),
				tempsecs = NaN;
			end
			secs(j) = tempsecs;
		end
	else
		findarmy = regexp(armytime,'[^:]','match'); %Remove colons
		armytime = cat(2,findarmy{:}); %Concatenate findarmy cell strings together
		tempsecs = str2num(armytime(1:2))*60*60 + str2num(armytime(3:4))*60 + str2num(armytime(5:6));
		if isempty(tempsecs),
			tempsecs = NaN;
		end
		secs = tempsecs;
	end

