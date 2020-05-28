function [fail,subject,abesapath] = hfo_besasteps(varargin)
%HFO_BESASTEPS - Script that follows the steps to extract HFOs from BESA files
%
% Copyright (C) 2018 Michael D. Nunez, <mdnunez1@uci.edu>
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
%  10/06/17      Michael Nunez                   Original code
%  10/12/17      Michael Nunez         Overlapping data loading
%  10/13/17      Michael Nunez           .csv save fixes
%  10/16/17      Michael Nunez       Change alpha parameter to .42
%                                 To match Charupanit & Lopour 2017
%              Remove double filtering, fix index of previous data
%  10/18/17      Michael Nunez      Find iEEG depth electrodes
%  10/20/17      Michael Nunez       Track army time
%  10/23/17      Michael Nunez      Better channel labels, notes
%  10/25/17      Michael Nunez   Import seizure data, track index
%  10/30/17      Michael Nunez       Skip seizures
%  10/31/17      Michael Nunez     Remove average ref only of groups
%                                   larger than 4 members
%  11/02/17      Michael Nunez    Fix loading future .besa files
%  11/06/17      Michael Nunez    Do not load nonexistant .besa files
%  11/08/17      Michael Nunez     Initialize fail vector
%  12/11/17      Michael Nunez     Incorporate localization csv tables
%  12/27/17      Michael Nunez       Save managable file sizes
%  01/04/18      Michael Nunez     Remove commas in channel locations
%  01/10/18      Michael Nunez       Change output file name
%  05/22/18      Michael Nunez    If the suggested reference is missing,
%                                  use the group average
%  06/22/18      Michael Nunez      Better failures
%  10/02/18      Michael Nunez    Output "subject" and "abesapath" for hfo_extractHFO
%  05/28/20      Michael Nunez             Remove patient identifiers

%To do:
%1) Export .mat file for modeling
%2) Add variables: Date, SOZ (seizure onset zone electrode), Propagation electrode
%3) Sleep stage the data?
%4) Incorporate artifact rejection

%% Initial
default_saveloc = 'data/hfos';
default_seizloc = 'data/hfos/seizure_data.csv';
[~,whichdir,saveloc,seizloc,alpha_param]=parsevar(varargin,'whichdir',pwd,'saveloc',default_saveloc,...
	'seizloc',default_seizloc,'alpha',.042);

%% Code
%https://www.mathworks.com/matlabcentral/fileexchange/32226-recursive-directory-listing-enhanced-rdir
besas = rdir([whichdir,'/**/*.besa']);
csvs = rdir([whichdir,'/**/*.csv']);

if isempty(besas)
	error('No .besa files found in directory %s\n',whichdir);
end

fprintf('Loading seizure data...\n');
seizdata = readtable(seizloc);
nrows = size(seizdata,1);

% fprintf('File load order:\n');
% for bnum = 1:length(besas)
%     fprintf('%s \n',besas(bnum).name);
% end

fail = zeros(1,length(besas));

tic;
besatrack = 0;
for bnum = 1:length(besas),
canloadbesa = 1;
try
	[header,fullpath,filestr,armytime] = loadbesa(besas,bnum);
	Fs = header.Fs;
	besatrack = str2num(filestr(14:15)); % Find file number
	currentsamp = 1;
	chunknum = 1;
	buffersamps = round(.2*Fs); %5 minute chunk with a 200 ms buffer
	endsamp = currentsamp + header.Fs*60*5 - 1 + buffersamps;
	read_besa_besa(fullpath,header,currentsamp,endsamp,1:header.nChans);
catch me
	fprintf('Unable to load because:\n');
	fprintf('%s\n',me.message);
	fail(bnum) = 1;
	canloadbesa = 0;
end
if canloadbesa,
	abesapath = fullpath;
	wheresub = regexp(besas(bnum).name,'Patient');
	subject = besas(bnum).name((wheresub(1)):(wheresub(1)+8));
	[splitname] = regexp(besas(bnum).name,'[.]','split');
	filestr = splitname{1};
	filestr = filestr((end-14):end);

	fprintf('Subject %s \n',subject);

	localization = [];
	for v=1:length(csvs),
		if ~isempty(regexp(csvs(v).name,sprintf('%s_localization.csv',subject)))
			fprintf('Loading known electrode locations from %s_localization.csv ...\n',subject);
			localization = readtable(csvs(v).name);
		end
	end

	if isempty(localization),
		%Find iEEG electrodes
		fprintf('Finding iEEG electrodes...\n');
		letters = regexp(header.label,'\D');
		numbers = regexp(header.label,'\d');
		likelyiEEG = zeros(1,length(header.label));
		changroup = zeros(1,length(header.label));
		chantype = cell(1,length(header.label));
		human = cell(1,length(header.label));
		reference = cell(1,length(header.label));
		[reference{1:length(header.label)}] = deal('Original');
		changrouplabels = cell(0); %Find channel groupings
		nchangroups  = 0; %Count channel groupings
		for c=1:length(header.label),
		    chantype{c} = header.label{c}(letters{c});
		    if ~any(strncmpi(changrouplabels,chantype{c},3)) %Ignore case, first 3 characters
		    	nchangroups = nchangroups+1;
		    	changrouplabels{nchangroups} = chantype{c};
		    	changroup(c) = nchangroups;
		    else
		    	wherematch = find(strncmpi(changrouplabels,chantype{c},3)); %Ignore case, first 3 characters
		    	changroup(c) = wherematch;
		    	chantype{c} = changrouplabels{wherematch};
		    end
		end

		%Categorize channels
		for c=1:length(changroup),
	    nsimchans = sum(changroup == changroup(c)); %find channels labels with same letters
	    if nsimchans >= 6 %number of electrodes in depth
	        likelyiEEG(c) = 1;
	    end
	    switch chantype{c} %Note that these are best guesses
		    case 'DC'
		    	human{c} = 'Analog inputs';
		    	likelyiEEG(c) = 0;
		    case 'AG'
		    	human{c} = 'Possible Anterior Grid';
		    	likelyiEEG(c) = 1;
		    case 'PG'
		    	human{c} = 'Possible Posterior Grid';
		    	likelyiEEG(c) = 1;
		    case 'LHH'
		    	human{c} = 'Possible Left Hippocampal Head';
		    	likelyiEEG(c) = 1;
		    case 'RHH'
		    	human{c} = 'Possible Right Hippocampal Head';
		    	likelyiEEG(c) = 1;
		    case 'RAC'
		    	human{c} = 'Possible Right Anterior Cingulate cortex';
		    	likelyiEEG(c) = 1;
		    case 'LAC'
		    	human{c} = 'Possible Left Anterior Cingulate cortex';
		    	likelyiEEG(c) = 1;
		    case 'RTH'
		    	human{c} = 'Possible Right Hippocampal Tail';
		    	likelyiEEG(c) = 1;
		    case 'LTH'
		    	human{c} = 'Possible Left Hippocampal Tail';
		    	likelyiEEG(c) = 1;
		    case 'LAM'
		    	human{c} = 'Possible Left Amygdala';
		    	likelyiEEG(c) = 1;
			case 'RAM'
		    	human{c} = 'Possible Right Amygdala';
		    	likelyiEEG(c) = 1;
			case 'RAH'
		    	human{c} = 'Possible Right Amygdala Hippocampal';
		    	likelyiEEG(c) = 1;
			case 'LAH'
		    	human{c} = 'Possible Left Amygdala Hippocampal';
		    	likelyiEEG(c) = 1;
			case 'ROC'
		    	human{c} = 'Possible Right Occipital cortex';
		    	likelyiEEG(c) = 1;
			case 'LOC'
		    	human{c} = 'Possible Left Occipital cortex';
		    	likelyiEEG(c) = 1;
			case 'ROF'
		    	human{c} = 'Possible Right Orbitofrontal cortex';
		    	likelyiEEG(c) = 1;
			case 'LOF'
		    	human{c} = 'Possible Left Orbitofrontal cortex';
		    	likelyiEEG(c) = 1;
		    case 'LPG'
		    	human{c} = 'Possible Left Parietal Grid';
		    	likelyiEEG(c) = 1;
			case 'RPG'
		    	human{c} = 'Possible Right Parietal Grid';
		    	likelyiEEG(c) = 1;
		    case 'LTG'
		    	human{c} = 'Possible Left Temporal Grid';
		    	likelyiEEG(c) = 1;
			case 'RTG'
		    	human{c} = 'Possible Right Temporal Grid';
		    	likelyiEEG(c) = 1;
		    case 'LIN'
		    	human{c} = 'Possible Left Insular cortex';
		    	likelyiEEG(c) = 1;
		    case 'RIN'
		    	human{c} = 'Possible Right Insular cortex';
		    	likelyiEEG(c) = 1;
		    case 'RSH'
		    	human{c} = 'Possible Right Superior Hippocampus?';
		    	likelyiEEG(c) = 1;
			case 'LSH'
		    	human{c} = 'Possible Left Superior Hippocampus?';
		    	likelyiEEG(c) = 1;
		    case 'RMF'
		    	human{c} = 'Possible Right Mesial Frontal';
		    	likelyiEEG(c) = 1;
		    case 'LMF'
		    	human{c} = 'Possible Left Mesial Frontal';
		    	likelyiEEG(c) = 1;
		    case 'LAD'
		    	human{c} = 'Possible Left Amygdala';
		    	likelyiEEG(c) = 1;
		    case 'RAD'
		    	human{c} = 'Possible Right Amygdala';
		    	likelyiEEG(c) = 1;
		    case 'MST'
		    	human{c} = 'Possible Medial Superior Temporal?';
		    	likelyiEEG(c) = 1;
		    case 'PTP'
		    	human{c} = 'Possible Parietal Temporal ???';
		    	likelyiEEG(c) = 1;
		    case 'G'
		    	human{c} = 'Possible Grid?';
		    	likelyiEEG(c) = 1;
		    case 'REF'
		    	human{c} = 'Reference electrode scalp? Forehead?';
		    	likelyiEEG(c) = 0;
		    case 'LLE'
		    	human{c} = '?';
		    	likelyiEEG(c) = 0;
		    case 'E'
		    	human{c} = 'Scalp electrode?';
		    	likelyiEEG(c) = 0;
		    case 'V'
		    	human{c} = '?';
		    	likelyiEEG(c) = 0;
		    case 'EKG'
		    	human{c} = 'Electrocardiogram';
		    	likelyiEEG(c) = 0;
		    otherwise
		    	human{c} = '?';
		    end
		end
	else
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

	while currentsamp < header.nSamples - buffersamps
		fprintf('Loading 5 minute chunk #%d...\n',chunknum);
		endsamp = currentsamp + header.Fs*60*5 - 1 + buffersamps;
		if endsamp >= header.nSamples & (bnum < length(besas))
			endsamp = header.nSamples;
			fprintf('Trying to load future data from next .besa ...\n');
			try
				[futureheader,futurepath,futurestr,futurearmy] = loadbesa(besas,bnum + 1);
				futureFs = futureheader.Fs;
				futuretrack = str2num(futurestr(14:15)); % Find file number
				if (Fs==futureFs) & (futuretrack == besatrack+1)
					fprintf('Loading future data from next .besa ...\n');
					[futuredata] = read_besa_besa(futurepath,futureheader,1,buffersamps,1:header.nChans);
				else
					fprintf('Next .besa is not sequential ...\n');
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
		[data] = read_besa_besa(fullpath,header,currentsamp,endsamp,1:header.nChans);
		data = [data futuredata];

		if isempty(localization),
			fprintf('Localization not found: \n');
			fprintf('Rereferencing each iEEG channel to the average of that group...\n')
			for g=1:nchangroups
				wheregroup = (changroup == g); %Find group channel locations
				if any(likelyiEEG(wheregroup)) & (sum(wheregroup) > 4) %If group is iEEG and number of electrodes is greater than 4
					avgref = mean(data(wheregroup,:),1);
					matref = ones(sum(wheregroup),1)*avgref; %matrix of values
					data(wheregroup,:) = data(wheregroup,:) - matref;
					[reference{wheregroup}] = deal('Average_of_Group');
				end
			end
		else
			fprintf('Localization found: \n');
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
		end

		% %Filter the data
		% fprintf('Filtering the data 80-500 Hz passband IIR...\n');
		% filtdata = filtereeg(data',header.Fs,[80 500],[70 510],60);

		%Detect HFOs using Charupanit algorithm
		fprintf('Detecting HFOs in filtered data using Charupanit algorithm...\n');
		fprintf('Using an alpha paramter of %.3f...\n', alpha_param);
		% hfo_candidate=HFOdetectionREC(filtdata',header.Fs,alpha_param);
		hfo_candidate=HFOdetectionREC(data,header.Fs,alpha_param);

		fprintf('Removing detected HFOs on the boundaries...\n');
		%Also remove data that starts in the end buffer so that it is not double recorded
		keepthese = find((hfo_candidate(:,2) < (endsamp-buffersamps)) & (hfo_candidate(:,2) ~= currentsamp));
		hfo_candidate = hfo_candidate(keepthese,:);

		hfo_candidate = sortrows(hfo_candidate,2); %Sort by onset time (secs)
		hfo_candidate(:,7:8) = hfo_candidate(:,2:3) + ...
		(currentsamp - 1)/header.Fs; % Convert to true seconds
		hfo_candidate(:,6) = hfo_candidate(:,8) - hfo_candidate(:,7); %Duration in secs
		hfo_candidate(:,2:3) = round(hfo_candidate(:,2:3)*header.Fs) + ...
		(currentsamp - 1); % Convert to true samples

		for n=1:nrows,
			%Fill in missing seizure onsets and offsets with 150 seconds
			if ~isnan(seizdata.Seizure_Samp_Start(n)) & isnan(seizdata.Seizure_Samp_End(n))
				seizdata.Seizure_Samp_End(n) = min([(seizdata.Seizure_Samp_Start(n) + 150*header.Fs) header.nSamples]);
			end
			if strcmp(filestr,seizdata.ExactFile_Start{n})
				fprintf('Removing detected HFOs during seizures...\n');
				%Remove detected HFOs if they start or stop in a seizure
				rmhfos1 = find((hfo_candidate(:,2) > seizdata.Seizure_Samp_Start(n)) & (hfo_candidate(:,2) < seizdata.Seizure_Samp_End(n)));
				rmhfos2 = find((hfo_candidate(:,3) > seizdata.Seizure_Samp_Start(n)) & (hfo_candidate(:,3) < seizdata.Seizure_Samp_End(n)));
				rmhfos4 = intersect(rmhfos1,rmhfos2);
				%Remove detected HFOs if they exist in two .besa files
				if ~strcmp(seizdata.ExactFile_Start{n},seizdata.ExactFile_End{n})
					rmhfos3 = find((hfo_candidate(:,2) > seizdata.Seizure_Samp_Start(n)));
				else
					rmhfos3 = [];
				end
				rmhfos = intersect(rmhfos4,rmhfos3);
				hfo_candidate(rmhfos,:) = [];
				fprintf('%d detected HFOs removed during seizures...\n',length(rmhfos));
			end
		end

		%Calculate army time
		armytimestart = toarmy(fromarmy(armytime) + hfo_candidate(:,7));
		armytimeend = toarmy(fromarmy(armytime) + hfo_candidate(:,8));

		cd(saveloc);
		hfocsvloc = sprintf('%s_hfofiltdata%d.csv',subject,besatrack);
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
		fprintf('Writing out table of Charupanit detected HFOs...\n');
		fileID = fopen(hfocsvloc,'a'); %Append to file
		for j=1:size(hfo_candidate,1),
			fprintf(fileID,'%s, %s, %s, %s, %s, %s, %s, %d, %d, %d, %6.4f, %6.4f, %.3f, %.2f, %.4f, %d, %d\n',...
				subject, ...
				filestr, ...
				header.label{hfo_candidate(j,1)},...
				human{hfo_candidate(j,1)},...
				armytimestart{j},...
				armytimeend{j},...
				reference{hfo_candidate(j,1)},...
				likelyiEEG(hfo_candidate(j,1)),...
				changroup(hfo_candidate(j,1)),...
				hfo_candidate(j,1),...
				hfo_candidate(j,7),...
				hfo_candidate(j,8),...
				hfo_candidate(j,4),...
				hfo_candidate(j,5),...
				hfo_candidate(j,6),...
				hfo_candidate(j,2),...
				hfo_candidate(j,3));
		end
		fclose(fileID);
		% dlmwrite(hfocsvloc, hfo_candidate,'-append','delimiter',',','precision',5);

		cd(whichdir);

		chunknum = chunknum +1;
		currentsamp = endsamp + 1 - buffersamps; %5 minute chunk with a 200 ms buffer

	end
end

fprintf('Finished! Total extraction took %d minutes!\n', round(toc/60));
end

function [header,fullpath,filestr,armytime] = loadbesa(besas,bnum)
	fullpath = besas(bnum).name;
	[splitname] = regexp(fullpath,'[.]','split');
	filestr = splitname{1};
	filestr = filestr((end-14):end);
	filestrs{bnum} = filestr;

	fprintf('Loading %s ...\n',fullpath);
	header = read_besa_besa(fullpath);
	minutes = header.nSamples/header.Fs/60;
	fprintf('This recording is %.2f minutes long with a sample rate of %d Hz\n',minutes,header.Fs);
	fprintf('There are %d electrode recordings in the data \n',header.nChans)
	armytime = header.orig.file_info.recording_date.start(9:14);
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

