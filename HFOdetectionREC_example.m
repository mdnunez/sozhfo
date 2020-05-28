function [channel,start_time_secs,end_time_secs,alpha_detec_param,theshold,armytimestart,armytimeend] = HFOdetectionREC_example(data,sample_rate)
%HFODETECTIONREC_EXAMPLE - Script that follows steps to extract HFOs
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

%Inputs:
% 	data: channel * sample data matrix
%   sample_rate: sample rate of the collected data     

%% Record of Revisions
%   Date           Programmers               Description of change
%   ====        =================            =====================
%  01/17/19      Michael Nunez                   Original code


alpha_param = .042; %Suggested parameter for iEEG data in Charupanit et al. (2017; Brain Topography)

%Detect HFOs using Charupanit algorithm
fprintf('Detecting HFOs in filtered data using Charupanit algorithm...\n');
fprintf('Using an alpha paramter of %.3f...\n', alpha_param);
% hfo_candidate=HFOdetectionREC(filtdata',header.Fs,alpha_param);
hfo_candidate=HFOdetectionREC(data,sample_rate,alpha_param);

hfo_candidate = sortrows(hfo_candidate,2); %Sort by onset time (secs)
hfo_candidate(:,6) = hfo_candidate(:,3) - hfo_candidate(:,2); %Duration in secs
hfo_candidate(:,7:8) = round(hfo_candidate(:,2:3)*sample_rate); %Convert to samples


%Calculate army time
armytimestart = toarmy(hfo_candidate(:,2));
armytimeend = toarmy(hfo_candidate(:,3));


channel = hfo_candidate(j,1);
start_time_secs = hfo_candidate(j,2);
end_time_secs = hfo_candidate(j,3);
alpha_detec_param = hfo_candidate(j,4);
theshold = hfo_candidate(j,5);
duration_secs = hfo_candidate(j,6),...
start_time_samps = hfo_candidate(j,7),...
end_time_samps = hfo_candidate(j,8));


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

