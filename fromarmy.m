function [secs] = fromarmy(armytime) %Converts army time to seconds 

% FROMARMY - Script that converts army time to seconds
%
% Copyright (C) 2017 Michael D. Nunez, <mdnunez1@uci.edu>
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
%  10/20/17      Michael Nunez                  Original code

%% Code
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
