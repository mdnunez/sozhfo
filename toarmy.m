function [armytime] = toarmy(secs) %Converts seconds to armytime

% TOARMY - Script that converts seconds to army time
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
