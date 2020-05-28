function [fail,subname,abesapath] = hfo_extractHFO(whichdir,starttrack,endtrack)
%HFO_extractHFO - Script that follows all steps to extract ripple-band HFOs and power spectrum from BESA files
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
%  10/02/18      Michael Nunez                   Original code
%  10/03/18      Michael Nunez               Fix saving line
%  01/17/19      Michael Nunez              Can load edf files
%  01/23/19      Michael Nunez     Remove white spaces in channel names from header

%% Initial
if isempty(whichdir),
	whichdir = pwd;
end
fprintf('Working from directory %s...\n',whichdir);
cd(whichdir);

edfflag = 0;
besas = rdir([whichdir,'/**/*.besa']);
csvs = rdir([whichdir,'/**/*.csv']);
if isempty(besas)
	edfflag = 1;
end

if ~edfflag,
	%Extract detailed ripple-band HFO information
	[fail,subname,abesapath] = hfo_besasteps;
else
	[fail,subname,abesapath] = hfo_edfsteps;
end

%Extract the power spectrum
hfo_besaspectrum('starttrack',starttrack,'endtrack',endtrack,'plotit',0);

%Extract summary HFO data and convert to counts per second
if ~edfflag,
	header = read_besa_besa(abesapath);
else
	header = read_edf(abesapath);
end
[count,whichchans] = hfo_timecourse1('whichsub',subname,'whichchans',strrep(header.label,' ',''),'starttrack',starttrack,'endtrack',endtrack,'plotit',0); 
saveloc = sprintf('/data/hfos/%s_HFOcounts_total.mat',subname);
fprintf('Saving count data at %s \n',saveloc);
save(saveloc,'count','whichchans');