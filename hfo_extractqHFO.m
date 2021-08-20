function [fail,subname,abesapath] = hfo_extractqHFO(whichdir,starttrack,endtrack)
%HFO_extractqHFO - Script that follows all steps to extract ripple-band HFOs with artifact removed (qHFO) and power spectrum from BESA files
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
%  10/28/20      Michael Nunez              Converted from hfo_extractHFO
%  11/10/20      Michael Nunez          Remove power spectrum line since it has already been calcualted with hfo_extractHFO

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
	[fail,subname,abesapath] = hfo_besasteps_qHFO;
else
	[fail,subname,abesapath] = hfo_edfsteps_qHFO;
end

%Extract summary HFO data and convert to counts per second
if ~edfflag,
	header = read_besa_besa(abesapath);
else
	header = read_edf(abesapath);
end
[count,whichchans,percent_removed] = hfo_timecourse_qHFO('whichsub',subname,'whichchans',strrep(header.label,' ',''),'starttrack',starttrack,'endtrack',endtrack,'plotit',0); 
saveloc = sprintf('/data/hfos/%s_qHFOcounts_total.mat',subname);
fprintf('Saving count data at %s \n',saveloc);
save(saveloc,'count','whichchans','percent_removed');



