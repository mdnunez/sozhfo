function [hdr, data, wherefile] = read_edf(fname, begsample, endsample)
% Read European Data Format file into MATLAB
%
% [hdr, data] = read_edf(fname, begsample, endsample)
%         Reads data from ALL RECORDS of file fname ('*.edf'). Header
%         information is returned in structure hdr, and the signals
%         (waveforms) are returned in the data array
%
%
% FORMAT SPEC: Source: http://www.edfplus.info/specs/edf.html SEE ALSO:
% http://www.dpmi.tu-graz.ac.at/~schloegl/matlab/eeg/edf_spec.htm
%
% The first 256 bytes of the header record specify the version number of
% this format, local patient and recording identification, time information
% about the recording, the number of data records and finally the number of
% signals (ns) in each data record. Then for each signal another 256 bytes
% follow in the header record, each specifying the type of signal (e.g.
% EEG, body temperature, etc.), amplitude calibration and the number of
% samples in each data record (from which the sampling frequency can be
% derived since the duration of a data record is also known). In this way,
% the format allows for different gains and sampling frequencies for each
% signal. The header record contains 256 + (ns * 256) bytes.
%
% Following the header record, each of the subsequent data records contains
% 'duration' seconds of 'ns' signals, with each signal being represented by
% the specified (in the header) number of samples. In order to reduce data
% size and adapt to commonly used software for acquisition, processing and
% graphical display of polygraphic signals, each sample value is
% represented as a 2-byte integer in 2's complement format. Figure 1 shows
% the detailed format of each data record.
%
% DATA SOURCE: Signals of various types (including the sample signal used
% below) are available from PHYSIONET: http://www.physionet.org/
%
%
% % EXAMPLE 1:
% % Read header associated with file 'ecgca998.edf':
%
% [header] = read_edf('ecgca998.edf');
%
% % EXAMPLE 2:
% % Read all waveforms/data between sample 1 and 5000:
%
% [header, data] = read_edf('ecgca998.edf',1,5000);
%
%
% Coded 8/27/09 by Brett Shoelson, PhD
% brett.shoelson@mathworks.com
% Modified from edfread on 01/24/19 by Michael D. Nunez, PhD

% HEADER RECORD
% 8 ascii : version of this data format (0)
% 80 ascii : local patient identification
% 80 ascii : local recording identification
% 8 ascii : startdate of recording (dd.mm.yy)
% 8 ascii : starttime of recording (hh.mm.ss)
% 8 ascii : number of bytes in header record
% 44 ascii : reserved
% 8 ascii : number of data records (-1 if unknown)
% 8 ascii : duration of a data record, in seconds
% 4 ascii : number of signals (ns) in data record
% ns * 16 ascii : ns * label (e.g. EEG FpzCz or Body temp)
% ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
% ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
% ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
% ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
% ns * 8 ascii : ns * digital minimum (e.g. -2048)
% ns * 8 ascii : ns * digital maximum (e.g. 2047)
% ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
% ns * 8 ascii : ns * nr of samples in each data record
% ns * 32 ascii : ns * reserved

% DATA RECORD
% nr of samples[1] * integer : first signal in the data record
% nr of samples[2] * integer : second signal
% ..
% ..
% nr of samples[ns] * integer : last signal

if nargin > 3
    error('EDFREAD: Too many input arguments.');
end

if ~nargin
    error('EDFREAD: Requires at least one input argument (filename to read).');
end

[fid,msg] = fopen(fname,'r');
if fid == -1
    error(msg)
end

% HEADER
hdr.ver        = str2double(char(fread(fid,8)'));
hdr.patientID  = fread(fid,80,'*char')';
hdr.recordID   = fread(fid,80,'*char')';
hdr.startdate  = fread(fid,8,'*char')';% (dd.mm.yy)
% hdr.startdate  = datestr(datenum(fread(fid,8,'*char')','dd.mm.yy'), 29); %'yyyy-mm-dd' (ISO 8601)
hdr.starttime  = fread(fid,8,'*char')';% (hh.mm.ss)
% hdr.starttime  = datestr(datenum(fread(fid,8,'*char')','hh.mm.ss'), 13); %'HH:MM:SS' (ISO 8601)
hdr.bytes      = str2double(fread(fid,8,'*char')');
reserved       = fread(fid,44);
hdr.records    = str2double(fread(fid,8,'*char')');
hdr.duration   = str2double(fread(fid,8,'*char')');
% Number of signals
hdr.ns    = str2double(fread(fid,4,'*char')');
for ii = 1:hdr.ns
    hdr.label{ii} = fread(fid,16,'*char')';
end
for ii = 1:hdr.ns
    hdr.transducer{ii} = fread(fid,80,'*char')';
end
% Physical dimension
for ii = 1:hdr.ns
    hdr.units{ii} = fread(fid,8,'*char')';
end
% Physical minimum
for ii = 1:hdr.ns
    hdr.physicalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Physical maximum
for ii = 1:hdr.ns
    hdr.physicalMax(ii) = str2double(fread(fid,8,'*char')');
end
% Digital minimum
for ii = 1:hdr.ns
    hdr.digitalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Digital maximum
for ii = 1:hdr.ns
    hdr.digitalMax(ii) = str2double(fread(fid,8,'*char')');
end
for ii = 1:hdr.ns
    hdr.prefilter{ii} = fread(fid,80,'*char')';
end
for ii = 1:hdr.ns
    hdr.samples(ii) = str2double(fread(fid,8,'*char')');
end
for ii = 1:hdr.ns
    reserved    = fread(fid,32,'*char')';
end
hdr.label = deblank(hdr.label);
hdr.units = deblank(hdr.units);

wherefile(1) = ftell(fid);
if nargout > 1
    if nargin < 3,
        endsample = hdr.records*hdr.samples(1);
        if nargin < 2,
            begsample = 1;
        end
    end

    % Scale data (linear scaling)
    scalefac = (hdr.physicalMax - hdr.physicalMin)./(hdr.digitalMax - hdr.digitalMin);
    dc = hdr.physicalMax - scalefac .* hdr.digitalMax;

    % Skip to beginning of required data
    skiprecs = floor(begsample/hdr.samples(1));
    truebegsample = skiprecs*hdr.samples(1) + 1;
    fseek(fid,wherefile(1)+skiprecs*hdr.ns*hdr.samples(1)*2,'bof');

     %Initialize channels by samples array
    data = zeros(hdr.ns, ceil(endsample/hdr.samples(1))*hdr.samples(1)-truebegsample+1);

    % Obtain data
    recnum = skiprecs;
    sampletrack =1;
    while ( ( endsample >= recnum*hdr.samples(1) ) & (recnum <hdr.records))
        recnum = recnum + 1;
        for ii = 1:hdr.ns
            % Use a cell array for DATA because number of samples may vary
            % from sample to sample
            data(ii, sampletrack:(sampletrack + hdr.samples(1)-1) ) = fread(fid,hdr.samples(ii),'int16') * scalefac(ii) + dc(ii);
            wherefile = [wherefile ftell(fid)];
        end
        sampletrack = sampletrack + hdr.samples(1);
    end
    if recnum==hdr.records,
        endsample=hdr.records*hdr.samples(1);
        fprintf('Reached the end of the file at sample %d ! \nReturning all data found before this sample...\n',endsample);
    end

    %Return only the requested samples
    data(:,1:(begsample - truebegsample)) = [];
    data(:,(endsample-begsample+2):end) =[]; %remove false end samples
end
fclose(fid);

