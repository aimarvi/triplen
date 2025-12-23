clear
% new data directory for @aim
datadir = '../datasets/NNN/';

%% load in electrode data + metadata for a single sessions
% file names
all_goodunit = dir([datadir filesep 'GoodUnit_*']);
all_procdata = dir([datadir filesep 'Processed_ses*']);

% session number (01-59)
session_num = 1;

% actually load in the single-session data
load(fullfile(datadir,all_goodunit(session_num).name));
load(fullfile(datadir,all_procdata(session_num).name));