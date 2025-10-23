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
% load(fullfile(datadir,all_goodunit(session_num).name));
% load(fullfile(datadir,all_procdata(session_num).name));

% manually load in fname to match the python version
load(fullfile(datadir,'GoodUnit_240927_ZhuangZhuang_NSD1000_LOC_g2.mat'));
load(fullfile(datadir,'Processed_ses32_240927_M3_2.mat'));


%% some stats about this session
fprintf('total number of sites for session %d: %d\n', session_num, UnitNum);
fprintf('\t%d single units\n', sum(UnitType==1));
fprintf('\t%d MUAs\n', sum(UnitType==2));
fprintf('\t%d non-somatic units\n', sum(UnitType>2));

unit_num = 2;
raster = GoodUnitStrc(unit_num).Raster;

fprintf('\ntotal number of trials: %d\n\n', length(raster));

%% visualizations
% raster: [trials x timeBins] logical/double

% raster plot
[nTrials, nTime] = size(raster);
t = 1:nTime;                     % or your real time vector
[row, col] = find(raster);            % spike coordinates

figure; plot(t(col), row, '.', 'MarkerSize', 2);
set(gca, 'YDir','reverse');      % trial 1 at top
xlabel('Time (bins)'); ylabel('Trial'); title('Spike raster (dots)');
