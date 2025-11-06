figure;
imagesc(response_basic);
colormap('hot'); % or 'parula', 'jet', etc.
colorbar;
xlabel('Image index');
ylabel('Unit index');
title('Averaged firing rates across units and images');


[coeff_img, score_img, ~] = pca(response_basic');
figure;
scatter(score_img(:,1), score_img(:,2), 20, 'filled');
xlabel('PC1'); ylabel('PC2');
title('PCA of image representations');

clustermap = clustergram(response_basic, 'Colormap', parula, 'Standardize', 'none');

%% how exactly are units defined?
% only sites (394 total) above noise record the voltage
% some algorithm to determine what signals are coming from the same
% 'neuron/unit'

% visualize recordings at each channel/site
% Params
uids = 1:50;                         % pick which units to show
nUnits = numel(uids);
nRows = 5; nCols = ceil(nUnits/nRows);  % layout (tweak as you like)

% Robust global CLim so colors are comparable across units
allVals = [];
for k = 1:nUnits
    wf = GoodUnitStrc(uids(k)).waveform;      % (nCh x nSamples)
    allVals = [allVals; wf(:)];
end
lo = prctile(allVals, 1);
hi = prctile(allVals, 99);

% Plot tiled heatmaps
figure('Color','w');
t = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');
for k = 1:nUnits
    wf = GoodUnitStrc(uids(k)).waveform;      % (nCh x nSamples)
    nexttile;
    imagesc(wf, [lo hi]);
    axis tight ij;                            % 'ij' puts channel 1 at top (optional)
    title(sprintf('U%d (%dx%d)', uids(k), size(wf,1), size(wf,2)), 'FontWeight','normal');
    set(gca, 'XTick',[], 'YTick',[]);
end
colormap(parula);
cb = colorbar('eastoutside'); 
cb.Label.String = 'Amplitude (µV)';          % adjust units to your dataset
title(t, 'Spike Waveforms – Heatmaps');
xlabel(t, 'Time samples'); ylabel(t, 'Channels');

%%
% Helper to choose a good offset per tile (handles different amplitudes)
offsetFactor = 1.3;  % spacing between channels (increase if traces overlap)

figure('Color','w');
t = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');
for k = 1:nUnits
    wf = GoodUnitStrc(uids(k)).waveform;          % (nCh x nSamples)
    [nCh, nSamp] = size(wf);
    % Scale/offset so traces don't overlap
    pr = prctile(wf(:), [1 99]);
    perange = max(1e-12, pr(2)-pr(1));            % avoid zero range
    ystep = perange * offsetFactor;

    nexttile; hold on;
    for ch = 1:nCh
        plot(wf(ch,:) + (ch-1)*ystep, 'k');       % one color; adjust if you want
    end
    % Mark the "main" channel (max absolute amplitude)
    [~, mainCh] = max(max(abs(wf), [], 2));
    plot(wf(mainCh,:) + (mainCh-1)*ystep, 'LineWidth', 1.5);  % emphasize main channel

    xlim([1 nSamp]);
    axis tight;
    set(gca, 'XTick',[], 'YTick',[]);
    title(sprintf('U%d', uids(k)), 'FontWeight','normal');
end
title(t, 'Spike Waveforms – Stacked Channel Traces');
xlabel(t, 'Time samples'); ylabel(t, 'Channels (offset)');

%% look at raster plot
% ----- Dot raster -----
uid = 1;
R = GoodUnitStrc(uid).Raster;          % [nTrials x 450]
binMs = 1; t = (0:size(R,2)-1)*binMs;

figure('Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; hold on;
% If counts can be >1, color by count
[row, col, val] = find(R);             % row=trial, col=timeBin, val=counts
scatter((col-1)*binMs, row, 4, val, 'filled');
set(gca, 'YDir','reverse');            % trial 1 at top
xlim([t(1) t(end)]); ylim([0.5 size(R,1)+0.5]);
xlabel('Time (ms)'); ylabel('Trial');
title(sprintf('Unit %d — raster (dots colored by count)', uid));
colorbar; xline([0 450], '--k', 'LineWidth', 0.5);

%% ----- PSTH -----
nexttile;
meanCounts = mean(R,1);
sigmaMs = 10; 
win = round(5*sigmaMs);
g = exp(-(-win:win).^2/(2*sigmaMs^2)); g = g/sum(g);
rateHz = conv(meanCounts, g, 'same') * 1000;  % Hz
plot(t, rateHz, 'k', 'LineWidth', 1.5); grid on;
xlim([t(1) t(end)]);
xlabel('Time (ms)'); ylabel('Firing rate (Hz)');
title('PSTH');

%% ------ AVG PSTH ------
iidxs = meta_data(uid).trial_valid_idx(meta_data(uid).trial_valid_idx~=0)';

uniqueImgs = unique(iidxs);
nImgs = numel(uniqueImgs);
nTime = size(R,2);
avgPSTH = nan(nImgs, nTime);

for ii = 1:nImgs
    img = uniqueImgs(ii);
    trialIdx = (iidxs == img);
    avgPSTH(ii,:) = mean(R(trialIdx,:), 1);  % average across trials of same image
end

figure;
plot(mean(avgPSTH,1), 'k', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Mean firing (spikes/bin)');
title(sprintf('Overall PSTH — Unit %d', uid));

%% ----- smooth -----
iid = 1071;                            % image index (1-1072)
meanPSTH = avgPSTH(iid, :);
binMs = 1;
sigma = 2;                        % smoothing width (ms)
win = round(5*sigma/binMs);        % ±5σ window
g = exp(-(-win:win).^2/(2*sigma^2));
g = g/sum(g);

psth_smooth = conv(meanPSTH, g, 'same');   % smoothed PSTH
rateHz = psth_smooth * (1000/binMs);       % convert to Hz

plot(1:size(rateHz, 2), rateHz, 'k', 'LineWidth', 1.5);