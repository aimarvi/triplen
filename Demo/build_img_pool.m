clear

savedir = '.';
datadir = '../../datasets/NNN/NSD1000_LOC';
n_images  = 1000;
img_pool = cell(n_images,1);
missing  = false(n_images,1);

for k = 1:n_images
    fname = fullfile(datadir, sprintf('%04d.bmp', k));  % 0001.bmp ... 1000.bmp
    if exist(fname, 'file') == 2
        img_pool{k} = imread(fname);
    else
        missing(k) = true;   % keep track of any missing files
        img_pool{k} = [];    % placeholder to preserve indexing
        warning('Missing file: %s', fname);
    end
end

% Try to build a packed array if all images are present and identical shape/type
buildPacked = ~any(missing);
if buildPacked
    first_img = img_pool{1};
    sameSize = all(cellfun(@(I) isequal(size(I), size(first_img)), img_pool));
    sameType = all(cellfun(@(I) isa(I, class(first_img)), img_pool));
    if sameSize && sameType
        if ndims(first_img) == 2
            % Grayscale: stack as H x W x N
            [H,W] = size(first_img);
            stack = zeros(H, W, nImages, 'like', first_img);
            for k = 1:nImages, stack(:,:,k) = img_pool{k}; end
        else
            % Color (or multispectral): stack as H x W x C x N
            [H,W,C] = size(first_img);
            stack = zeros(H, W, C, n_images, 'like', first_img);
            for k = 1:n_images, stack(:,:,:,k) = img_pool{k}; end
        end
    else
        buildPacked = false; % shapes or types differ; keep just the cell array
    end
end

% Save to img_pool.mat (v7.3 handles large variables)
save([savedir filesep 'img_pool.mat'], 'img_pool', 'missing', '-v7.3');
if exist('stack', 'var')
    save([savedir filesep 'img_pool.mat'], 'stack', '-append');
end