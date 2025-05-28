%% Extract mean response for Middle Body from Subject3
clear
load img_pool.mat
interesred_ROI = 'MB3';
manual_data = readtable('exclude_area.xls');

for row_in_table = 1:size(manual_data,1)
    if(~strcmp(manual_data.AREALABEL{row_in_table},interesred_ROI))
        continue
    end

    ses_idx = manual_data.SesIdx(row_in_table);
    fprintf('extracting from session %d \n', ses_idx)

    all_procdata = dir(fullfile('Data', sprintf('Processed_ses%d*', ses_idx)));
    proc_data = load(fullfile('Data', all_procdata.name));

    % y location limit for spike position
    y1_here = manual_data.y1(row_in_table);
    y2_here = manual_data.y2(row_in_table);

    % combine location and reliability
    good_unit_idx = find(proc_data.pos>y1_here & proc_data.pos<y2_here & proc_data.reliability_best>0.4);

    
    rsp_mtx = proc_data.response_best(good_unit_idx,1:1000);
    rsp_mtx = zscore(rsp_mtx, 0, 2); % normalize for each unit, someone would prefer skip this

    figure;set(gcf,'Position',[5 500 2000 350])
    subplot(2,10,1); hold on
    BSI = proc_data.B_SI(good_unit_idx);
    histogram(BSI,-2:0.2:2,'EdgeAlpha',0)
    xline(median(BSI),'LineWidth',2)
    title(sprintf('%.02f percent over 0.2', 100*sum(BSI>0.2)./length(BSI)))
    xlabel('Body selectivity')


    rsp_mean = mean(rsp_mtx); % averaged across units
    [~,img_order] = sort(rsp_mean,'descend');
    img_show = [];
    for example_idx = [1:5] % show some most/least preferred img
        img_show = [img_show, img_pool{img_order(example_idx)}];
    end
    subplot(2,10,2:5)
    imshow(img_show)
    title('Most Preferred')
    
    img_show = [];
    for example_idx = [996:1000] % show some most/least preferred img
        img_show = [img_show, img_pool{img_order(example_idx)}];
    end
    subplot(2,10,6:9)
    imshow(img_show)
    title('Least Preferred')

    % show populational response (similar to fig.2)
    subplot(2,10,11:19)
    imagesc(rsp_mtx(:,img_order));
    clim([-1,2])
    colormap('Hot')
    xlabel('Img, sorted by population mean')
    ylabel('Unit')

    saveas(gcf,sprintf('demo2_%d.png',ses_idx))
end

%%