function export_calce_numeric(inputdir, outdir)
% Numeric copy of the CALCE PL17 and PL21 tables, readable without MATLAB.
ids = {'PL17','PL21'};
for c = 1:numel(ids)
    id = ids{c};
    source = load(fullfile(inputdir,[id '.mat']),id);
    cells = source.(id);
    blocks = struct('operation_index',{},'operation',{},'start_date',{},'columns',{},'data',{});
    for j = 2:size(cells,1)
        t = cells{j,3};
        rec.operation_index = j-1;
        rec.operation = char(string(cells{j,1}));
        rec.start_date = char(string(cells{j,2}));
        if istable(t)
            rec.columns = t.Properties.VariableNames;
            rec.data = table2array(t);
            assert(isnumeric(rec.data));
        else
            assert(isempty(t));
            rec.columns = {};
            rec.data = zeros(0,8);
        end
        blocks(end+1) = rec; %#ok<SAGROW>
    end
    save(fullfile(outdir,[id '_numeric.mat']),'blocks','-v7');
    fprintf('%s: %d blocks\n',id,numel(blocks));
end
end
