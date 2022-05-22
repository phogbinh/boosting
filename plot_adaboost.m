train_data = readmatrix("data/basic.txt");
train_data_N = numel( train_data(:, 1) );

train_pos = zeros(train_data_N, 2);
train_pos_N = 0;
train_neg = zeros(train_data_N, 2);
train_neg_N = 0;
for i = 1:train_data_N
    if train_data(i, 3) == 1
        train_pos_N = train_pos_N + 1;
        train_pos(train_pos_N, :) = train_data(i, 1:2);
    else
        train_neg_N = train_neg_N + 1;
        train_neg(train_neg_N, :) = train_data(i, 1:2);
    end
end
train_pos = train_pos(1:train_pos_N, :);
train_neg = train_neg(1:train_neg_N, :);

xmin = min( train_data(:, 1) ) - 1;
xmax = max( train_data(:, 1) ) + 1;
ymin = min( train_data(:, 2) ) - 1;
ymax = max( train_data(:, 2) ) + 1;

row_N = 1000;
row = linspace(ymin, ymax, row_N);
col_N = 1000;
col = linspace(xmin, xmax, col_N);

[C, R] = meshgrid(col, row);

f = readmatrix("f.txt");

figure;
hold on;
[~, c] = contourf(C, R, f, [-inf -1 0 1]);
colormap( [0 0 0.5;
           0 0.5 1;
           1 0.5 0;
           0.5 0 0] );
caxis([-2 1]);
drawnow; % make FacePrims available
fp = c.FacePrims;
[fp.ColorType] = deal('truecoloralpha'); % default 'truecolor'
for i = 1:numel(fp)
    fp(i).ColorData(4) = 150; % default 255
end

plot(train_pos(:, 1), train_pos(:, 2), '+r', ...
     'LineWidth', 3, ...
     'MarkerSize', 10);
plot(train_neg(:, 1), train_neg(:, 2), 'ob', ...
     'LineWidth', 3, ...
     'MarkerSize', 10);
hold off;