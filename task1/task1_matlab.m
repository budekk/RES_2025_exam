% Open the TIFF files
tif_1_lst = imread('t1_lst2023_Jul_Aug.tif');
tif_1_ndvi = imread('t1_ndvi2023_Jul_Aug.tif');
tif_2_lst = imread('t2_lst2023_Jul_Aug.tif');
tif_2_ndvi = imread('t2_ndvi2023_Jul_Aug.tif');

% Function to display an image in grayscale
function display_grayscale(image, titleText, ax)
    imshow(image, [], 'Parent', ax);
    colormap(ax, gray);
    title(ax, titleText);
    ax.XTick = [];
    ax.YTick = [];
end

% Function to plot a histogram for an image
function plot_histogram(image, titleText, ax)
    histogram(ax, image(:), 'BinLimits', [0, 255], 'FaceColor', 'blue', 'FaceAlpha', 0.7);
    title(ax, titleText);
    xlim(ax, [0, 255]);
end

% Plot histograms for all four TIFF files
figure;
subplot(2, 2, 1);
plot_histogram(tif_1_lst, 'Histogram TIF 1 LST', gca);
subplot(2, 2, 2);
plot_histogram(tif_1_ndvi, 'Histogram TIF 1 NDVI', gca);
subplot(2, 2, 3);
plot_histogram(tif_2_lst, 'Histogram TIF 2 LST', gca);
subplot(2, 2, 4);
plot_histogram(tif_2_ndvi, 'Histogram TIF 2 NDVI', gca);

% Display the grayscale images
figure;
subplot(2, 2, 1);
display_grayscale(tif_1_lst, 'Grayscale TIF 1 LST', gca);
subplot(2, 2, 2);
display_grayscale(tif_1_ndvi, 'Grayscale TIF 1 NDVI', gca);
subplot(2, 2, 3);
display_grayscale(tif_2_lst, 'Grayscale TIF 2 LST', gca);
subplot(2, 2, 4);
display_grayscale(tif_2_ndvi, 'Grayscale TIF 2 NDVI', gca);

% Function to create a scatter plot
function create_scatter_plot(ndvi_image, lst_image, titleText, ax)
    scatter(ax, double(ndvi_image(:)), double(lst_image(:)), 1, 'filled', 'MarkerFaceAlpha', 0.5);
    title(ax, titleText);
    xlabel(ax, 'NDVI');
    ylabel(ax, 'Temperature (LST)');
end

% Create scatter plots for the two sets of images
figure;
subplot(1, 2, 1);
create_scatter_plot(tif_1_ndvi, tif_1_lst, 'Scatter Plot TIF 1', gca);
subplot(1, 2, 2);
create_scatter_plot(tif_2_ndvi, tif_2_lst, 'Scatter Plot TIF 2', gca);
