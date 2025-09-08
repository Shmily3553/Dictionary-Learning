inputFolder = 'trans_images';
outputFolder = 'images';

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

imageFiles = dir(fullfile(inputFolder, '*.gif'));

for k = 1:length(imageFiles)
    filename = imageFiles(k).name;
    filepath = fullfile(inputFolder, filename);
    
    img = imread(filepath);
    
    if size(img, 3) == 3
        img = rgb2gray(img);  % Convert to grayscale if needed
    end
    
    img = imresize(img, [512 512]);  % Resize to 512x512
    
    [~, name, ~] = fileparts(filename);
    outputFile = fullfile(outputFolder, [name '.png']);
    imwrite(img, outputFile);
end
