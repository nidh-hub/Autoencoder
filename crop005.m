files = dir('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/train');
for i = 4:length(files)
    disp(files(i).name)
     theImage = imread(append('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/train/',files(i).name));
%      disp(theImage);
     croppedImage055 = imcrop(theImage,[228,58,649,649]);
%      baseFileName = sprintf('Image #%d.jpg', i);
     
%      fullFileName = fullfile(files,baseFileName);
     imwrite(croppedImage055, append('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/crop_tr/',files(i).name));
% disp(files(i).name);
% image= imread(append('/Users/apple/Documents/MSCPROJ/images/images_0_5_5/train/',files(i).name));
end