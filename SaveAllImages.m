%T = readtable('../../database/filenames.csv');
data=textread('../../database/filenames.csv','%s','delimiter','\n');  % cell string array
disp(size(data,1))

%for gama=0:30:30 % loop for gama angle 0 to 30 in steps of 5
    %s = 0
    gama=0 %just for 0 
    for beta=5:5:85  % loop for beta angle 5 to 85 in steps of 5
        disp(strcat(int2str(gama),'_',int2str(beta))) 
        dir_name=strcat('0_',int2str(beta),'_',int2str(gama))
        status = mkdir(strcat('../../images/images_',dir_name));
        disp(status)
        if status==1
            for i=2:size(data,1)
                try
                    filepath=strcat('../../database/',data{i});
                    AngularPlot(filepath,'abg',[0 beta gama],'minangle',6,'maxangle',25)
                    imagefilepath=strcat('../../images/images_',dir_name,'/',data{i},'_',dir_name,'.png');
                    saveas(gcf,imagefilepath)   
                catch
                    disp(strcat('exception for',imagefilepath))
                    continue
                end
            end    
        end
        %s = s + 1
    end  
    %t = t + 1
%end
%disp(X)
%AngularPlot('../../database/l89.1_d29.7_flat.h5','abg',[0 10 10],'minangle',6,'maxangle',25)
% Requires R2020a or later
%saveas(gcf,'l89.1_d29.7_0_10_10.png')
