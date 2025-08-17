%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       This code will co-register images in time searies sequentially       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% There are 2 co-registration methos: crosscorrelation and intensity 
% based registration
%
% The code take series images and their segmentation masks as input
% The image and masks must be organized in specific structure (see the
% example input data)
%
% Image and mask names should be the same
% the naming of files should indicate time stamp in order to sort them sequentially
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;

% path to the time series images
Timeseries_path='Example_Input_Data\Images';

% path to the time series masks
Masks_path='Example_Input_Data\Masks';

% path to the save location
save_path='Co_registered_images';

% list all time series
[Timeseries_list] = ListDir(Timeseries_path,'FOLDER');
for s=1:size(Timeseries_list,1)

    fprintf('%d : %s\n',s,Timeseries_list{s,1});

    % set pathes
    Co_registered_imagse_Save_Path=fullfile(save_path,'Reg_Images',Timeseries_list{s,1});
    Co_registered_Masks_Save_Path=fullfile(save_path,'Reg_Masks',Timeseries_list{s,1});

    if ~exist(Co_registered_imagse_Save_Path, 'dir'), mkdir (Co_registered_imagse_Save_Path); end
    if ~exist(Co_registered_Masks_Save_Path, 'dir'), mkdir (Co_registered_Masks_Save_Path); end


    % list images within the series and masks
    I_Path=fullfile(Timeseries_path,Timeseries_list{s,1});
    M_Path=fullfile(Masks_path,Timeseries_list{s,1});
    [Images] = ListDir(I_Path,'FILE','tif');
    Images = sort(Images);

    % Read first image and first mask (time 1)
    Fixed_Image_Path=fullfile(I_Path,Images{1,1});
    Fixed_Mask_Path=fullfile(M_Path,Images{1,1});
    Fixed_I=imread(Fixed_Image_Path);
    Fixed_M=imread(Fixed_Mask_Path);

    % save fixed
    SaveName=fullfile(Co_registered_imagse_Save_Path,Images{1,1});
    imwrite(Fixed_I, SaveName);

    SaveName=fullfile(Co_registered_Masks_Save_Path,Images{1,1});
    imwrite(uint8((Fixed_M)),SaveName);

    % start co-registering following images. Strat from the second
    for i=2:size(Images,1)

        % set pathes to the following image and mask
        [~,FileName,~]=fileparts(Images{i,1});
        MImage_Path=fullfile(I_Path,Images{i,1});
        Mask_Path=fullfile(M_Path,Images{i,1});

        %read the follow image and mask
        [MOVING_I,cmap_I]=imread(MImage_Path);
        [MOVING_M,cmap_M]=imread(Mask_Path);



        %------------------------------------------------
        %subpixel image registration by crosscorrelation
        %------------------------------------------------
        usfac = 0; % subpixel parameter (see dftregistration.m file)
        [~, MOVING_I,MOVING_M] = dftregistration(Fixed_I,MOVING_I,MOVING_M,usfac);
        

        %------------------------------------------------
        % Intensity based co-registration
        %------------------------------------------------
        Reg_Type='similarity';   % 'rigid", 'similarity', or 'affine'
        Fillin_Value=130;  % 130=gray, 255=white, 0=black
        GrowthFactor=2; % gradient Growth Factor
        Iter_No=100;
        [MOVING_I,MOVING_M] = CoRegister_Images(MOVING_I,MOVING_M,Fixed_I,Reg_Type,Fillin_Value,GrowthFactor,Iter_No);
        MOVING_M=logical(MOVING_M);

        

        % Save coregistered image and mask
        SaveName=fullfile(Co_registered_imagse_Save_Path,Images{i,1});
        imwrite(MOVING_I, SaveName);

        SaveName=fullfile(Co_registered_Masks_Save_Path,Images{i,1});
        imwrite(uint8((MOVING_M)),SaveName);


        % set the fix image to be used in the follow step 
        Fixed_I=MOVING_I;
    end
end
