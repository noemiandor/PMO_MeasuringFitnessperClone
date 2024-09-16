%returns either a list of subfolder names in the given folder, or
% a list of file names in the given folder
% If the file extention is given, then only files with that extension will
% be listed, otherwise all files will be listed.
function [List] = ListDir(FolderPath,ListType, extension)
%read the folders contents
if(strcmp(ListType, 'FOLDER'))
    try
        CohortsDir = dir(char(FolderPath));
    catch
        List =[];
        fprintf('Cannot read folder %s.\n',char(FolderPath));
        return
    end
    isub = [CohortsDir(:).isdir];
    List = {CohortsDir(isub).name}';
    List(ismember(List,{'.','..'})) = [];
elseif (strcmpi(ListType , 'FILE'))
    if nargin ==2
        extension='*';
    end
    filetype= strcat('*.',extension);
    FolderPath = fullfile(FolderPath,filetype);
    try
        CohortsDir = dir(char(FolderPath));
    catch
        List =[];
        fprintf('Cannot read folder %s.\n',char(FolderPath));
        
        return
    end
    isub = ~[CohortsDir(:).isdir];
    List = {CohortsDir(isub).name}';
end
end