function [co_registered_Image,co_registered_Mask] = CoRegister_Images(moving_I,moving_Mask,fixed,Reg_Type,Fillin_Value,GrowthFactor,Iter_No)

% convert to gray level image
if(size(fixed,3)>1)
    Fixed_Image=rgb2gray(fixed);
else
    Fixed_Image=fixed;
end

if(size(moving_I,3)>1)
    Moving_Image=rgb2gray(moving_I);
else
    Moving_Image=moving_I;
end


[TM] = CoRegister(Moving_Image,Fixed_Image,Reg_Type,GrowthFactor,Iter_No);

% apply co-registration matrix to moving image 
co_registered_Image=zeros(size(Fixed_Image,1),size(Fixed_Image,2),size(moving_I,3),class(moving_I));
for i=1:size(moving_I,3)
    co_registered_Image(:,:,i) = imwarp((moving_I(:,:,i)),TM,'OutputView',imref2d(size((Fixed_Image))),'FillValues',Fillin_Value);
end

% apply co-registration matrix to moving mask 
co_registered_Mask = imwarp((moving_Mask),TM,"nearest" ,'OutputView',imref2d(size((Fixed_Image))),'FillValues',Fillin_Value);

end

