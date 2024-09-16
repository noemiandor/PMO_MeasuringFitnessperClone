function [tform] = CoRegister(moving,fixed,regist_type,GrowthFactor,Iter_No)
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.00005;
optimizer.Epsilon = 1.5e-6;
optimizer.GrowthFactor = GrowthFactor;
optimizer.MaximumIterations = Iter_No;
tform = imregtform(moving, fixed, regist_type, optimizer, metric);
end

