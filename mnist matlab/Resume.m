% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images.idx3-ubyte');
Ytrain = loadMNISTLabels('train-labels.idx1-ubyte');
 
Xtrain = reshape(images, 28, 28, 1, 60000);

[blocksize,~,~,numTrainingFiles]=size(X);

idx = randperm(size(Xtrain,4),10000);

Xtest=Xtrain(:,:,:,idx);
Ytest=Ytrain(idx);

Xtrain(:,:,:,idx)=[];
Ytrain(idx)=[];

% Set aside 1000 of the images for network validation.
idx = randperm(size(Xtrain,4),1000);

Xvalidation = Xtrain(:,:,:,idx);
Yvalidation = Ytrain(idx);

Xtrain(:,:,:,idx) = [];
Ytrain(idx) = [];


options = trainingOptions('sgdm', ...
    'MaxEpochs',50,...
    'InitialLearnRate',3e-4, ...
    'Verbose',false, ...
    'MiniBatchSize', 64,...
    'CheckpointPath','./nets',...
    'Plots','training-progress', ...
    'ValidationData',{Xvalidation,categorical(Yvalidation) });

load('./nets/net_checkpoint__38250__2019_04_25__23_35_29.mat','net');
net = trainNetwork(Xtrain,categorical(Ytrain),net.Layers,options);

YPred = classify(net,Xtest);

accuracy = sum(YPred == categorical(Ytest))/numel(Ytest)


