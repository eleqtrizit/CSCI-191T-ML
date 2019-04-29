% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images.idx3-ubyte');
Y = loadMNISTLabels('train-labels.idx1-ubyte');
 
X = reshape(images, 28, 28, 1, 60000);

[blocksize,~,~,numTrainingFiles]=size(X);

Xtrain=X(:,:,:,1:50000);
Xtest=X(:,:,:,50001:60000);
Ytrain=Y(1:50000);
Ytest=Y(50001:60000);

% Set aside 1000 of the images for network validation.
idx = randperm(size(Xtrain,4),1000);
Xvalidation = Xtrain(:,:,:,idx);
Xtrain(:,:,:,idx) = [];
Yvalidation = Ytrain(idx);
Ytrain(idx) = [];


layers = [
    imageInputLayer([28 28 1])

    % (Kernel Size, Num Filters)
    convolution2dLayer(5, 50, 'Padding', 0)
    maxPooling2dLayer(2, 'Stride', 2)
    reluLayer()
    batchNormalizationLayer()

    convolution2dLayer(3, 40, 'Padding', 1)
    maxPooling2dLayer(2, 'Stride', 2)
    reluLayer()
    batchNormalizationLayer()

    convolution2dLayer(3, 30, 'Padding', 1)
    reluLayer()
    batchNormalizationLayer()

    fullyConnectedLayer(48)
    leakyReluLayer()
    dropoutLayer(0.1)
    fullyConnectedLayer(32)
    leakyReluLayer()
    dropoutLayer(0.1)
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];


options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.045, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.98, ...
    'LearnRateDropPeriod', 1, ...
    'L2Regularization', 0.00001, ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 200, ...
    'Momentum', 0.65, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',{Xvalidation,categorical(Yvalidation) }, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5000, ...
    'Verbose', true, ...
    'VerboseFrequency', 50);



%% Image Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-7.5 7]);

augimds = augmentedImageDatastore([28 28 1],Xtrain,categorical(Ytrain),'DataAugmentation',imageAugmenter);

net = trainNetwork(Xtrain,categorical(Ytrain),layers,options);

YPred = classify(net,Xtest);

accuracy = sum(YPred == categorical(Ytest))/numel(Ytest)


