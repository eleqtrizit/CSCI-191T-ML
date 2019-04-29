tr = csvread('train.csv', 1, 0);                  % read train.csv
 
n = size(tr, 1);                    % number of samples in the dataset
Y = tr(:,1);                 % 1st column is |label|
inputs = tr(:,2:end);               % the rest of columns are predictors

b=28;
d=1;
X=zeros(b,b,d,n);

for i=1:n
    X(:,:,:,i)=reshape(tr(i, 2:end), [28,28])'; 
end


Xtrain=X(:,:,:,1:38000);
Xtest=X(:,:,:,38001:42000);
Ytrain=Y(1:38000);
Ytest=Y(38001:42000);

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


