%% figure out which one was best and use it

testset = csvread('test.csv', 1, 0);   

% list of files in folderName
netCheckpoints=what('netsold');
% get the list of the epochs
numberOfEpochs=size(netCheckpoints.mat,1); 
testNet=load(['netsold/' netCheckpoints.mat{numberOfEpochs}]);
ImageId=[];
Label=[];

for i = 1:28000                                    
    digit = reshape(testset(i, 1:end), [28,28])';   % row = 28 x 28 image
    [Ycat,YPred] = classify(testNet.net,digit);
    ImageId=[ImageId i];
    Label=[Label Ycat];
end

ImageId=ImageId';
Label=Label';
writetable(table(ImageId, Label), 'submission.csv')