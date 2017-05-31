% Simple Convolutional Neural Network for Deep Learning Regression

%% Load Training Data

% Load the digit training set as 4-D array data using digitTrain4DArrayData

[trainImages,~,trainAngles] = digitTrain4DArrayData;

%% Display 20 random sample training digits using imshow
numTrainImages = size(trainImages,4);

figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)

    imshow(trainImages(:,:,:,idx(i)))
    drawnow
end

%% Create Network Layers

% create the layers of the network and include a regression layer at the 
% end of the network

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(12,25)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

%% Train Network

% Create the network training options
options = trainingOptions('sgdm','InitialLearnRate',0.001, ...
    'MaxEpochs',15);

% Create the network using trainNetwork
% If the accuracy is too low, or the RMSE is too high, then try increasing
% the value of 'MaxEpochs' in the call to trainingOptions
net = trainNetwork(trainImages,trainAngles,layers,options)

% Examine the details of the network architecture contained in the Layers
% property of net
% net.Layers

%% Test Network

% Test the performance of the network by evaluating the prediction accuracy
% of held out test data

% Load the digit test set
[testImages,~,testAngles] = digitTest4DArrayData;

% Use predict to predict the angles of rotation of the test images
predictedTestAngles = predict(net,testImages);

% Calculate the prediction error between the predicted and actual angles of rotation
predictionError = testAngles - predictedTestAngles;

% Calculate the number of predictions within an acceptable error margin 
% from the true angles
% Set the threshold to be 10 degrees 
% Calculate the percentage of predictions within this threshold

thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numTestImages = size(testImages,4);

accuracy = numCorrect/numTestImages

% Use the root-mean-square error (RMSE) to measure the differences between
% the predicted and actual angles of rotation

squares = predictionError.^2;
rmse = sqrt(mean(squares))


