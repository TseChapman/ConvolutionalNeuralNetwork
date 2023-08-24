epochs = 10; 
learningRate = 0.1; 
numNeurons = 100; 
inputSize = 6760; 
outputSize = 10;
kernelSize = 3;
numFilters = 40;

variableLearningRateThreshold = 1.04;
growthCoefficient = 1.5;
decayCoefficient = 0.7;
varyLearningRate = 0;

% Read training data
training = readmatrix("train.csv");
% Randomize training data order
training = training(randperm(size(training, 1)), :);

% split the training data into 70% train and 30% validate
index = randperm(size(training,1));
n70 = round(0.7 * numel(index));
n30 = numel(index) - n70;

% Get all the training data
[trainingSetAll, tIDs, tLabelsSingle] = readTrainingMatrix(training);
disp(size(trainingSetAll))

% Split into training and validate sets
trainingSet = trainingSetAll(index(1:n70),:);
%trainingSet = trainingSetAll;
trainingSet = reshape(trainingSet', 28, 28, []);
trainingSet = double(trainingSet) ./ double(255);
trainingTarget = tLabelsSingle(index(1:n70),:)';
%trainingTarget = tLabelsSingle';
trainingTarget(trainingTarget == 0) = 10;

validateSet = trainingSetAll(index(n70+1:end),:);
validateSet = reshape(validateSet', 28, 28, []);
validateSet = double(validateSet) ./ double(255);
validateTarget = tLabelsSingle(index(n70+1:end),:)';
validateTarget(validateTarget == 0) = 10;

% Display the size of the training and validation set
disp(size(trainingSet))
disp(size(trainingTarget))

disp(size(validateSet))
disp(size(validateTarget))

% define the network based on the parameters
obj = Convolution(inputSize, outputSize, numNeurons, kernelSize, numFilters, learningRate, decayCoefficient, variableLearningRateThreshold, growthCoefficient);

% train the network using training set and validate in each epoch using the
%   validate set
obj = obj.trainTest(trainingSet, trainingTarget, epochs, varyLearningRate, validateSet, validateTarget);

disp("Done")

sub = createSubmission(obj, "test.csv");
writematrix(sub,'Convolution-submission.csv');

% createSub
% read the testing file and run the testing data on the network
% return a matrix representing in following collumns: Id and label
% Id is the testing image id read from the testing data
% label is the result from running the testing image into the network and
%   produce a classification result from the network
function createSub = createSubmission(Network, testSet)
    % Read testing data and format it
    testing = readmatrix(testSet);
    [testingSet, testIDs] = readTestTemp(testing);
    testingSet = reshape(testingSet', 28, 28, []);
    testingSet = double(testingSet) ./ double(255);
    sz = size(testIDs,1);
    createSub = zeros(sz,2);

    for idx = 1:sz
        % Get classification output from network based on the image
        image = testingSet(:,:,idx);
        [in2, in3, in4, wp2, a] = Network.forward(image);
        [~, temp] = max(a);
        tempAnswer = find(a==1,1);
        if (isempty(tempAnswer)) 
            tempAnswer = 0;
        end

        % Write the result to the result matrix
        createSub(idx,1) = testIDs(idx,1);
        if (temp == 10)
            createSub(idx,2) = 0;
        else
            createSub(idx,2) = temp;
        end
    end

end

% readTestTemp
% a temporary function used to get the testing image pixels and testing
%   image's Id
function [testingSet, testIDs] = readTestTemp(testing)
    testIDs = testing(:,1);
    testingSet = testing(:,2:end);
end