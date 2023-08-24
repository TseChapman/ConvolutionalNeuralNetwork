%reads parameters from a matrix instead of file
function[images, IDs, labels] = readTrainingMatrix(shuffle)
IDs = shuffle(:,1);
labels = shuffle(:,2);
images = shuffle(:,3:end);


