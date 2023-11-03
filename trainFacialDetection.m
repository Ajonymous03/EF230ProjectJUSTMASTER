function returned = trainFacialDetection()
% Purpose: Creates a trained network of the faces in the database
% Input: NONE
% Output: The trained network that can then be passed to the facial detection
%          function
% Usage: network = trainFacialDetection()
    g=alexnet; % Creates an AlexNet network
    layers=g.Layers; % Creates an array of layers for the network
    layers(23)=fullyConnectedLayer(4); % Defines a fully connected layer with an output size of 4
    layers(25)=classificationLayer; % Defines classification layer
    allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames'); % Creates an ImageDatastore in order to work with database images
    opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64); % Creates training options for the solver
    myNet1=trainNetwork(allImages,layers,opts); % Trains the network according to previously defined parameters
    returned = myNet1;
end