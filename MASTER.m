function MASTER(rvr)
    noAlarm = true;
    trainedNetwork = trainFacialDetection();
    while noAlarm
        runFacialDetection(trainedNetwork, rvr);
        detected = motionDetection(rvr);
        if detected == true
            playMotionDetectedAudio();
            faceScanCountdown();
            faceInSystem = runFacialDetection(rvr);
                if faceInSystem
        end
    end
    function faceScanCountdown()
        h = waitbar(0, 'Facial Recognition Occurs When Bar is Full');
        for i = 1:10
            pause(1);
            waitbar(i/10, h);
        end
        close(h)
    end
    function playMotionDetectedAudio()
        for i = 1:1
            [y, Fs] = audioread('motionDetected.mp3');
            sound(y, Fs, 16);
            pause(4.1);
        end
    end
    function playFaceAcceptedAudioJonathan()
    function driveRover(rvr)
        rvr.resetHeading
        tstart = tic;
        g = 1;
        while g < 5
            rvr.setDriveSpeed(50);
            while toc(tstart)<3
                msgbox('Driving');
                pause(3)
        
            end
            g+1
            turnAngle(90)
        end
        rvr.stop
        msgbox('Route Done')
    end
    function isMotion = motionDetection(rvr)
        function getVideo(rvr)
            h = waitbar(0,'Taking Video');
            myVideo1 = VideoWriter('cam1.avi');
            myVideo1.FrameRate = 10;
            open(myVideo1);
            totalFrames = 20;
            for i=1:totalFrames
                img1 = getImage(rvr);
                writeVideo(myVideo1, img1);
                waitbar(i/20, h);
            end
            close(h);
            close(myVideo1);
        end
        function motionDetected = MotionBasedMultiObjectTrackingExample(rvr)
            % Create System objects used for reading video, detecting moving objects,
            % displaying the results.
            % Also collects video that is processed
            getVideo(rvr);
    
            obj = setupSystemObjects();
            
            tracks = initializeTracks(); % Create an empty array of tracks.
            
            nextId = 1; % ID of the next track
            totalMotionDetected = [];
            % Detect moving objects, and track them across video frames.
            while hasFrame(obj.reader)
                frame = readFrame(obj.reader);
                [centroids, bboxes, mask] = detectObjects(frame);
                [a, ~, ~] = detectObjects(frame);
                if a > 0
                    totalMotionDetected = [totalMotionDetected 1];
                else
                    totalMotionDetected = [totalMotionDetected 0];
                end
                predictNewLocationsOfTracks();
                [assignments, unassignedTracks, unassignedDetections] = ...
                    detectionToTrackAssignment();
            
                updateAssignedTracks();
                updateUnassignedTracks();
                deleteLostTracks();
                createNewTracks();
            end
            function obj = setupSystemObjects()
                % Initialize Video I/O
                % Create objects for reading a video from a file, drawing the tracked
                % objects in each frame, and playing the video.
        
                % Create a video reader.
                obj.reader = VideoReader('cam1.avi');
        
                % Create two video players, one to display the video,
                % and one to display the foreground mask.
                obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
                obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
        
                % Create System objects for foreground detection and blob analysis
        
                % The foreground detector is used to segment moving objects from
                % the background. It outputs a binary mask, where the pixel value
                % of 1 corresponds to the foreground and the value of 0 corresponds
                % to the background.
        
                obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
                    'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        
                % Connected groups of foreground pixels are likely to correspond to moving
                % objects.  The blob analysis System object is used to find such groups
                % (called 'blobs' or 'connected components'), and compute their
                % characteristics, such as area, centroid, and the bounding box.
        
                obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
                    'AreaOutputPort', true, 'CentroidOutputPort', true, ...
                    'MinimumBlobArea', 400);
            end
            function tracks = initializeTracks()
                % create an empty array of tracks
                tracks = struct(...
                    'id', {}, ...
                    'bbox', {}, ...
                    'kalmanFilter', {}, ...
                    'age', {}, ...
                    'totalVisibleCount', {}, ...
                    'consecutiveInvisibleCount', {});
            end
            function [centroids, bboxes, mask] = detectObjects(frame)
        
                % Detect foreground.
                mask = obj.detector.step(frame);
        
                % Apply morphological operations to remove noise and fill in holes.
                mask = imopen(mask, strel('rectangle', [3,3]));
                mask = imclose(mask, strel('rectangle', [15, 15]));
                mask = imfill(mask, 'holes');
        
                % Perform blob analysis to find connected components.
                [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
            end
        
            function predictNewLocationsOfTracks()
                for i = 1:length(tracks)
                    bbox = tracks(i).bbox;
        
                    % Predict the current location of the track.
                    predictedCentroid = predict(tracks(i).kalmanFilter);
        
                    % Shift the bounding box so that its center is at
                    % the predicted location.
                    predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
                    tracks(i).bbox = [predictedCentroid, bbox(3:4)];
                end
            end
        
            function [assignments, unassignedTracks, unassignedDetections] = ...
                    detectionToTrackAssignment()
        
                nTracks = length(tracks);
                nDetections = size(centroids, 1);
        
                % Compute the cost of assigning each detection to each track.
                cost = zeros(nTracks, nDetections);
                for i = 1:nTracks
                    cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
                end
        
                % Solve the assignment problem.
                costOfNonAssignment = 20;
                [assignments, unassignedTracks, unassignedDetections] = ...
                    assignDetectionsToTracks(cost, costOfNonAssignment);
            end
            function updateAssignedTracks()
                numAssignedTracks = size(assignments, 1);
                for i = 1:numAssignedTracks
                    trackIdx = assignments(i, 1);
                    detectionIdx = assignments(i, 2);
                    centroid = centroids(detectionIdx, :);
                    bbox = bboxes(detectionIdx, :);
        
                    % Correct the estimate of the object's location
                    % using the new detection.
                    correct(tracks(trackIdx).kalmanFilter, centroid);
        
                    % Replace predicted bounding box with detected
                    % bounding box.
                    tracks(trackIdx).bbox = bbox;
        
                    % Update track's age.
                    tracks(trackIdx).age = tracks(trackIdx).age + 1;
        
                    % Update visibility.
                    tracks(trackIdx).totalVisibleCount = ...
                        tracks(trackIdx).totalVisibleCount + 1;
                    tracks(trackIdx).consecutiveInvisibleCount = 0;
                end
            end
            function updateUnassignedTracks()
                for i = 1:length(unassignedTracks)
                    ind = unassignedTracks(i);
                    tracks(ind).age = tracks(ind).age + 1;
                    tracks(ind).consecutiveInvisibleCount = ...
                        tracks(ind).consecutiveInvisibleCount + 1;
                end
            end
            function deleteLostTracks()
                if isempty(tracks)
                    return;
                end
        
                invisibleForTooLong = 20;
                ageThreshold = 8;
        
                % Compute the fraction of the track's age for which it was visible.
                ages = [tracks(:).age];
                totalVisibleCounts = [tracks(:).totalVisibleCount];
                visibility = totalVisibleCounts ./ ages;
        
                % Find the indices of 'lost' tracks.
                lostInds = (ages < ageThreshold & visibility < 0.6) | ...
                    [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
                % Delete lost tracks.
                tracks = tracks(~lostInds);
            end
            function createNewTracks()
                centroids = centroids(unassignedDetections, :);
                bboxes = bboxes(unassignedDetections, :);
        
                for i = 1:size(centroids, 1)
        
                    centroid = centroids(i,:);
                    bbox = bboxes(i, :);
        
                    % Create a Kalman filter object.
                    kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                        centroid, [200, 50], [100, 25], 100);
        
                    % Create a new track.
                    newTrack = struct(...
                        'id', nextId, ...
                        'bbox', bbox, ...
                        'kalmanFilter', kalmanFilter, ...
                        'age', 1, ...
                        'totalVisibleCount', 1, ...
                        'consecutiveInvisibleCount', 0);
        
                    % Add it to the array of tracks.
                    tracks(end + 1) = newTrack;
        
                    % Increment the next id.
                    nextId = nextId + 1;
                end
            end
            motionDetected = totalMotionDetected;
        end   
        if mean(MotionBasedMultiObjectTrackingExample(rvr)) > 0.5
            isMotion = true;
        else
            isMotion = false;
        end
    end
    function detected = runFacialDetection(network, rvr)
        h = waitbar(0,'Running Facial Detection');
        faceDetector=vision.CascadeObjectDetector;
        labels = [];
        for i = 1:11
            e=rvr.getImage();
            bboxes =step(faceDetector,e);
            if(sum(sum(bboxes))~=0)
                es=imcrop(e,bboxes(1,:));
                es=imresize(es,[227 227]);
                label=classify(network,es);
                labels = [labels label];
            else
                labels = [labels 'Face Not Detected'];
            end
            waitbar(i/11, h);
        end
        close(h);
        disp(labels)
        name = mode(labels);
        disp(name)
        if name == "Jonathan" || name == "Yoltic" || name == "Jacob"
            detected = true;
        else
            detected = false;
        end
    end
    function returned = trainFacialDetection()
        g=alexnet;
        layers=g.Layers;
        layers(23)=fullyConnectedLayer(4);
        layers(25)=classificationLayer;
        allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
        opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
        myNet1=trainNetwork(allImages,layers,opts);
        returned = myNet1;
    end
end

