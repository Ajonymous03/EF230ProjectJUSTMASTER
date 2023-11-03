function MASTER(rvr, cam, trainedNetwork, myVideo1)

    % Run both of these functions in the command window prior to running
    % the master code
    % cam = webcam;
    % trainedNetwork = trainFacialDetection();
    % myVideo1 = VideoWriter('cam1.avi');

    while true
        driveRobot(rvr, cam, trainedNetwork, myVideo1);
        pause(2);
    end
           
    function mainCode(rvr, cam, trainedNetwork, myVideo1)
    % Purpose: Runs the main block of code that occurs at every corner of
    %          the square the robot drives on 
    % Input: cam (the webcam attached to the laptop), trainedNetwork (the
    %        trained network used to detect whose face is in front of the
    %        camera), and rvr (the variable associated with the robot)
    % Output: NONE
    % Usage: mainCode(cam, trainedNetwork, rvr)
        detected = motionDetection(rvr, myVideo1);
        if detected == true
            playMotionDetectedAudio();
            faceScanCountdown();
            faceInSystem = runFacialDetection(trainedNetwork, cam);
                if faceInSystem
                    playFaceAcceptedAudio();
                else
                    inputPassword();
                end
        end
    end

    function faceScanCountdown()
    % Purpose: Increments a waitbar as a signal for when a person needs to
    %          be in front of the camera for facial recognition
    % Input: NONE
    % Output: NONE
    % Usage: faceScanCountdown()
        h = waitbar(0, 'Facial Recognition Occurs When Bar is Full');
        for i = 1:10
            pause(1);
            waitbar(i/10, h);
        end
        close(h)
    end

    function playMotionDetectedAudio()
    % Purpose: Play a sound letting the person know that motion was
    %          detected in the room and they need to proceed to get their
    %          face scanned to determine their identity
    % Input: NONE
    % Output: NONE
    % Usage: playMotionDetectedAudio()
        clear sound;
        [y, Fs] = audioread('motionDetected.mp3');
        sound(y, Fs, 16);
        pause(4.25);
    end

    function playFaceAcceptedAudio()
    % Purpose: Play a sound letting the person know that their facial
    %          recognition scan was successful and they are free to go
    %          about their business
    % Input: NONE
    % Output: NONE
    % Usage: playFaceAcceptedAudio()
        clear sound;
        [y, Fs] = audioread('scanSuccessful.mp3');
        sound(y, Fs, 16);
        pause(3.25);
    end

    function playAlarmDisabledAudio()
    % Purpose: Play a sound letting the person know that the alarm has been
    %          disabled
    % Input: NONE
    % Output: NONE
    % Usage playAlarmDisabledAudio()
        clear sound;
        [y, Fs] = audioread('alarmDisabled.mp3');
        sound(y, Fs, 16);
        pause(1.5);
    end

    function playAlarmSiren()
    % Purpose: Play the alarm that sounds when the password attempts fail
    % Input: NONE
    % Output: NONE
    % Usage: playAlarmSiren()
        clear sound;
        [y, Fs] = audioread('alarmSiren.mp3');
        sound(y, Fs, 16);
    end

    function playAttemptFailedAudio()
    % Purpose: Play an audio that lets the user know their password attempt
    %          failed
    % Input: NONE
    % Output: NONE
    % Usage: playAttemptFailedAudio()
        clear sound;
        [y, Fs] = audioread('attemptFailed.mp3');
        sound(y, Fs, 16);
        pause(2.75);
    end

    function playPasswordCorrectAudio()
    % Purpose: Play an audo that lets the user know their password attempt 
    %          was successful 
    % Input: NONE
    % Output: NONE
    % Usage: playPasswordCorrectAudio()
        clear sound;
        [y, Fs] = audioread('passwordCorrectAudio.mp3');
        sound(y, Fs, 16);
        pause(2.5);
    end

    function playNoMoreAttemptsAudio()
    % Purpose: Play an audio that lets the user know they have no more
    %          password attempts left and that the alarm will sound
    % Input: NONE
    % Output: NONE
    % Usage: playNoMoreAttemptsAudio()
        clear sound;
        [y, Fs] = audioread('noMoreAttempts.mp3');
        sound(y, Fs, 16); 
        pause(3.25);
    end

    function playTooManyFailedAttemptsAudio()
    % Purpose: Plays an audio that lets the user know that they have failed
    %          too many attempts and the password service will shut down
    %          till law enforcement arrives
    % Input: NONE
    % Output: NONE
    % Usage: playTooManyFailedAttemptsAudio()
        clear sound;
        [y, Fs] = audioroead('tooManyFailedAttempts.mp3')
        sound(y, Fs, 16);
        pause(7.5);
    end

    function inputPassword()
    % Purpose: Starts password input if face recognition fails
    % Input: NONE
    % Output: NONE
    % Usage: inputPassword()
        Attempt1 = inputdlg('Enter Password:'); % First password input
        Check = str2double(Attempt1);
        passwordCorrect = 124680; % Correct Password
        if Check == passwordCorrect % First password check
            playPasswordCorrectAudio();
        else
            playAttemptFailedAudio();
            Attempt2 = inputdlg('Enter Password (Try 2):'); % Second password input
            Check = str2double(Attempt2);
        end
        if Check == passwordCorrect % Second password check
            playPasswordCorrectAudio();
        else
            playAttemptFailedAudio();
            Attempt3 = inputdlg('Enter Password (Try 3):'); % Third password input
            Check = str2double(Attempt3);
        end
        if Check == passwordCorrect % Third Password Check
            playPasswordCorrectAudio();
        else
            playNoMoreAttemptsAudio();
            playAlarmSiren();
            endAlarm();
        end
    end

    function endAlarm()
    % Purpose: Ends the alarm sound if the disable password is correctly entered
    % Input: Value of panic function
    % Output: NONE
    % Usage: endAlarm()
        ADC = 123456;
        ADCattempt1 = inputdlg('Enter ADC (Try 1):'); % First password input
        Check = str2double(ADCattempt1);
        if Check == ADC % First password check
            playAlarmDisabledAudio();
        else
            ADCattempt2 = inputdlg('Enter ADC (Try 2):'); % Second password input
            Check = str2double(ADCattempt2);
        end
        if Check == ADC % Second password check
            playAlarmDisabledAudio();
        else
            ADCattempt3 = inputdlg('Enter ADC (Try 3):'); % Second password input
            Check = str2double(ADCattempt3);
        end
        if Check == ADC % Third password check
            playAlarmDisabledAudio();
        else
            
        end
    end

    function driveRobot(rvr, cam, trainedNetwork, myVideo1)
    % Purpose: Drive the robot in a square, checking for motion at each
    %          corner
    % Input: The variable the robot is assigned to, the webcam, and the
    %        trainedNetwork
    % Output: NONE
    % Usage: driveRobot(rvr)
        for i = 1:4
           rvr.setDriveSpeed(50,50) ;
           pause(1);
           rvr.turnAngle(94);
           pause(1);
           mainCode(rvr, cam, trainedNetwork, myVideo1)
        end
        rvr.stop;
    end

    function getVideo(rvr, myVideo1)
    % Purpose: Get video from the robot to run through motion detection
    %          code, video saves to current folder
    % Input: The variable the robot is assigned to
    % Output: NONE
    % Usage: getVideo(rvr), where rvr is the variable assigned to the
    %        robot
        h = waitbar(0,'Taking Video');
        open(myVideo1);
        disp('hello');
        totalFrames = 20; % Sets total number of frames to be taken for video
        for i=1:totalFrames
            img1 = rvr.getImage;
            disp(i);
            writeVideo(myVideo1, img1); % Adds the image taken to the video
            disp(i + 1);
            waitbar(i/20, h);
        end
        close(h);
        close(myVideo1);
    end

    function motionDetected = motionTracking(rvr, myVideo1)
    % Purpose: Analyze the video from getVideo(rvr) to determine if motion
    %          has occurred within the field of view of the camera
    % Input: The variable the robot is assigned to
    % Output: An array consisting of 1s and 0s, where a 1 indicates motion
    %          was detected within a frame, while a 0 indicates motion was
    %          not detected within a frame
    % Usage: motionDetected = motionTracking(rvr)
        getVideo(rvr, myVideo1); % Collects video to be analyzed using getVideo function
    
        obj = setupSystemObjects();
        
        tracks = initializeTracks(); % Create an empty array of tracks.
        
        nextId = 1; % ID of the next track
        totalMotionDetected = [];

        while hasFrame(obj.reader)
            frame = readFrame(obj.reader);
            [centroids, bboxes, ~] = detectObjects(frame);
            [a, ~, ~] = detectObjects(frame);
            if a > 0 % Adds a 1 to totalMotionDetected if there are 1 or more objects detected in a frame
                totalMotionDetected = [totalMotionDetected 1];
            else % Adds a 0 to totalMotionDetected if there are 0 objects detected in a frame
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
        % Purpose: Initializes video I/O
        % Input: NONE
        % Output: An object that reads a video and plays the video
        % Usage: obj = setupSystemObjects()
            obj.reader = VideoReader('cam1.avi'); % Creates a video reader
    
            obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]); % Creates a video player

            obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
                'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7); % Create System objects for foreground detection and blob analysis
    
            obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
                'AreaOutputPort', true, 'CentroidOutputPort', true, ...
                'MinimumBlobArea', 400); % Detects areas in the video that likely mean motion has occurred
        end

        function tracks = initializeTracks()
        % Purpose: Creates an empty array of tracks
        % Input: NONE
        % Output: The tracks structure with id, bbox, kalmanFilter, age,
        %          totalVisibleCount, and consecutiveInvisibleCount
        % Usage: tracks = initializeTracks()
            tracks = struct(...
                'id', {}, ...
                'bbox', {}, ...
                'kalmanFilter', {}, ...
                'age', {}, ...
                'totalVisibleCount', {}, ...
                'consecutiveInvisibleCount', {});
        end

        function [centroids, bboxes, mask] = detectObjects(frame)
        % Purpose: Finds connected components in a frame of a video, which
        %          can signify motion
        % Input: Frame from readFrame(obj.reader)
        % Output: 
        % Usage: [centroids, bboxes, mask] = detectObjects(frame)
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
        % Purpose: Predicts the new location of a track and shifts the
        %          tracks bounding box to the predicted location
        % Input: NONE
        % Output: NONE
        % Usage: predictNewLocationsOfTracks()
            for i = 1:length(tracks)
                bbox = tracks(i).bbox;
    
                predictedCentroid = predict(tracks(i).kalmanFilter); % Predict the current location of the track.
    
                % Shift the bounding box so that its center is at the predicted location.
                predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
                tracks(i).bbox = [predictedCentroid, bbox(3:4)];
            end
        end
    
        function [assignments, unassignedTracks, unassignedDetections] = ...
                detectionToTrackAssignment()
        % Purpose: Computes
        % Input: NONE
        % Output: Array that includes the assignments (an Lx2 matrix of
        %         index pairs and detections), unassignedTracks (P-element
        %         where P is the number of unassigned tracks), and
        %         unassignedDirections (Q-element vector where Q is the
        %         number of unassigned detections)
        % Usage: [assignments, unassignedTracks, unassignedDetections] = ... 
        %        detectionToTrackAssignment()
    
            nTracks = length(tracks);
            nDetections = size(centroids, 1);
    
            % Compute the cost of assigning each detection to each track.
            cost = zeros(nTracks, nDetections);
            for i = 1:nTracks
                cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
            end

            costOfNonAssignment = 20;
            [assignments, unassignedTracks, unassignedDetections] = ...
                assignDetectionsToTracks(cost, costOfNonAssignment); % Solve the assignment problem.
        end

        function updateAssignedTracks()
        % Purpose: Updates assigned tracks with the estimated location and
        %          replaces the bounding box with the new estimated one
        % Input: NONE
        % Output: NONE
        % Usage: updateAssignedTracks()
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
    
                tracks(trackIdx).age = tracks(trackIdx).age + 1; % Update track's age.
    
                tracks(trackIdx).totalVisibleCount = ...
                    tracks(trackIdx).totalVisibleCount + 1; % Update visibility.
                tracks(trackIdx).consecutiveInvisibleCount = 0;
            end
        end

        function updateUnassignedTracks()
        % Purpose: Updates the age and consecutiveInvisibleCount of
        %          unassigned tracks
        % Input: NONE   
        % Output: NONE
        % Usage: updateUnassignedTracks()
            for i = 1:length(unassignedTracks)
                ind = unassignedTracks(i);
                tracks(ind).age = tracks(ind).age + 1;
                tracks(ind).consecutiveInvisibleCount = ...
                    tracks(ind).consecutiveInvisibleCount + 1;
            end
        end

        function deleteLostTracks()
        % Purpose: Deletes tracks of motion that have been lost out of
        %          frame or have not moved in some time
        % Input: NONE
        % Output: NONE
        % Usage: deleteLostTracks()
            if isempty(tracks)
                return;
            end
    
            invisibleForTooLong = 20;
            ageThreshold = 8;
   
            ages = [tracks(:).age];
            totalVisibleCounts = [tracks(:).totalVisibleCount];
            visibility = totalVisibleCounts ./ ages; % Compute the fraction of the track's age for which it was visible.
    
            lostInds = (ages < ageThreshold & visibility < 0.6) | ...
                [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong; % Find the indices of 'lost' tracks.
    
            tracks = tracks(~lostInds); % Delete lost tracks.
        end

        function createNewTracks()
        % Purpose: Create new tracks for tracking motion in frame
        % Input: NONE
        % Output: NONE
        % Usage: createNewTracks()
            centroids = centroids(unassignedDetections, :);
            bboxes = bboxes(unassignedDetections, :);
    
            for i = 1:size(centroids, 1)
    
                centroid = centroids(i,:);
                bbox = bboxes(i, :);
    
                kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                    centroid, [200, 50], [100, 25], 100); % Create a Kalman filter object.
    
                newTrack = struct(...
                    'id', nextId, ...
                    'bbox', bbox, ...
                    'kalmanFilter', kalmanFilter, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0); % Create a new track.
    
                tracks(end + 1) = newTrack; % Add it to the array of tracks.
    
                nextId = nextId + 1;
            end
        end
        motionDetected = totalMotionDetected;
    end   

    function isMotion = motionDetection(rvr, myVideo1)
    % Purpose: Detect whether there is motion within the room
    % Input: The variable the robot is assigned to
    % Output: Returns true if motion is detected in over 50% of frames, returns false if motion
    %          is not detected in over 50% of frames
    % Usage: isMotion = motionDetection(rvr)
        if mean(motionTracking(rvr, myVideo1)) > 0.5
            isMotion = true;
        else
            isMotion = false;
        end
    end

    function detected = runFacialDetection(network, cam)
    % Purpose: Run facial detection code to determine if the person in
    %          front of the webcam is within the database
    % Input: (network, cam) - network is the trained network used to
    %         determine faces, cam is the webcam connected to the device
    % Output: Returns true if the face detected is in the database, returns
    %          false otherwise
    % Usage: detected = runFacialDetection(network, cam)
        h = waitbar(0,'Running Facial Detection');
        faceDetector=vision.CascadeObjectDetector; % Creates a vision object detector
        labels = []; % Initializes list used to store names of face detected in photo
        for i = 1:11
            e = snapshot(cam);
            bboxes =step(faceDetector,e); % Detects a face and assigns it a bounding box
            if(sum(sum(bboxes))~=0)
                es=imcrop(e,bboxes(1,:));
                es=imresize(es,[227 227]);
                label=classify(network,es); % Determines whose face is in the photo and adds that label to labels
                labels = [labels label];
            else
                labels = [labels 'Face Not Detected']; % Adds 'Face Not Detected' if the detected face does not match any in the database
            end
            waitbar(i/11, h);
        end
        close(h);
        name = mode(labels); % Determines the most common item in labels
        if name == "Jonathan" || name == "Yoltic" || name == "Jacob"
            detected = true; % Returns true if the most commonly detected image in labels is someone in the database
        else
            detected = false; % Returns false if the most commonly detected image in labels is not someone in the database
        end
    end
end
