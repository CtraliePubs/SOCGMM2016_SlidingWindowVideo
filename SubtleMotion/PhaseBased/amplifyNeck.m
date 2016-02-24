clear;


defaultPyrType = 'halfOctave'; % Half octave pyramid is default as discussed in paper
scaleAndClipLargeVideos = true; % With this enabled, approximately 4GB of memory is used

% Uncomment to use octave bandwidth pyramid: speeds up processing,
% but will produce slightly different results
%defaultPyrType = 'octave'; 

% Uncomment to process full video sequences (uses about 16GB of memory)
%scaleAndClipLargeVideos = false;

%% Throat
inFile = '../myneck.ogg';
samplingRate = 25; % Hz
loCutoff = 0.5;    % Hz
hiCutoff = 3;    % Hz
alpha = 30;    
sigma = 3;         % Pixels
pyrType = 'octave';

if (scaleAndClipLargeVideos)
    phaseAmplify(inFile, alpha, loCutoff, hiCutoff, samplingRate, '../','sigma', sigma,'pyrType', pyrType, 'scaleVideo', 2/3);
else
    phaseAmplify(inFile, alpha, loCutoff, hiCutoff, samplingRate, '../','sigma', sigma,'pyrType', pyrType, 'scaleVideo', 1);
end
