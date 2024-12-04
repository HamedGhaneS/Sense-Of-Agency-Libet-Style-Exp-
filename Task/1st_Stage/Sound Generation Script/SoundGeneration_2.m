% Author: Hamed Ghane
% Date: 5th May 2023
% This script generates and plays a sound composed of 10 sinusoidal 
% frequencies, each combined to create a complex waveform. The sound 
% is normalized to prevent clipping and then saved as a .wav file.
%
% Overview:
% - Sample rate: 22100 Hz
% - Duration: 500 ms
% - Frequencies used: 500 Hz, 1000 Hz, 1500 Hz, 2000 Hz, 
%   3000 Hz, 4000 Hz, 6000 Hz, 8000 Hz, 10000 Hz, 12000 Hz.
%
% Steps:
% 1. Define the sample rate and time vector.
% 2. Define the frequencies for the sinusoids.
% 3. Generate the sound by summing the sinusoids.
% 4. Normalize the sound to avoid distortion.
% 5. Play the sound using `soundsc`.
% 6. Save the sound to a file named 'sound.wav'.

Fs = 22100; % Sample rate in Hz
t = 0:1/Fs:0.5-1/Fs; % Time vector, for 500ms sound

% Define your 10 frequencies (in Hz)
frequencies = [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000];

% Create a sound that is a sum of 10 sinusoids with these frequencies
sound = zeros(size(t));
for i = 1:10
    sound = sound + sin(2*pi*frequencies(i)*t);
end
sound = sound / max(abs(sound)); % Normalize the sound

% Play the sound
soundsc(sound, Fs);

% Save the sound
audiowrite('sound.wav', sound, Fs);
