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
