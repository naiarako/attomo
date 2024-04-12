%% Example 2D k-Wave

clear all; clc; close all;

% simulation settings
DATA_CAST       = 'gpuArray-single';     % set to 'single' or 'gpuArray-single' to speed up computations


%% CREATE MESHGRID

% set the size of the perfectly matched layer (PML)
pml_x_size = 20;                % [grid points]
pml_y_size = 20;                % [grid points]

Nx = 2048 - 2*pml_x_size;     % [grid points]
Ny = 2200  - 2*pml_y_size;     % [grid points] 
dx = 25e-6;  % # grid size in x
dy = 25e-6;  % # grid size in y

kgrid = kWaveGrid(Nx, dx, Ny, dy);  % create grid class

%% BASIC MEDIUM PROPERTIES

c0 = 1540; % [m/s]
alpha_coeff0 = 0.5; % [dB/(MHz^y cm)]
alpha_power0 = 1.0; 
rho0 = 1000; % [kg/m^3]
medium.alpha_mode = 'no_dispersion';

rng(20)

for ireal = 1:5

% define a random distribution of scatterers for the medium
background_map_mean = 1;
background_map_std = 0.008; 
scat_locs = randn([Nx, Ny]);
scat_locs2 = randn([Nx, Ny]);
background_map = background_map_mean + background_map_std * scat_locs;

% define properties
sound_speed_map = c0 * ones(Nx, Ny);
density_map = rho0 * ones(Nx, Ny).* background_map;
alpha_coeff_map = alpha_coeff0 * ones(Nx, Ny);

% make circular attenuating inclusion 
region1 = makeDisc(Nx, Ny, 610, 0, 200, 'False');
alpha_coeff_map(region1 == 1) = 1.0;

% assign to the medium inputs
medium.sound_speed = sound_speed_map;
medium.density = density_map;
medium.alpha_coeff = alpha_coeff_map;
medium.alpha_power = alpha_power0;

% create the axis variables
x_axis = [0, Nx * dx * 1e3];    % [mm]
y_axis = [0, Ny * dy * 1e3];    % [mm]


% plot the distribution of attenuation coefficient in the medium
figure;
imagesc(y_axis, x_axis, medium.alpha_coeff);
axis image;
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');
title('Attenuation coefficient');
colorbar;

% create the time array
t_end = (Nx * dx) * 2.2 / c0;   % [s]
kgrid.makeTime(c0, 0.6, t_end);


%% Define probe and input signal

% define source mask for a linear transducer 
pts_pitch = 8;                          % points per pitch
num_elements = 256*pts_pitch;           % [grid points]
x_offset = 10;                          % [grid points] (depth of transducer)
source.u_mask = zeros(Nx, Ny);
start_index = Ny/2 - round(1*num_elements/2) + 1;
source.u_mask(x_offset, start_index:1:start_index + 1*num_elements - 1) = 1;

% define the properties of the emitted tone burst 
sampling_freq = 1/kgrid.dt;     % [Hz]
element_spacing = dy;           % [m]
tone_burst_freq = 5e6;          % [Hz]
tone_burst_cycles = 7;
time_offset = 920;              %[time samples]

% create an element index relative to the centre element of the transducer
element_index = -(num_elements/pts_pitch - 1)/2:(num_elements/pts_pitch - 1)/2;

% Define sensor
sensor.record = {'p'};
sensor.mask = zeros(Nx, Ny);
sensor.mask(x_offset, start_index:1:start_index + num_elements - 1) = 1;


%% Define transmit angles and run all:

% range of steering angles 
steering_angles = -30:0.5:30;

% preallocate output
nangles = length(steering_angles);
scan_lines = zeros(kgrid.Nt, num_elements/pts_pitch, nangles);

% assign the input options
input_args = {'DisplayMask', source.u_mask,'PlotScale', [-5e4, 5e4],... 
    'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], 'PlotSim', false, ...
    'DataCast', DATA_CAST,'PMLAlpha',4, 'ScaleSourceTerms', false};


for itx = 1:nangles
    
    steering_angle = steering_angles(itx)
    
    % use geometric beam forming to calculate the tone burst offsets for each
    % transducer element based on the element index
    tone_burst_offset = time_offset + element_spacing * pts_pitch * element_index * ...
        sin(steering_angle * pi/180) / (c0 * kgrid.dt);
    
    % create the tone burst signals (same delay for all points in each element)
    source.ux = repelem(toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles, ...
        'SignalOffset', tone_burst_offset, 'Envelope', 'Gaussian'), pts_pitch, 1);

    source.u_mode = 'additive-no-correction';
    
    % run the simulation
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    
    % avearge recordings in all points of each element
    scan_lines(:,:,itx) = sensor_data.p(1:pts_pitch:end,:)' + sensor_data.p(2:pts_pitch:end,:)' ...
	    + sensor_data.p(3:pts_pitch:end,:)' + sensor_data.p(4:pts_pitch:end,:)' ...
	    + sensor_data.p(5:pts_pitch:end,:)' + sensor_data.p(6:pts_pitch:end,:)' ...
	    + sensor_data.p(7:pts_pitch:end,:)' + sensor_data.p(8:pts_pitch:end,:)';
        
end

% define properties to save
properties.freq = tone_burst_freq;
properties.pitch = pts_pitch*element_spacing;
properties.c0 = c0;
properties.num_elements = num_elements/pts_pitch;
properties.t0 = (time_offset - 1) * kgrid.dt;
properties.dt =  kgrid.dt;
properties.angles = steering_angles;


folder = './';
filename = [folder, 'acquisition_', num2str(ireal),'.mat']


% save the scan lines to disk
save(filename, 'scan_lines', 'properties');

%end

end

