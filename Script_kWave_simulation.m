%% Example 2D k-Wave

clear all; clc; close all

% simulation settings
DATA_CAST       = 'gpuArray-single';     % set to 'single' or 'gpuArray-single' to speed up computations


%% CREATE MESHGRID

% set the size of the perfectly matched layer (PML)
pml_x_size = 20;                % [grid points]
pml_y_size = 20;                % [grid points]

range = 47.2e-3; % [m]

Nx = 512 - 2*pml_x_size;       % [grid points]
Ny = 1024  - 2*pml_y_size;     % [grid points] 
dx = range/Nx;                 % grid size in x
dy = 1e-4;                     % grid size in y

kgrid = kWaveGrid(Nx, dx, Ny, dy);  % create grid class

%% BASIC MEDIUM PROPERTIES

c0 = 1480;          % [m/s]
alpha_coeff0 = 0.5; % [dB/(MHz^y cm)] (alpha_coeff0 = 0 for calibration data)
alpha_power0 = 1.9; 
rho0 = 1000;        % [kg/m^3]

rng(20) 

for ireal = 1:5

% define a random distribution of scatterers for the medium
background_map_mean = 1;
background_map_std = 0.008; 
scat_locs = randn([Nx, Ny]);
background_map = background_map_mean + background_map_std * scat_locs;

% define properties
sound_speed_map = c0 * ones(Nx, Ny);
density_map = rho0 * ones(Nx, Ny).* background_map;
alpha_coeff_map = alpha_coeff0 * ones(Nx, Ny);

% make circular attenuating inclusion (comment this for calibration data)
region1 = makeDisc(Nx, Ny, 150, 0, 50, 'False');
alpha_coeff_map(region1 == 1) = 1.0;

% assign to the medium inputs
medium.sound_speed = sound_speed_map;
medium.density = density_map;
medium.alpha_coeff = alpha_coeff_map;
medium.alpha_power = alpha_power0;

% create the axis variables
x_axis = [0, Nx * dx * 1e3];    % [mm]
y_axis = [0, Ny * dy * 1e3];    % [mm]

% plot medium attenuation coefficient
figure;
imagesc(y_axis, x_axis, medium.alpha_coeff);
axis image;
xlabel('Horizontal Position [mm]');
ylabel('Depth [mm]');
title('Attenuation coefficient');
colorbar;

% create the time array
t_end = (Nx * dx) * 2.2 / c0;   % [s]
kgrid.makeTime(c0, [], t_end);

%% Define probe and input signal

% define source mask for a linear transducer with an odd number of elements  
num_elements = 256;      % [grid points]
x_offset = 1;            % [grid points] 
source.u_mask = zeros(Nx, Ny);
start_index = Ny/2 - round(2*num_elements/2) + 1;
source.u_mask(x_offset, start_index:2:start_index + 2*num_elements - 1) = 1;

% define the properties of the tone burst used to drive the transducer
sampling_freq = 1/kgrid.dt;     % [Hz]
element_spacing = 2*dy;         % [m]
tone_burst_freq = 3e6;          % [Hz]
tone_burst_cycles = 5;
time_offset = 600;              %[time samples]

% create an element index relative to the centre element of the transducer
element_index = -(num_elements - 1)/2:(num_elements - 1)/2;

% Define sensor
sensor.record = {'p'};
sensor.mask = source.u_mask;

%% Define transmit angles and run all:

% range of steering angles to test
steering_angles = -27.5:0.5:27.5;
nangles = length(steering_angles);

% preallocate the storage
scan_lines = zeros(kgrid.Nt, num_elements, nangles);

% assign the input options
input_args = {'DisplayMask', source.u_mask,'PlotScale', [-5e4, 5e4],... 
    'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], 'PlotSim', false, ...
    'DataCast', DATA_CAST,'PMLAlpha',4, 'ScaleSourceTerms', false};


for itx = 1:nangles
    
    steering_angle = steering_angles(itx)
    
    % use geometric beam forming to calculate the tone burst offsets for each
    % transducer element based on the element index
    tone_burst_offset = time_offset + element_spacing * element_index * ...
        sin(steering_angle * pi/180) / (c0 * kgrid.dt);
    
    % create the tone burst signals
    source.ux = toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles, ...
        'SignalOffset', tone_burst_offset, 'Envelope', 'Gaussian');

    source.u_mode = 'additive-no-correction';
    
    % run the simulation
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    
    scan_lines(:,:,itx) = sensor_data.p';

end

properties.freq = tone_burst_freq;
properties.pitch = element_spacing;
properties.c0 = c0;
properties.num_elements = num_elements;
properties.t0 = (time_offset - 1) * kgrid.dt;
properties.dt =  kgrid.dt;
properties.angles = steering_angles;


folder = './';
filename = [folder, 'acquisition_', num2str(ireal),'.mat']

% save the scan lines to disk
save(filename, 'scan_lines', 'properties');

end


