import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
from matplotlib import cm
import torch.nn as nn



from utils.steerable_pyramid import SteerablePyramid, SuboctaveSP
from utils.phase_utils import *
from utils.pyramid_utils import build_level, build_level_batch, recon_level_batch

# ----- Load and scale video -----
video_path = "test_videos/crane_crop.avi"

scale_factor = 0.75

# ----- Initialize video reader -----
bgr_frames = []
frames = [] # frames for processing
cap = cv2.VideoCapture(video_path)

# video sampling rate
fs = cap.get(cv2.CAP_PROP_FPS)

idx = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    if idx == 0:
        og_h, og_w, _ = frame.shape
        w = int(og_w*scale_factor)
        h = int(og_h*scale_factor)

    # store original frames
    bgr_frames.append(frame)

    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255)
    yiq = rgb2yiq(rgb)

    # append resized Luma channel 
    frames.append(cv2.resize(yiq[:, :, 0], (w, h)))

    idx += 1
    
    
cap.release()
cv2.destroyAllWindows()
del cap

# ----- Luma Channel Frames Image -----
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Test Frames (Luma Channel)", size=18)
ax[0].imshow(frames[0], cmap='gray')
ax[0].set_title("Frame: 0")
ax[1].imshow(np.float32(frames[1]) - np.float32(frames[0]), cmap='gray');
ax[1].set_title("Frame Delta: 1 - 0")

plt.savefig("Phase_Based_Magnification/generated_images/luma_channel_frames.png")
plt.clf()


# ----- Set Hyperparameters -----
# use first frame as the reference frame
ref_idx = 0
ref_frame = frames[ref_idx]
h, w = ref_frame.shape

# video length
num_frames = len(frames)

# factor to avoid division by 0
eps = 1e-6

# CUDA Parallelization
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Bandpass Filter Frequencies
f_lo = 0.2 
f_hi = 0.25 

# weighted amplitude blur parameters
sigma = 5.0 

# attenuate other frequencies
attenuate = True

# phase magnification factor
phase_mag = 25.0

# ----- Gaussian Kernel for Amplitude Weighted Blurring -----
# ensure ksize is odd or the filtering will take too long
# see warning in: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
ksize = np.max((3, np.ceil(4*sigma) - 1)).astype(int)
if ((ksize % 2) != 1):
    ksize += 1

# get Gaussian Blur Kernel for reference only
gk = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
gauss_kernel = torch.tensor(gk @ gk.T).type(torch.float32) \
                                      .to(device) \
                                      .unsqueeze(0) \
                                      .unsqueeze(0)

# check that the Gaussian Kernel dimensions are odd
print(gauss_kernel.shape)   

plt.imshow(gk @ gk.T);
plt.title("Gaussian Kernel");
plt.savefig("Phase_Based_Magnification/generated_images/gaussian_kernel.png")
plt.clf()

# ----- Get Temporal Filter -----
from scipy import signal

# normalize freqeuncies to the Nyquist rate range of [0, 1]
norm_f_lo = f_lo / fs * 2
norm_f_hi = f_hi / fs * 2

# Get Bandpass Impulse Response
bandpass = signal.firwin(numtaps=len(frames), 
                         cutoff=[norm_f_lo, norm_f_hi], 
                         pass_zero=False)

# Get Frequency Domain Transfer Function
transfer_function = torch.fft.fft(
    torch.fft.ifftshift(torch.tensor(bandpass))).to(device) \
                                                .type(torch.complex64)
transfer_function = torch.tile(transfer_function, 
                               [1, 1, 1, 1]).permute(0, 3, 1, 2)

# ----- Plot Impulse Response -----
plt.plot(bandpass)
plt.title("Impulse Response");
plt.savefig("Phase_Based_Magnification/generated_images/impulse_response.png")
plt.clf()

# ----- Frequency vs Phase Response Graph -----
norm_freqs, response = signal.freqz(bandpass)
freqs = norm_freqs / np.pi * fs/ 2 

_, ax = plt.subplots(2, 1, figsize=(15, 7))
ax[0].plot(freqs, 20*np.log10(np.abs(response)));
ax[0].set_title("Frequency Response");
ax[0].set_ylabel("Amplitude");

ax[1].plot(freqs, np.angle(response));
ax[1].set_title("Phase Response");
ax[1].set_xlabel("Freqeuncy (Hz)");
ax[1].set_ylabel("Angle (radians)");
plt.savefig("Phase_Based_Magnification/generated_images/frequency_vs_phase_response.png")

# ----- Get Complex Steerable Pyramid Filters -----
max_depth = int(np.floor(np.log2(np.min(np.array(ref_frame.shape)))) - 2)

# Regular Pyramids with SubOctave Filter Scaling
# csp = SteerablePyramid(depth=max_depth, orientations=4, filters_per_octave=1, twidth=1.0, complex_pyr=True)
csp = SteerablePyramid(depth=max_depth, orientations=8, filters_per_octave=2, twidth=0.75, complex_pyr=True)
# csp = SteerablePyramid(depth=max_depth, orientations=8, filters_per_octave=2, twidth=0.25, complex_pyr=True)

# SubOctave Pyramids
# csp = SuboctaveSP(depth=max_depth-1, orientations=8, filters_per_octave=2, cos_order=6, complex_pyr=True) 
# csp = SuboctaveSP(depth=max_depth, orientations=8, filters_per_octave=4, cos_order=6, complex_pyr=True) 
filters, crops = csp.get_filters(h, w, cropped=False)

print(f"Number of Pyramid Filters: {len(filters)}")

# ----- Display Frequency Partition of Oriented Sub-Band Filters -----
filter_partition = np.dstack(filters[1:-1]).sum(axis=-1)

plt.imshow(filter_partition, cmap='jet');
plt.title("Sub-Band Filter Partitions");
plt.colorbar();
plt.savefig("Phase_Based_Magnification/generated_images/sub_band_filter_partitions.png")
plt.clf()

# ----- Display Color-Coded Sub-Band Filter Partitions -----
cmap = cm.get_cmap('gist_rainbow', len(filters) - 2)
color_filter_partition = np.zeros_like(filter_partition[:, :, None].repeat(3, axis=2))

for i in range(1, len(filters[1:-1])):
    # scramble the colors!
    tmp_filt = filters[i][:, :, None].repeat(3, axis=2) * np.array(cmap(i)[:3])

    # scramble the colors!
    # blue = int(i*30 % 256)/255
    # green = int(i*103 % 256)/255
    # red = int(i*50 % 256)/255

    # tmp_filt = filters[i][:, :, None].repeat(3, axis=2) * 2*np.array([red, blue, green])
    
    color_filter_partition += tmp_filt/3

plt.imshow(color_filter_partition)
plt.title("Sub-Band Filter Partitions");
plt.savefig("Phase_Based_Magnification/generated_images/sub_band_filter_partitions_color.png")

# ----- Display Color-Coded Sub-Band Filter Partitions -----
_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(filter_partition, cmap='jet')
ax[0].set_title("Sub-Band Filter Partitions")

ax[1].imshow(color_filter_partition)
ax[1].set_title("Colored Sub-Band Filter Partitions");

plt.savefig("Phase_Based_Magnification/generated_images/sub_band_filter_partitions_color_comparison.png")
plt.clf()

# ----- Filters Tensor -----
filters_tensor = torch.tensor(np.array(filters)).type(torch.float32).to(device)
frames_tensor = torch.tensor(np.array(frames)).type(torch.float32).to(device)

# ----- Compute Discrete Fourier Transform for each frame -----
video_dft = torch.fft.fftshift(torch.fft.fft2(frames_tensor, dim=(1,2))).type(torch.complex64).to(device)

# ----- Store DFT of motion magnified frames -----
recon_dft = torch.zeros((len(frames), h, w), dtype=torch.complex64).to(device)

# ----- Set up torch filter -----
filter2D = nn.Conv2d(in_channels=1, out_channels=1,
                     kernel_size=gauss_kernel.shape[2:], 
                     padding='same',
                     padding_mode='circular',
                     groups=1, 
                     bias=False)

filter2D.weight.data = gauss_kernel
filter2D.weight.requires_grad = False

# ----- Loop over pyramid -----
phase_deltas = torch.zeros((batch_size, len(frames), h, w), dtype=torch.complex64).to(device)

for level in range(1, len(filters) - 1, batch_size):

    # get batch indices
    idx1 = level
    idx2 = level + batch_size

    # get current filter batch
    filter_batch = filters_tensor[idx1:idx2]

    ## get reference frame pyramid and phase (DC)
    ref_pyr = build_level_batch(video_dft[ref_idx, :, :].unsqueeze(0), filter_batch)
    ref_phase = torch.angle(ref_pyr)

    ## Get Phase Deltas for each frame
    for vid_idx in range(num_frames):

        curr_pyr = build_level_batch(
                        video_dft[vid_idx, :, :].unsqueeze(0), filter_batch)

        # unwrapped phase delta
        _delta = torch.angle(curr_pyr) - ref_phase 

        # get phase delta wrapped to [-pi, pi]
        phase_deltas[:, vid_idx, :, :] = ((torch.pi + _delta) \
                                            % 2*torch.pi) - torch.pi
    
    ## Temporally Filter the phase deltas
    # Filter in Frequency Domain and convert back to phase space
    phase_deltas = torch.fft.ifft(transfer_function \
                                  * torch.fft.fft(phase_deltas, dim=1),  
                                  dim=1).real

    ## Apply Motion Magnifications
    for vid_idx in range(num_frames):

        curr_pyr = build_level_batch(video_dft[vid_idx, :, :].unsqueeze(0), filter_batch)
        delta = phase_deltas[:, vid_idx, :, :]

        ## Perform Amplitude Weighted Blurring
        if sigma != 0:
            amplitude_weight = torch.abs(curr_pyr) + eps
            
            # Torch Functional Approach (faster)
            weight = F.conv2d(input=amplitude_weight.unsqueeze(1), 
                              weight=gauss_kernel, 
                              padding='same').squeeze(1)
            
            delta = F.conv2d(input=(amplitude_weight * delta).unsqueeze(1), 
                              weight=gauss_kernel, 
                              padding='same').squeeze(1) 

            # Torch nn approach with circular padding (SLOWER)
            # weight = filter2D(amplitude_weight.unsqueeze(1)).squeeze(1)
            # delta = filter2D((amplitude_weight * delta).unsqueeze(1)).squeeze(1)
            
            # get weighted Phase Deltas
            delta /= weight

        ## Modify phase variation
        modifed_phase = delta * phase_mag

        ## Attenuate other frequencies by scaling magnitude by reference phase
        if attenuate:
            curr_pyr = torch.abs(curr_pyr) * (ref_pyr/torch.abs(ref_pyr)) 

        ## apply modified phase to current level pyramid decomposition
        # if modified_phase = 0, then no change!
        curr_pyr = curr_pyr * torch.exp(1.0j*modifed_phase) # ensures correct type casting

        ## accumulate reconstruced levels
        recon_dft[vid_idx, :, :] += recon_level_batch(curr_pyr, filter_batch).sum(dim=0)




# ----- Reconstruct Video ----- 

# ----- Add Back Lo and Hi Pass Components -----

# adding hipass component seems to cause bad artifacts and leaving
# it out doesn't seem to impact the overall quality
hipass = filters_tensor[0]
lopass = filters_tensor[-1]

## add back lo and hi pass components
for vid_idx in range(num_frames):
    # accumulate Lo Pass Components
    curr_pyr_lo = build_level(video_dft[vid_idx, :, :], lopass)
    dft_lo = torch.fft.fftshift(torch.fft.fft2(curr_pyr_lo))
    recon_dft[vid_idx, :, :] += dft_lo*lopass

    # # OPTIONAL accumulate Lo Pass Components
    # curr_pyr_hi = build_level(video_dft[vid_idx, :, :], hipass)
    # dft_hi = torch.fft.fftshift(torch.fft.fft2(curr_pyr_hi)) 
    # recon_dft[vid_idx, :, :] += dft_hi*hipass

# ----- Show DFT Magnitude of Magnified Video -----
plt.imshow(np.log(np.abs(recon_dft[10, :, :].cpu()) + 1e-6));
plt.title("DFT Magnitude of Reconstructed Video");
plt.savefig("Phase_Based_Magnification/generated_images/dft_magnitude_maagnified.png")
plt.clf()

# ----- Inverse DFT of results DFT -----
result_video = torch.fft.ifft2(torch.fft.ifftshift(recon_dft, dim=(1,2)), dim=(1,2)).real

result_video = result_video.cpu()

# ----- Check Reconstruction Error -----
print(torch.sum(torch.abs(frames_tensor[0, :, :].cpu() - result_video[0, :, :])),
      torch.mean(torch.square(frames_tensor[0, :, :].cpu() - result_video[0, :, :]).float()))

# ----- Check first frame -----
plt.imshow(result_video[0, :, :], cmap='gray');
plt.title("Reconstructed Video");
plt.savefig("Phase_Based_Magnification/generated_images/reconstructed_video.png")
plt.clf()

# ----- Reconstruct with RGB Channels ----- 
rgb_modified = []

for vid_idx in range(num_frames):
    # get current OG YIQ frame and resized result Luma Frame
    rgb = np.float32(cv2.cvtColor(bgr_frames[vid_idx], cv2.COLOR_BGR2RGB)/255)
    yiq_frame = rgb2yiq(rgb)

    result_frame = cv2.resize(result_video[vid_idx, :, :].numpy(), (og_w, og_h))

    # modify YIQ frame with motion magnified Luma channel
    yiq_frame[:, :, 0] = result_frame

    # convert to rgb
    rgb_frame = yiq2rgb(yiq_frame)

    # normalize
    rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=255, beta=0)

    rgb_modified.append(rgb_frame)


# ----- Save Reconstructed Video -----
stacked_frames = []
middle = np.zeros((og_h, 3, 3)).astype(np.uint8)

for vid_idx in range(len(frames)):
    frame = np.hstack((bgr_frames[vid_idx], 
                       middle, 
                       cv2.cvtColor(rgb_modified[vid_idx], cv2.COLOR_RGB2BGR)))
    stacked_frames.append(frame)

stacked_frames[0].shape

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(stacked_frames[-1], cv2.COLOR_BGR2RGB));
plt.savefig("Phase_Based_Magnification/generated_images/reconstructed_frame_rgb.png")
plt.clf()

# get width and height for video frames
_h, _w, _ = stacked_frames[-1].shape

# save to mp4
out = cv2.VideoWriter(f"Phase_Based_Magnification/generated_videos/crane_stacked_{int(phase_mag)}x.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))
 
for frame in stacked_frames:
    out.write(frame)

out.release()
del out



# ----- Bonus: Motion Magnification Comparisons -----
stacked_rgb = []
for frame in stacked_frames:
    stacked_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

stacked_rgb = np.array(stacked_rgb)
idx = 220 # horizontal index
v1 = 45   # vertical index 1
v2 = 135  # vertical index 2


idx = 120 # horizontal index
v1 = 145   # vertical index 1
v2 = 265  # vertical index 2

h_offset = bgr_frames[0].shape[1]
idx2 = idx + h_offset

# ----- Show where we want to test magnification -----
sample_image = stacked_rgb[0, ...].copy()
cv2.line(sample_image, (idx, v1), (idx, v2), (0,255,0), 5)
cv2.line(sample_image, (idx2, v1), (idx2, v2), (0,255,0), 5)
plt.imshow(sample_image)
plt.title("Frame 0");
plt.savefig("Phase_Based_Magnification/generated_images/video_magnification_selection.png")
plt.clf()


# ----- Magnification of Region 1 -----
fig, ax = plt.subplots(1, 2, figsize=(7, 5))

fig.suptitle("Motion Magnification across all frames")
ax[0].imshow(stacked_rgb[:, v1:v2, idx, :]) # .transpose(1, 0, 2))
ax[0].set_title("Original")
ax[1].imshow(stacked_rgb[:, v1:v2, idx2, :])
ax[1].set_title("Motion Magnified");
plt.savefig("Phase_Based_Magnification/generated_images/video_magnification_region_1.png")
plt.clf()


# ----- Magnification of Region 2 -----
fig, ax = plt.subplots(1, 2, figsize=(7, 5))

fig.suptitle("Motion Magnification across all frames")
ax[0].imshow(stacked_rgb[:, v1:v2, idx, :]) # .transpose(1, 0, 2))
ax[0].set_title("Original")
ax[1].imshow(stacked_rgb[:, v1:v2, idx2, :])
ax[1].set_title("Motion Magnified");
plt.savefig("Phase_Based_Magnification/generated_images/video_magnification_region_2.png")
plt.clf()


















