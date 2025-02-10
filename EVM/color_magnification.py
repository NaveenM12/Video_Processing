# Relevant imports
import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image 

#Path to image
DATA_PATH = r"/Users/naveenmirapuri/VideoProcessing/test_videos/"
VIDEO_NAME = "face.mp4"
VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_NAME)

print(os.path.exists(VIDEO_PATH))

# ------ EVM Hyperparameters for BPM ------
# video magnification factor
ALPHA = 50.0

# Gaussian Pyramid Level of which to apply magnfication
LEVEL = 4

# Temporal Filter parameters
f_lo = 50/60
f_hi = 60/60

# OPTIONAL: override fs
MANUAL_FS = None
VIDEO_FS = None

# video frame scale factor
SCALE_FACTOR = 1.0

# ------ Color Conversion ------
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq


def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


inv_colorspace = lambda x: cv2.normalize(
    yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)


## ------ Extract Video Frames ------
frames = [] # frames for processing
cap = cv2.VideoCapture(VIDEO_PATH)

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
        w = int(og_w*SCALE_FACTOR)
        h = int(og_h*SCALE_FACTOR)

    # convert normalized uint8 BGR to the desired color space
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = bgr2yiq(np.float32(frame/255))

    # append resized frame
    frames.append(cv2.resize(frame, (w, h)))

    idx += 1
    
    
cap.release()
cv2.destroyAllWindows()
del cap

NUM_FRAMES = len(frames)
print(f"Number of frames: {NUM_FRAMES}")

print(f"Detected Video Sampling rate: {fs}")

# ------ IF Override Sampling Rate ------
if MANUAL_FS:
    print(f"Overriding to: {MANUAL_FS}")
    fs = MANUAL_FS
    VIDEO_FS = fs
else:
    VIDEO_FS = fs


# ------ Design FIR Filter------
bandpass = signal.firwin(numtaps=NUM_FRAMES,
                         cutoff=(f_lo, f_hi),
                         fs=fs,
                         pass_zero=False)

transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

# ------ Plot Transfer Function, Impulse Response, Frequency Response ------
plt.plot(np.abs(transfer_function))
plt.title("Transfer Function");
plt.savefig("EVM/generated_images/transfer_function.png")
plt.clf()

plt.plot(bandpass)
plt.title("Impulse Response");
plt.savefig("EVM/generated_images/impulse_response.png")
plt.clf()

norm_freqs, response = signal.freqz(bandpass)
freqs = norm_freqs / np.pi * fs/ 2 

_, ax = plt.subplots(2, 1, figsize=(15, 7))
ax[0].plot(freqs, 20*np.log10(np.abs(response)));
ax[0].plot([f_lo, f_lo], [-100, -10], color='m')
ax[0].plot([f_hi, f_hi], [-100, -10], color='m')
ax[0].set_title("Frequency Response");
ax[0].set_ylabel("Amplitude");

ax[1].plot(freqs, np.angle(response));
ax[1].set_title("Phase Response");
ax[1].set_xlabel("Freqeuncy (Hz)");
ax[1].set_ylabel("Angle (radians)");
plt.savefig("EVM/generated_images/frequency_response.png")
plt.clf()

# ------ Build Gaussian Pyramids ------
def gaussian_pyramid(image, level):
    """ Obtains single band of a Gaussian Pyramid Decomposition
        Inputs: 
            image - single channel input image
            num_levels - number of pyramid levels
        Outputs:
            pyramid - Pyramid decomposition tensor
        """ 
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    for i in range(0, level):
        # image = cv2.pyrDown(image)

        image = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        rows, cols, _ = image.shape

        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = image[:, :, c]

    return pyramid

# get Gaussian Pyramid Level
rows, cols, colors = frames[0].shape
scale = 2**LEVEL
pyramid_stack = np.zeros((NUM_FRAMES, colors, rows//scale, cols//scale))

for i, frame in enumerate(frames):
    pyramid = gaussian_pyramid(frame, LEVEL)
    pyramid_stack[i, :, :, :] = pyramid

plt.imshow(pyramid_stack[0, :, :, :].transpose(1, 0, 2).reshape((pyramid.shape[1], -1)), cmap='gray');
plt.savefig("EVM/generated_images/gaussian_blur.png")
plt.clf()

# ------ Apply Temporal Filter ------
pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
_filtered_pyramid = pyr_stack_fft * transfer_function[:, None, None, None].astype(np.complex64)
filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

# ------ Plots and Explorations ------
print(pyr_stack_fft.shape)

_, ax = plt.subplots(2, 1, figsize=(10, 5), sharey=True)
ax[0].plot(np.abs(pyr_stack_fft[2:-2, 0, 20, 12]))
ax[0].set_title("Unfiltered Signal at (20, 12)")
ax[1].plot(np.abs(_filtered_pyramid[2:-2, 0, 20, 12]))
ax[1].set_title("Filtered Signal at (20, 12)");
plt.tight_layout();
plt.savefig("EVM/generated_images/filtered_signal.png")
plt.clf()

_, ax = plt.subplots(1, 2)
ax[0].imshow(pyramid_stack[50, 0, :, :], cmap='gray')
ax[0].set_title("Unfiltered Luma Channel")
ax[1].imshow(filtered_pyramid[50, 0, :, :], cmap='gray')
ax[1].set_title("Filtered Luma Channel");
plt.savefig("EVM/generated_images/filtered_luma_channel.png")
plt.clf()

plt.plot(pyramid_stack[:, 0, 12, 20] - pyramid_stack[:, 0, 12, 20].mean())
plt.plot(filtered_pyramid[:, 0, 12, 20]);
plt.savefig("EVM/generated_images/filtered_pyramid_comparison.png")
plt.clf()

# ------ Magnify and Reconstruct Video ------
magnified_pyramid = filtered_pyramid * ALPHA

magnified = []
magnified_only = []

for i in range(NUM_FRAMES):
    y_chan = frames[i][:, :, 0]
    i_chan = frames[i][:, :, 1] 
    q_chan = frames[i][:, :, 2] 
    
    fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (cols, rows))
    fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (cols, rows))
    fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (cols, rows))

    # apply magnification
    mag = np.dstack((
        y_chan + fy_chan,
        i_chan + fi_chan,
        q_chan + fq_chan,
    ))
    
    # normalize and convert to RGB
    mag = inv_colorspace(mag)

    # store magnified frames
    magnified.append(mag)

    # store magified only for reference
    magnified_only.append(np.dstack((fy_chan, fi_chan, fq_chan)))

# ------ Plot Magnification ------
og_reds = []
og_blues = []
og_greens = []

reds = []
blues = []
greens = []
for i in range(NUM_FRAMES):
    # convert YIQ to RGB
    frame = inv_colorspace(frames[i])
    og_reds.append(frame[0, :, :].sum())
    og_blues.append(frame[1, :, :].sum())
    og_greens.append(frame[2, :, :].sum())

    reds.append(magnified[i][0, :, :].sum())
    blues.append(magnified[i][1, :, :].sum())
    greens.append(magnified[i][2, :, :].sum())

times = np.arange(0, NUM_FRAMES)/fs

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax[0].plot(times, og_reds, color='red')
ax[0].plot(times, og_blues, color='blue')
ax[0].plot(times, og_greens, color='green')
ax[0].set_title("Original", size=18)
ax[0].set_xlabel("Time", size=16)
ax[0].set_ylabel("Intensity", size=16)

ax[1].plot(times, reds, color='red')
ax[1].plot(times, blues, color='blue')
ax[1].plot(times, greens, color='green')
ax[1].set_title("Filtered", size=18)
ax[1].set_xlabel("Time", size=16);
plt.savefig("EVM/generated_images/magnification_comparison.png")
plt.clf()

freqs = np.fft.rfftfreq(NUM_FRAMES) * fs
rates = np.abs(np.fft.rfft(reds))/NUM_FRAMES
plt.plot(freqs[1:], rates[1:]);
plt.title("DFT of Red channel Intensities")
plt.xlabel("Freuqency")
plt.ylabel("Amplitude");
plt.savefig("EVM/generated_images/red_channel_fourier.png")
plt.clf()

# ------ Save Magnified Video ------






# ------ Reconstruct Video ------



