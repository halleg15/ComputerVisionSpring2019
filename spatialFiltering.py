import cv2
import numpy as np
import matplotlib.pyplot as plt


#display original image
img1 = cv2.imread('dogOriginal.jpg')
cv2.imshow('Test', img1)
cv2.waitKey()

#display original image
img1Noise = cv2.imread('dogOriginal.jpg')
cv2.imshow('Noise Dog', img1Noise)
cv2.waitKey()


#####################################################################################
#read image and resize by half (aka square filter)
#path = '\smt\PycharmProjects\CVproj1\dogOriginal.JPG'
img = cv2.imread('dogOriginal.jpg')
small = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Small',small)
cv2.waitKey()

#again but with noisy dog picture
imgNoise = cv2.imread('dogNoisy.jpg')
smallNoise = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Small Noise',smallNoise)
cv2.waitKey()

img2 = img / img.max()
cv2.imshow('g',img2.astype(np.uint8))

#####################################################################################
#apply filter to any color image and have resulting image also be
# in color

#####################################################################################
#smoothing & denoising (gaussian and median filters)

#gaussian
g = cv2.GaussianBlur(small, (75,75), 3)
g2 = g/g.max()*255
cv2.imshow('g', g2.astype(np.uint8))
cv2.waitKey(0)

#median
median = cv2.medianBlur(small,5)
cv2.imshow('median',median)
cv2.waitKey(0)

#again but with noisy dog pic
#gaussian
gNoise = cv2.GaussianBlur(smallNoise, (75,75), 3)
g2Noise = gNoise/gNoise.max()*255
cv2.imshow('gNoise', g2Noise.astype(np.uint8))
cv2.waitKey(0)

#median
medianNoise = cv2.medianBlur(smallNoise,5)
cv2.imshow('median',medianNoise)
cv2.waitKey(0)


#####################################################################################
# Create noisy image as g = f + sigma * noise
# with noise scaled by sigma = .2 max(f)/max(noise)
noise = np.random.randn(small.shape[0], small.shape[1])
smallNoisy = np.zeros(small.shape, np.float64)
sigma = 0.2 * small.max()/noise.max()

# Color images need noise added to all channels
if len(small.shape) == 2:
    smallNoisy = small + sigma * noise
else:
    smallNoisy[:, :, 0] = small[:, :, 0] + sigma * noise
    smallNoisy[:, :, 1] = small[:, :, 1] + sigma * noise
    smallNoisy[:, :, 2] = small[:, :, 2] + sigma * noise

# Calculate the index of the middle column in the image
col = int(smallNoisy.shape[1]/2)
# Obtain the image data for this column
#colData = smallNoisy[0:smallNoisy.shape[0], col, 0]
colData = small[0:smallNoisy.shape[0], col, 0]

# Plot the column data as a function
xvalues = np.linspace(0, len(colData) - 1, len(colData))
plt.plot(xvalues, colData, 'b')
#plt.savefig('/Users/paucavp1/Temp/function.png', bbox_inches='tight')
#plt.clf()
plt.show()


#####################################################################################
# Compute the 1-D Fourier transform of colData
F_colData = np.fft.fft(colData.astype(float))

# Plot the magnitude of the Fourier coefficients as a stem plot
# Notice the use off fftshift() to center the low frequency coefficients around 0
xvalues = np.linspace(-int(len(colData)/2), int(len(colData)/2)-1, len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, np.fft.fftshift(np.abs(F_colData)), 'g')
xvalues = np.linspace(0, len(colData), len(colData))
markerline, stemlines, baseline = plt.stem(xvalues, (np.abs(F_colData)), 'g')
plt.setp(markerline, 'markerfacecolor', 'g')
plt.setp(baseline, 'color','r', 'linewidth', 0.5)
plt.show()
cv2.waitKey(0)

print(np.abs(F_colData))

# Convert small image to grayscale so that a 3D plot is easier to make
graySmall = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayImage.jpg", graySmall)
cv2.imshow("grayDog",graySmall)
cv2.waitKey()

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:graySmall.shape[0], 0:graySmall.shape[1]]
# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, graySmall, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
plt.show()

#########################################################################################
# Take the 2-D DFT and plot the magnitude of the corresponding Fourier coefficients

F2_graySmall = np.fft.fft2(graySmall.astype(float))
fig = plt.figure()
ax = fig.gca(projection='3d')
Y = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
X = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
X, Y = np.meshgrid(X, Y)

# Standard plot: range of values makes small differences hard to see
ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2_graySmall)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

#Log(magnitude + 1) plot: shrinks the range so that small differences are visible
ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_graySmall)+1)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
# Plot the magnitude as image
plt.show()


############################################################################
#butterworth low pass and high pass filtering

# Explore the Butterworth filter
# U and V are arrays that give all integer coordinates in the 2-D plane
#  [-m/2 , m/2] x [-n/2 , n/2].
# Use U and V to create 3-D functions over (U,V)
U = (np.linspace(-int(graySmall.shape[0]/2), int(graySmall.shape[0]/2)-1, graySmall.shape[0]))
V = (np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1]))
U, V = np.meshgrid(U, V)
# The function over (U,V) is distance between each point (u,v) to (0,0)
D = np.sqrt(X*X + Y*Y)
# create x-points for plotting
xval = np.linspace(-int(graySmall.shape[1]/2), int(graySmall.shape[1]/2)-1, graySmall.shape[1])
# Specify a frequency cutoff value as a function of D.max()
D0 = 0.25 * D.max()

# The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
# and all D(u,v) where D(u,v) > 0 equal to 0
idealLowPass = D <= D0

# Filter our small grayscale image with the ideal lowpass filter
# 1. DFT of image
print(graySmall.dtype)
FTgraySmall = np.fft.fft2(graySmall.astype(float))
# 2. Butterworth filter is already defined in Fourier space
# 3. Elementwise product in Fourier space (notice fftshift of the filter)
FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(idealLowPass)
# 4. Inverse DFT to take filtered image back to the spatial domain
graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))

# Save the filter and the filtered image (after scaling)
#idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())

#graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
cv2.imwrite("idealLowPass.jpg", idealLowPass)
cv2.imwrite("grayImageIdealLowpassFiltered.jpg", graySmallFiltered)

# Plot the ideal filter and then create and plot Butterworth filters of order
# n = 1, 2, 3, 4
plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
colors='brgkmc'
for n in range(1, 5):
    # Create Butterworth filter of order n
    H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
    # Apply the filter to the grayscaled image
    FTgraySmallFiltered = FTgraySmall * np.fft.fftshift(H)
    graySmallFiltered = np.abs(np.fft.ifft2(FTgraySmallFiltered))
    #graySmallFiltered = ski.img_as_ubyte(graySmallFiltered / graySmallFiltered.max())
    cv2.imwrite("grayImageButterworth-n" + str(n) + ".jpg", graySmallFiltered)
    # cv2.imshow('H', H)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #H = ski.img_as_ubyte(H / H.max())
    cv2.imwrite("butter-n" + str(n) + ".jpg", H)
    # Get a slice through the center of the filter to plot in 2-D
    slice = H[int(H.shape[0]/2), :]
    plt.plot(xval, slice, colors[n-1], label='n='+str(n))
    plt.legend(loc='upper left')

# plt.show()
plt.savefig('butterworthFilters.jpg', bbox_inches='tight')