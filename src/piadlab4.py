import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal

#DYSKRETYZACJA
def fun(f, Fs):
    for i in Fs:
        t = np.arange(0, 1, 1 / i)
        x = np.sin(2 * np.pi * f * t)
        plt.plot(t, x)
        plt.show()

f = 10  # czestotliwosc sygnalu
Fs = [20, 21, 30, 45, 50, 100, 150, 200, 250, 1000]  # czestotliwosc probkowania
fun(f, Fs)

#4 - Jest to twierdzenia Nyquista-Shannona
#5 - Jest to Aliasing

#6
bricks = plt.imread('../img/bricks.png')
#7
fig, axs = plt.subplots(1, 2)
axs[0].imshow(bricks, interpolation='nearest')
axs[1].imshow(bricks, interpolation='lanczos')
plt.show()

#KWANTYZACJA
#1
bricks = plt.imread('../img/bricks.png')
#2
print(f'wymiary obrazka: {np.ndim(bricks)}')
#3
print(f'wartosci piksela: {np.shape(bricks)[-1]}')
#4
wart = np.shape(bricks)
im1, im2, im3 = bricks.copy(), bricks.copy(), bricks.copy()
for x in range(wart[0]):
    for y in range(wart[1]):
        for i in range(3):
            im1[x, y, i] = (max(im1[x, y, 0], im1[x, y, 1], im2[x, y, 2]) + min(im1[x, y, 0], im1[x, y, 1], im2[x, y, 2]))/2
            im2[x, y, i] = (im2[x, y, 0] + im2[x, y, 1] + im2[x, y, 2]) / 3
            im3[x, y, i] = 0.21 * im3[x, y, 0] + 0.72 * im3[x, y, 1] + 0.07 * im3[x, y, 2]

#5
print(np.histogram(im1))
print(np.histogram(im2))
print(np.histogram(im3))

#7
im4 = im3.copy()
size = np.shape(im4)
hist, bins = np.histogram(im4, bins=16)
for x in range(size[0]):
    for y in range(size[1]):
        for i in range(1, len(bins)):
            if im4[x, y, 0] < bins[i] and im4[x, y, 0] >= bins[i-1]:
                im4[x, y, 0] = (bins[i] + bins[i-1]) / 2
                im4[x, y, 1] = (bins[i] + bins[i-1]) / 2
                im4[x, y, 2] = (bins[i] + bins[i-1]) / 2

#8
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(im1)
axs[0, 0].set_title('im1')
axs[0, 1].imshow(im2)
axs[0, 1].set_title('im2')
axs[1, 0].imshow(im3)
axs[1, 0].set_title('im3')
axs[1, 1].imshow(im4)
axs[1, 1].set_title('im4')
plt.show()

#6, 8
bin = 16
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(im1.ravel(), bins=bin)
axs[0, 0].set_title('im1')
axs[0, 1].hist(im2.ravel(), bins=bin)
axs[0, 1].set_title('im2')
axs[1, 0].hist(im3.ravel(), bins=bin)
axs[1, 0].set_title('im3')
axs[1, 1].hist(im4.ravel(), bins=bin)
axs[1, 1].set_title('im4')
plt.show()

#BINARYZACJA
#1
im5 = plt.imread('../img/abstr.png')
size = np.shape(im5)
#2
for x in range(size[0]):
    for y in range(size[1]):
        for i in range(3):
            im5[x, y, i] = 0.21 * im5[x, y, 0] + 0.72 * im5[x, y, 1] + 0.07 * im5[x, y, 2]
im5Hist, bins = np.histogram(im5)
plt.imshow(im5)
plt.show()
plt.hist(im5.ravel())
plt.show()
#minimum = sp.signal.argrelextrema(im5Hist, np.less)
#print(minimum[0][0])
def findMinimum(hist, start, end):
    cen = int(start + (end-start)/2)
    if (cen == 0 or hist[cen-1] > hist[cen]) and (cen == len(hist) - 1 or hist[cen+1] > hist[cen]):
        return cen
    elif cen > 0 and hist[cen] > hist[cen-1]:
        return findMinimum(hist, start, cen)
    else:
        return findMinimum(hist, cen+1, end)

#7
im6 = im5.copy()
for x in range(size[0]):
    for y in range(size[1]):
        if im6[x, y, 0] < bins[findMinimum(im5Hist, 0, len(im5Hist))]:
            im6[x, y, 0], im6[x, y, 1], im6[x, y, 2] = 0, 0, 0
        else:
            im6[x, y, 0], im6[x, y, 1], im6[x, y, 2] = 1, 1, 1
#8
plt.imshow(im6)
plt.show()


