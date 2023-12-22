import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import (
    label,
    regionprops,
    moments,
    moments_central,
    moments_normalized,
    moments_hu,
)
from skimage import io, exposure, color, filters, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# Pre step 1: Import the images into this folder and open them in the code below
images = [
    "a.bmp",
    "d.bmp",
    "m.bmp",
    "n.bmp",
    "o.bmp",
    "p.bmp",
    "q.bmp",
    "r.bmp",
    "u.bmp",
    "w.bmp",
]
i = 0
features = []  # so what will features store?

for file_path in images:
    img = io.imread(file_path)
    th = 200
    img_binary = (img < th).astype(np.double)
    kernel_size = 6
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # modified_image = cv2.dilate(img_binary, kernel, iterations=1)
    # modified_image = cv2.erode(modified_image, kernel, iterations=1)
    modified_image = img_binary

    # chnage line below back to modified image instead of img_binary
    img_label = label(modified_image, background=0)
    print(np.amax(img_label))

    unfilteredRegions = regionprops(img_label)

    # Filter regions based on area (greater than 5x5 pixels)
    regions = [region for region in unfilteredRegions if region.area > 6 * 5]
    io.imshow(modified_image)
    ax = plt.gca()

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        roi = modified_image[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)

        hu = moments_hu(nu)
        features.append(hu)
        if i == 0:
            features.append(hu)
            i = i + 1
        # so this is now 800x7 ish

        if minr - 3 >= 0:
            ax.add_patch(
                Rectangle(
                    (minc, minr - 3),
                    maxc - minc,
                    maxr - minr,
                    fill=False,
                    edgecolor="red",
                    linewidth=1,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (minc, minr),
                    maxc - minc,
                    maxr - minr,
                    fill=False,
                    edgecolor="red",
                    linewidth=1,
                )
            )
        # ----------------new--------------------
    ax.set_title("Bounding Boxes   ")
    io.show()


# labels are a, d, m, n, o, p, q, r
print("shape:")
features = np.array(features)
print(features.shape)
imgFinal = io.imread("test.bmp")

Features_test = []
# features_test is a 10x7 array where each row is the avearged out wu moments of that row.  Each row is a letter

th2 = 190  # threshold for binarizing
imgFinal_binary = (imgFinal < th2).astype(np.double)
kernel_size2 = 6
kernel2 = np.ones((kernel_size2, kernel_size2), np.uint8)
# Dilate the binary image
modified_imageFinal = cv2.dilate(imgFinal_binary, kernel2, iterations=1)
# Erode the dilated image
modified_imageFinal = cv2.erode(modified_imageFinal, kernel2, iterations=1)
img_label2 = label(modified_imageFinal, background=0)
print(np.amax(img_label2))
regions2 = regionprops(img_label2)
io.imshow(modified_imageFinal)
ax2 = plt.gca()

locations = []
classes = []

for props2 in regions2:
    minr, minc, maxr, maxc = props2.bbox
    roi = imgFinal_binary[minr:maxr, minc:maxc]
    m = moments(roi)
    cc = m[0, 1] / m[0, 0]
    cr = m[1, 0] / m[0, 0]
    locations.append((cc, cr))
    mu = moments_central(roi, center=(cr, cc))
    nu = moments_normalized(mu)

    hu = moments_hu(nu)
    Features_test.append(hu)

    # this is to fill gaps in data in random spots so as to not skew data

    # this is to make the bounding box more accurate

Features_Array = np.array(features)
Features_testNorm = np.array(Features_test)
print("feature_tests shape:")
print(Features_testNorm.shape)
ft_means = np.mean(Features_Array, axis=0)
ft_sds = np.std(Features_Array, axis=0)
Normalized_Features = (Features_Array - ft_means) / ft_sds
Normalized_Features_test = (np.array(Features_testNorm) - ft_means) / ft_sds

labels = ["a", "d", "m", "n", "o", "p", "q", "r", "u", "w"]
distances = cdist(Normalized_Features, Normalized_Features_test, metric="euclidean")

# Identify the index of the nearest neighbor for each row in Features_test
nearest_neighbors_indices = np.argmin(distances, axis=0)

# Use the nearest neighbor indices to get the corresponding labels for each row in Features_test
recognized_letters = []
for index in nearest_neighbors_indices:
    # Use integer division to get the corresponding label
    letter = labels[index // 80]
    recognized_letters.append(letter)
    classes.append(letter)

print(recognized_letters)
print("size of recognized letters:")
print(len(recognized_letters))

# Y true is meant to show the number of each symbol.  I added some to match it up with the number of characters the code was
# recognizing

# THIS MUST BE CHANGED FOR OTHER CASES
Ytrue = (
    ["a"] * 7
    + ["d"] * 7
    + ["m"] * 7
    + ["n"] * 7
    + ["o"] * 7
    + ["p"] * 7
    + ["q"] * 7
    + ["r"] * 7
    + ["u"] * 7
    + ["w"] * 7
)
print(len(Ytrue))


i = 0
for props2 in regions2:
    minr, minc, maxr, maxc = props2.bbox
    ax2.add_patch(
        Rectangle(
            (minc, minr - 3),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=1,
        )
    )
    ax2.text(
        maxr,
        minc + 5,
        recognized_letters[i],
        fontsize=12,
        ha="center",
        va="bottom",
        color="red",
        clip_on=True,
    )
    i = i + 1
ax2.set_title("Bounding Boxes2   ")
io.show()

# Now, let's calculate the recognition rate
correct = 0
corInd = 0
for k in recognized_letters:
    if corInd == len(Ytrue):
        break
    if k == Ytrue[corInd]:
        correct = correct + 1
    corInd = corInd + 1

ratio = correct / len(Ytrue)
print("The percentage correct is: ")
print(ratio * 100)

# These comparisons make sure we dont hit any index out of bounds errors
if len(Normalized_Features) <= len(Normalized_Features_test):
    D = cdist(Normalized_Features, Normalized_Features_test[: len(Normalized_Features)])
    io.imshow(D)
    plt.title("Distance Matrix")
    io.show()
    D_index = np.argsort(D, axis=1)
else:
    D = cdist(
        Normalized_Features[: len(Normalized_Features_test)], Normalized_Features_test
    )
    io.imshow(D)
    plt.title("Distance Matrix")
    io.show()
    D_index = np.argsort(D, axis=1)

if len(Ytrue) >= len(recognized_letters):
    confM = confusion_matrix(Ytrue[: len(recognized_letters)], recognized_letters)
    io.imshow(confM)
    plt.title("Confusion Matrix")
    io.show()
else:
    confM = confusion_matrix(Ytrue, recognized_letters[: len(Ytrue)])
    io.imshow(confM)
    plt.title("Confusion Matrix")
    io.show()
