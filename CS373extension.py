from audioop import minmax
import math
import sys
from pickle import TRUE
from pathlib import Path
from PIL import Image
import easyocr

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b

# queue data structure from the coderunner question


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows,
     rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):

    new_array = [[initValue for x in range(
        image_width)] for y in range(image_height)]
    return new_array

# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!

# read the input image, convert rgb data to greyscale and stretch the values to lie between 0 and 255
# week 10 coderunner


def computeRGBToGreyScale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    grey_pixel_array = createInitializedGreyscalePixelArray(
        image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            grey_pixel_array[x][y] = round(
                0.299*pixel_array_r[x][y] + 0.587*pixel_array_g[x][y] + 0.114*pixel_array_b[x][y])

    return grey_pixel_array


def scaleTo0And255(pixel_array, image_width, image_height):
    scaleImage = createInitializedGreyscalePixelArray(
        image_width, image_height)
    min = 255
    max = 0

    for r in range(image_height):
        for c in range(image_width):
            if pixel_array[r][c] < min:
                min = pixel_array[r][c]
            if pixel_array[r][c] > max:
                max = pixel_array[r][c]

    multiplyValue = (255/(max-min))
    for r in range(image_height):
        for c in range(image_width):
            scaleImage[r][c] = (pixel_array[r][c] - min)*multiplyValue

    return scaleImage

# Find structures with high contrast in the image by computing the standard deviation in the pixel neighbourhood
# week 11 coderunner


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    sd_result = createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(1, image_height-2):
        for c in range(1, image_width-2):
            nums = []
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    nums.append(pixel_array[r+i][c+j])

            mean = sum(nums)/len(nums)
            variance = sum((x - mean)**2 for x in nums)/len(nums)
            sd_result[r][c] = variance ** 0.5

    return sd_result


# perform a thresholding operation to get the high contrast regions as a binary image
def computeThresholdingOperation(pixel_array, image_width, image_height, thresholdValue):
    img = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < thresholdValue:
                img[i][j] = 0
            else:
                img[i][j] = 255

    return img

#  Perform several 3x3 dilation steps followed by several 3x3 erosion steps to get a “blob” region for the license plate (morphological closing)
# week 12 coderunner


def compute3x3Dilation(pixel_array, image_width, image_height):
    dilation = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            for a in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    if (i+a in range(image_height)
                        and j+b in range(image_width)
                            and pixel_array[i+a][j+b] != 0):
                        dilation[i][j] = 1
    return dilation


def compute3x3Erosion(pixel_array, image_width, image_height):
    erosion = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if (i == 0) or (i == image_height - 1) or (j == 0) or (j == image_width - 1):
                continue

            pv = 1
            for a in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    if pixel_array[i+a][j+b] == 0:
                        pv = 0

            erosion[i][j] = pv

    return erosion


def computeLargestConnectedComponent(pixel_array, image_width, image_height):
    llc = createInitializedGreyscalePixelArray(
        image_width, image_height)
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    visited = set()
    array = []
    count = 0
    num = 1
    xMin = image_width-1
    xMax = 0
    yMin = image_height-1
    yMax = 0

    for a in range(image_height):
        for b in range(image_width):
            if (a, b) in visited or pixel_array[a][b] == 0:
                continue

            q = Queue()
            q.enqueue((a, b))
            visited.add((a, b))

            while q.isEmpty() == False:
                (r, c) = q.dequeue()
                llc[r][c] = num
                count = count + 1
                xMin = min(xMin, c)
                xMax = max(xMax, c)
                yMin = min(yMin, r)
                yMax = max(yMax, r)

                for (i, j) in directions:
                    row = r + i
                    col = c + j

                    if (row in range(image_height) and col in range(image_width) and pixel_array[row][col] != 0 and (row, col) not in visited):
                        q.enqueue((row, col))
                        visited.add((row, col))

            array.append((num, count, xMin, xMax, yMin, yMax))
            num = num + 1
            count = 0
            xMin = image_width-1
            xMax = 0
            yMin = image_height-1
            yMax = 0

    return (llc, array)


def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / \
        Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    cropped_path = Path("cropped_license_image")
    if not cropped_path.exists():
        cropped_path.mkdir(parents=True, exist_ok=True)

    cropped_filename = str(cropped_path) + "/" + \
        str(input_filename.replace(".png", "_cropped_license_plate.png"))
    output_filename = output_path / \
        Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        cropped_filename = str(command_line_arguments[1])
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g,
     px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here

    # PART 1: computing rgb to greyscale and contrast stretch
    px_array_gray = computeRGBToGreyScale(
        px_array_r, px_array_g, px_array_b, image_width, image_height)
    axs1[0, 0].set_title('Greyscale')
    axs1[0, 0].imshow(px_array_gray, cmap='gray')

    px_array_contrast = scaleTo0And255(
        px_array_gray, image_width, image_height)
    # PART 2: standard deviation and then stretch
    px_array_sd = computeStandardDeviationImage5x5(
        px_array_contrast, image_width, image_height)
    px_array_contrast1 = scaleTo0And255(px_array_sd, image_width, image_height)

    # PART 3: threshold hint value of 150
    px_array_threshold = computeThresholdingOperation(
        px_array_contrast1, image_width, image_height, 150)
    px_array_threshold_contrast = scaleTo0And255(
        px_array_threshold, image_width, image_height)
    axs1[1, 0].set_title('Scale and Thresholding')
    axs1[1, 0].imshow(px_array_threshold_contrast, cmap='gray')

    # dilation and erosion ( times)
    px_array_dilation = compute3x3Dilation(
        px_array_threshold_contrast, image_width, image_height)

    for i in range(3):
        px_array_dilation = compute3x3Dilation(
            px_array_dilation, image_width, image_height)

    px_array_erosion = compute3x3Erosion(
        px_array_dilation, image_width, image_height)
    for i in range(3):
        px_array_erosion = compute3x3Erosion(
            px_array_erosion, image_width, image_height)

    axs1[1, 0].set_title('Morphological closing')
    axs1[1, 0].imshow(px_array_erosion, cmap='gray')

    (img, array) = computeLargestConnectedComponent(
        px_array_erosion, image_width, image_height)

    bbox_min_x = 0
    bbox_max_x = 0
    bbox_min_y = 0
    bbox_max_y = 0

    array.sort(key=lambda x: x[1], reverse=True)

    for (num, count, xMin, xMax, yMin, yMax) in array:
        ar = ((xMax-xMin+1)/(yMax-yMin+1))
        if (1.5 <= ar <= 5):
            bbox_min_x = xMin
            bbox_max_x = xMax
            bbox_min_y = yMin
            bbox_max_y = yMax
            break

    # easyocr implementation
    image = Image.open(input_filename)
    cropped = image.crop(
        (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y))
    cropped.save(cropped_filename)
    reader = easyocr.Reader(['en'], gpu=False)
    reader_word = reader.readtext(cropped_filename, min_size=100)
    area = 0
    license_word = reader_word[0]
    for word in reader_word:
        if area < (word[0][2][0]-word[0][0][0])*(word[0][3][1]-word[0][1][1]):
            license_word = word
        license_plate = license_word[1]

    print("The License Plate number is:", license_plate)

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array_gray, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(
        fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
