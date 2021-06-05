from matplotlib import pyplot
from matplotlib.patches import Rectangle
import math
import imageIO.png


def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

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


# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r, g, b, w, h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    x1 = createInitializedGreyscalePixelArray(image_width, image_height)
    sum1 = list()
    for i in pixel_array:
        for j in i:
            sum1.append(j)
    max1 = max(sum1)
    min1 = min(sum1)
    if max1 == min1:
        return x1
    else:
        pass
    for x in range(image_height):
        for y in range(image_width):
            nums = ((pixel_array[x][y] - min1) / (max1 - min1))
            x1[x][y] = round(255*nums)
    return x1


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for y in range(image_width):
            x = (0.299 * pixel_array_r[i][y] + 0.587 * pixel_array_g[i][y] + 0.114 * pixel_array_b[i][y])
            greyscale_pixel_array[i][y] = round(x)
    return greyscale_pixel_array

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    result = []
    for i in range(image_height):
        result.append([0]*image_width)
    for i in range(image_height):
        for j in range(image_width):
            if i==0 or j==0 or i==image_height-1 or j==image_width-1:
                pass
            else:
                result[i][j] = (1/8)*((pixel_array[i-1][j-1]) + -1*(pixel_array[i+1][j-1]) + 2*(pixel_array[i-1][j]) + -2*(pixel_array[i+1][j]) +
                (pixel_array[i-1][j+1]) + -1*(pixel_array[i+1][j+1]))
    return result


def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    result = list()
    x1 = 0
    x2 = 0
    while x1 != image_height:
        result.append([0] * image_width)
        x1 += 1
    while x2 != image_height:
        for j in range(image_width):
            if x2 == 0 or j == 0 or x2 == image_height - 1 or j == image_width - 1:
                pass
            else:
                result[x2][j] = (1 / 8) * (
                            -1 * (pixel_array[x2 - 1][j - 1]) + -2 * (pixel_array[x2][j - 1]) + -1 * (
                    pixel_array[x2 + 1][j - 1]) + (pixel_array[x2 - 1][j + 1]) +
                            2 * (pixel_array[x2][j + 1]) + (pixel_array[x2 + 1][j + 1]))
        x2 += 1
    return result

def computeTotal(horizontal,vertical,image_width,image_height):
    x1 = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            sum1 = math.sqrt(horizontal[i][j]*horizontal[i][j] + vertical[i][j]*vertical[i][j])
            x1[i][j] = sum1
    return x1

def smoothEdge(pixel_array,image_width,image_height):
    nums = 8
    x1 =0
    while x1 != nums:
        for i in range(1,image_height-1):
            for j in range(1,image_width-1):
                total = (pixel_array[i][j] + pixel_array[i-1][j-1]+ pixel_array[i-1][j]+
                         pixel_array[i-1][j+1]+pixel_array[i][j-1]+pixel_array[i][j+1]+
                         pixel_array[i+1][j-1]+pixel_array[i+1][j]+pixel_array[i+1][j+1])
                pixel_array[i][j] = 1/9 * total
        x1+=1
    pixel_array = scaleTo0And255AndQuantize(pixel_array, image_width, image_height)
    return pixel_array
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    #x1 = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            if pixel_array[x][y] >= threshold_value:
                pixel_array[x][y]= 255
            else:
                pixel_array[x][y] = 0
    return pixel_array

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    list1 = list()
    list2 = list()
    x =0
    while x!= image_height:
        y=0
        while y != image_width:
            list2.append(0)
            y+=1
        x+=1
        list1.append(list2)
        list2 = list()
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            if(pixel_array[i][j]>0):
                if(pixel_array[i-1][j-1]>= 1 and pixel_array[i-1][j]>=1 and pixel_array[i-1][j+1]>=1):
                    if(pixel_array[i][j-1]>= 1 and pixel_array[i][j]>=1 and pixel_array[i][j+1]>=1):
                        if(pixel_array[i+1][j-1]>= 1 and pixel_array[i+1][j]>=1 and pixel_array[i+1][j+1]>=1):
                            list1[i][j] = 1
                        else:
                            list1[i][j] = 0
                    else:
                        list1[i][j] = 0
                else:
                    list1[i][j] = 0
            else:
                list1[i][j] = 0
    return list1
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    x1 = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            if (pixel_array[i][j]>0):
                x1[i-1][j-1], x1[i-1][j], x1[i-1][j+1] = 1,1,1
                x1[i][j-1], x1[i][j], x1[i][j+1] = 1,1,1
                x1[i+1][j-1], x1[i+1][j], x1[i+1][j+1] = 1,1,1
    i=0
    j2 = 0
    while j2 != image_width:
        if pixel_array[i][j2] >0:
            if j2-1 >=0:
                x1[i][j2-1] = 1
                x1[i+1][j2-1] = 1
            x1[i][j2] = 1
            x1[i+1][j2]=1
            x1[i][j2+1] = 1
            x1[i+1][j2+1] = 1
        j2+=1
    return x1
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    q1 = Queue()
    list1 = []
    list2 = []
    dict1 = {}
    x1 = 1
    for x in range(image_height):
        l1 = []
        l2 = []
        for y in range(image_width):
            l1.append(False)
            l2.append(0)
        list1.append(l1)
        list2.append(l2)
    for i in range(image_height):
        for j in range(image_width):
            if not list1[i][j] and pixel_array[i][j] != 0:
                sum1 = count_x(pixel_array, list1, i, j, image_width, image_height, list2, x1)
                dict1[x1] = sum1
                x1 += 1
    return (list2, dict1)


def count_x(pixel_array, list1, i, j, image_width, image_height, list2, x1):
    x = [-1, 0, 1, 0]
    y = [0, 1, 0, -1]
    total = 0
    q1 = Queue()
    q1.enqueue((i, j))
    list1[i][j] = True

    while not q1.isEmpty():
        total += 1
        e, k = q1.dequeue()
        list2[e][k] = x1
        count = 0
        while count != 4:
            x_final = e + x[count]
            y_final = k + y[count]
            if x_final >= 0 and x_final < image_height:
                if y_final >= 0 and y_final < image_width:
                    if not list1[x_final][y_final] and pixel_array[x_final][y_final] != 0:
                        list1[x_final][y_final] = True
                        q1.enqueue((x_final, y_final))
            count += 1
    return total
def largest_connected_object(pixel_array, dict1,image_width,image_height):
    x1 = createInitializedGreyscalePixelArray(image_width, image_height)
    max1 = 0
    count = 0
    for key in dict1.keys():
        value = dict1[key]
        if value >= count:
            count = value
            max1 = key
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] == max1:
                x1[i][j] = 10000000
    return x1
def main():
    filename = "./images/covid19QRCode/poster1small.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    greyscale_pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    greyscale_pixel_array = scaleTo0And255AndQuantize(greyscale_pixel_array, image_width, image_height)
    horizontal = computeHorizontalEdgesSobelAbsolute(greyscale_pixel_array, image_width, image_height)
    vertical = computeVerticalEdgesSobelAbsolute(greyscale_pixel_array, image_width, image_height)
    edge_mag = computeTotal(horizontal,vertical,image_width,image_height)
    smooth_mag = smoothEdge(edge_mag,image_width,image_height)
    threshold_value = 70
    binary_image = computeThresholdGE(smooth_mag, threshold_value, image_width, image_height)

    array1 = binary_image
    for i in range(2):
        array1 = computeDilation8Nbh3x3FlatSE(smooth_mag, image_width, image_height)
        array1 = computeErosion8Nbh3x3FlatSE(array1, image_width, image_height)
    (arrayx, dict1) = computeConnectedComponentLabeling(array1, image_width, image_height)
    dict_final =largest_connected_object(arrayx, dict1, image_width,image_height)

    x1 = 0
    y1 = 0
    for i in range(image_height):
        if 10000000 in dict_final[i]:
            y1 = i
            break
    for i in range(image_width):
        if dict_final[y1][i] == 10000000:
            x1 = i
            break

    finalx = 0
    finaly = 0
    for i in range(y1,image_height):
        if 10000000 in dict_final[i]:
            finaly = i

    for i in range(x1, image_width):
        if dict_final[y1][i] == 10000000:
            finalx = i



    rgbImage = prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height)
    pyplot.imshow(rgbImage)
    # get access to the current pyplot figure
    axes = pyplot.gca()


    # create a 70x50 rectangle that starts at location 10,30, with a line width of 3
    rect = Rectangle((x1, y1), finalx-x1, finaly-y1, linewidth=3, edgecolor='g', facecolor='none')
    # paint the rectangle over the current plot
    axes.add_patch(rect)

    # plot the current figure
    pyplot.show()


if __name__ == "__main__":
    main()