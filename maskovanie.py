from PIL import Image
import cupy
import os
import numpy as np

# Function for image masking
def extend(image: cupy.array, extension):
    extended = cupy.zeros((len(image) + extension * 2, len(image[0]) + extension * 2, 3))
    #print(len(extended), len(extended[0]), extension)
    new_row = []

    for row in range(len(extended)):
        for col in range(len(extended[0])):
            #print(row, col)
            # Handle pixels outside the original image
            if row < extension:
                if col < extension:
                    new_row.append(image[0][0])  # Copy pixel from top-left corner
                elif col > len(image) + extension - 1:
                    new_row.append(image[0][-1])  # Copy pixel from top-right corner
                else:
                    new_row.append(image[0][col - extension])  # Copy pixel from top row
            elif row > len(image) + extension - 1:
                if col < extension:
                    new_row.append(image[-1][0])  # Copy pixel from bottom-left corner
                elif col > len(image) + extension - 1:
                    new_row.append(image[-1][-1])  # Copy pixel from bottom-right corner
                else:
                    new_row.append(image[-1][col - extension])  # Copy pixel from bottom row
            elif col < extension:
                new_row.append(image[row - extension][0])  # Copy pixel from left column
            elif col > len(image) + extension - 1:
                new_row.append(image[row - extension][-1])  # Copy pixel from right column
            else:
                #print(row - extension, col - extension)
                new_row.append(image[row - extension][col - extension])  # Copy pixel from original image
        
        extended[row] = cupy.asarray(new_row)
        new_row = []

    return extended


test = Image.open(os.getcwd() + "\\dataset\\training_set\\cats\\cat.1.jpg")
array = cupy.asarray(test)
mask_matrix = cupy.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])
extend_by = 1

extended = extend(array, extend_by)
#nparr = array.get().astype(np.uint8)
#Image.fromarray(nparr).show()

# The following code is commented out

output = cupy.ndarray(shape=(len(array), len(array[0]),3), dtype=cupy.uint8)
accumulator = 0
for row in range(len(array)):
    for col in range(len(array[0])):
        for i in range(3):
            for j in range(3):
                accumulator += extended[row + i][col + j] * mask_matrix[i][j]
        
                #print(row, col)
                #print(accumulator,extended[row + i][col + j] , mask_matrix[i][j])
    c = sum(accumulator)/3/255
    #print(c)
    if c > 1:
        output[row][col] = cupy.array([255, 255, 255])
    elif c > 0.25:
        output[row][col] = cupy.array([128, 128, 128])
    else:
        output[row][col] = cupy.array([0, 0, 0])
    accumulator = 0
nparr = output.get()
Image.fromarray(nparr).show()
