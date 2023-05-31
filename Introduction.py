# Using library PIL
from PIL import Image

# Make sure that image for you want to get output should be present
# in same folder as of python file. Run this code using some others images too :)
# Check type also- .jpeg or .png or any other while changing name
Image_name = "d.jpeg"

# Open the image file
img = Image.open(Image_name)

# Get the RGB values of each pixel in the image
pixels = img.load()

# Open a file named "output.txt" in write mode
file = open("output.txt", "w")

# Get size of the image (Image is a matrix) into file
file.write(f"Size of image = {img.size}")

# Loop through each pixel and print its RGB value and gray_scale value
for x in range(img.size[0]):
    for y in range(img.size[1]):

        # Initializing gray_scale value to 0
        gray_scale = 0

        # Get pixel value from image (can be list of ints or an int)
        pixel_value = img.getpixel((x, y))

        # Check if already image is gray_scale then pixel value is an int
        if type(pixel_value) == int:
            gray_scale = pixel_value
        # If image is not gray_scale then pixel_value is a list consisting 3 values r, g, b.
        else:
            # Finding length of the list pixel_value
            length = len(pixel_value)
            # Calculate gray scale value
            for val in pixel_value:   
                gray_scale= gray_scale + val/length

        # Rounding off value upto 2 decimals
        gray_scale = round(gray_scale, 2)

        # Writing the pixel number, it's gray scale and RGB value in file fol all pixels
        file.write(f"Pixel ({x}, {y}): Gray={gray_scale}, RGB={pixel_value}\n")

        # To print pixels who have non-zero gray scale value- uncomment(remove hash(#). Keep proper indentation) following 2 lines and comment(add hash(#) in front of the line) above line
        #if gray_scale != 0:
        #    file.write(f"Pixel ({x}, {y}): Gray={gray_scale}, RGB={pixel_value}\n")
         

# Close the file
file.close()

#### output will be saved in output.txt check that ####
