#!/usr/bin/env python
# coding: utf-8

# # Particle size measurment with distribution

# ### IMAGE FORMAT CONVERSION FROM TIF TO JPG AND THEN CROPPING

# In[1]:


from PIL import Image
image_path="C:/Users/A/Downloads/DS-115_003" 

# Load the TIFF image
tiff_image = Image.open(image_path +'.tif')  # Replace input.tif with your TIFF image file path

# Save it as a JPEG image
tiff_image.save(image_path+'.jpg', 'JPEG')  # 'output.jpg' is the desired output file name


# In[2]:


# Importing Image class from PIL module
from PIL import Image

# Opens a image in RGB mode
im = Image.open(image_path+'.jpg')

# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size

# Setting the points for cropped image
left = 0
top = 0
right = width
bottom = height-70

# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))


# Shows the image in image viewer
im1.save(image_path+'.jpg')


# ### THE ORIGINAL IMAGE

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread(image_path+'.jpg',1)
    
plt.imshow(image)
#plt.savefig("/Users/shivashankar/Downloads/original.jpg")


# In[4]:


import cv2
import numpy as np
import sys

image_path = r"C:\Users\A\Downloads\DS-115_003.tif"  # Raw string prevents escape sequence interpretation

image = cv2.imread(image_path, 1)

if np.lib.NumpyVersion(np.__version__) > "1.20.0" and sys.version_info >= (3, 9):
    NumPyArrayGeneric = np.ndarray
else:
    NumPyArrayGeneric = np.ndarray

# Check if image was loaded successfully
if image is None:
    print(f"Error: Could not read image from {image_path}")
else:
    print("Image loaded successfully.")


# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image path
image_path = r"C:\Users\A\Downloads\DS-115_003"

# Display the original image
original_image = cv2.imread(image_path + '.jpg', 1)

# Check if the image was loaded successfully
if original_image is None:
    print(f"Error: Could not read the original image from {image_path}.jpg")
else:
    # Convert BGR to RGB for proper display
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')  # Hide axes for better display
    plt.show()

# Display the cropped image
cropped_image = cv2.imread(image_path + '_cropped.jpg', 1)

# Check if the cropped image was loaded successfully
if cropped_image is None:
    print(f"Error: Could not read the cropped image from {image_path}_cropped.jpg")
else:
    # Convert BGR to RGB for proper display
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cropped_image_rgb)
    plt.title('Cropped Image')
    plt.axis('off')
    plt.show()


# ### THE IMAGE AFTER BRIGHTNESS ADJUSTED

# In[6]:


# Adjust the brightness and contrast
# Adjusts the brightness by adding 10 to each pixel value
brightness = 20
contrast = 1

# Adjust brightness and contrast of the cropped image
image2 = cv2.addWeighted(cropped_image, contrast, np.zeros(cropped_image.shape, cropped_image.dtype), 0, brightness)

# Display the brightness and contrast adjusted image
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
plt.title("Brightness & Contrast Adjusted (Cropped Image)")
plt.imshow(image2_rgb)
plt.axis('off')
plt.show()


# ### THE IMAGE AFTER SHARPENING

# In[7]:


# Create the sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Sharpen the image (brightness & contrast adjusted cropped image)
sharpened_image = cv2.filter2D(image2, -1, kernel)

# Convert to RGB for proper display
sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

# Display the sharpened image
plt.imshow(sharpened_image_rgb)
plt.title("Sharpened Image (Brightness & Contrast Adjusted Cropped Image)")
plt.axis('off')
plt.show()


# In[8]:


# Dimensions of the image in micrometers
image_width_micrometers = 37.4  # Width of the image in micrometers
image_height_micrometers = 32.4  # Height of the image in micrometers

# Dimensions of the image in pixels (replace these with your actual pixel dimensions)
image_width_pixels = sharpened_image.shape[1]  # Width in pixels
image_height_pixels = sharpened_image.shape[0]  # Height in pixels

# Calculate the conversion factor from pixels to micrometers
conversion_factor = image_width_micrometers / image_width_pixels

# Convert the cropped image to grayscale
gray_cropped = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray_cropped, (5, 5), 0)

# Apply thresholding to segment particles
threshold_value, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours of particles
contours, threshold_value = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate particle diameters and collect them in a list
particle_diameters_micrometers = []
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter_pixels = radius * 2
    diameter_micrometers = diameter_pixels * conversion_factor  # Convert to micrometers
    particle_diameters_micrometers.append(diameter_micrometers)

# Draw particle boundaries
cv2.drawContours(sharpened_image, contours, -1, (100, 0, 255), 2)  # Green color, thickness 2

# Calculate the average diameter in micrometers
average_diameter_micrometers = np.mean(particle_diameters_micrometers) if particle_diameters_micrometers else 0
print("Average Diameter of Particles:", average_diameter_micrometers, "(micrometers)")

# Display the cropped image with particle boundaries
plt.figure(figsize=(10, 10))
plt.imshow(sharpened_image)
plt.axis('off')  # Optional: Hide axis for better visualization
plt.title("Particles with Boundaries")
plt.show()


# ### THE IMAGE AFTER BLURRING, THRESHOLDING TO SEPARATE PARTICLES, DRAW CONTOURS AROUND PARTICLES

# In[9]:


# Convert the image to grayscale
gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply thresholding to segment particles
threshold_value, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours of particles
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables for maximum width and height
max_width = 0
max_height = 0
particle_diameters = []

# Loop through contours to find the largest bounding box and calculate particle diameters
for contour in contours:
    # Bounding box for contour
    x, y, width, height = cv2.boundingRect(contour)
    max_width = max(max_width, width)
    max_height = max(max_height, height)

    # Calculate particle diameter from the enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2
    particle_diameters.append(diameter)

    # Draw the bounding box on the original image
    cv2.rectangle(sharpened_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Convert image to RGB for display in matplotlib
sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

# Calculate the average particle diameter
average_diameter = np.mean(particle_diameters)

# Print the dimensions of the image and average particle diameter
print("Width in pixels:", max_width)
print("Height in pixels:", max_height)
print("Average Diameter of Particles:", average_diameter, "(pixels)")

# Display the original image with bounding boxes and in large size
plt.figure(figsize=(10, 10))
plt.imshow(sharpened_image_rgb)
plt.title(f"Image with Bounding Boxes\nWidth: {max_width}px, Height: {max_height}px\nAvg Particle Diameter: {average_diameter:.2f}px")
plt.axis('off')
plt.show()


# In[10]:


for i in range(len(particle_diameters)):
    particle_diameters[i]=round(particle_diameters[i],2)


# In[11]:


particle_diameters.sort()
print(particle_diameters)


# In[12]:


temp=particle_diameters


# In[13]:


print(len(temp))


# In[14]:


temp_copy = temp[:]
for i in range(len(temp_copy)):
    if temp_copy[i] == 0.0 or temp_copy[i] >= 90 or temp_copy[i]<20:
        temp.remove(temp_copy[i])


# In[15]:


print(len(particle_diameters))


# # MEAN SIZE OF THE PARTICLE after excluding outliers and cluserters

# In[16]:


print(np.mean(particle_diameters))


# In[17]:


import matplotlib.pyplot as plt
import numpy as np


# Define the bins for particle diameter ranges
bins = [20, 30, 40, 50,60,70,80,90]  # Adjust these bins according to your data

# Create a histogram of the particle diameters
hist, bins = np.histogram(particle_diameters, bins=bins)

# Calculate the midpoints of each bin for labeling the x-axis
bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

# Create the bar chart
plt.bar(bin_midpoints, hist, width=10, align='center')

# Label the axes
plt.xlabel('Particle Diameter')
plt.ylabel('Frequency')

# Set the x-axis ticks to be the midpoints of the bins
plt.xticks(bin_midpoints)

# Add a title to the chart
plt.title('Particle Diameter Distribution')

# Show the chart
plt.show()


# In[18]:


# import matplotlib.pyplot as plt
# import numpy as np



# # Calculate the number of bins using the Freedman-Diaconis rule
# data_range = max(particle_diameters) - min(particle_diameters)
# print(data_range)
# bin_width = 2 * np.percentile(particle_diameters, 75) / (len(particle_diameters) ** (1/3))
# print(bin_width)
# num_bins = int(data_range / bin_width)
# print(num_bins)

# # Create the histogram
# plt.hist(particle_diameters, bins=num_bins, edgecolor='k', alpha=0.75)

# # Label the axes
# plt.xlabel('Particle Diameter')
# plt.ylabel('Frequency')

# # Add a title to the chart
# plt.title('Particle Diameter Distribution')

# # Show the chart
# plt.show()


# In[19]:


gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 50, 150)

# Display the original and edge-detected images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Sharpend Image")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')

plt.show()


# In[20]:


# New Analysis: Texture Analysis using GLCM (Gray Level Co-occurrence Matrix)
import mahotas.features.texture as texture
import numpy as np  # Add this line to import NumPy

# Convert the image to uint8 (mahotas requires images to be in this format)
gray_image_uint8 = (gray_image * 255).astype(np.uint8)

# Compute GLCM
glcm = texture.haralick(gray_image_uint8, return_mean=True)

# Display the GLCM features
print("GLCM Features:")
print(glcm)


# In[21]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Flatten the GLCM
glcm_flat = glcm.flatten()

# Reshape for clustering
glcm_flat = np.reshape(glcm_flat, (-1, 1))

resized_original_image = cv2.resize(sharpened_image, (len(glcm_flat), 1))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(glcm_flat)

# Reshape the labels to the resized original image shape
cluster_labels = kmeans.labels_.reshape(resized_original_image.shape[0], resized_original_image.shape[1])

# Display the clustered regions
plt.imshow(cluster_labels, cmap='viridis', interpolation='nearest')
plt.title("Clustered Regions based on Texture Features")
plt.show()


# In[22]:


# Display mean particle diameter
mean_diameter = np.mean(particle_diameters)
print("Mean Diameter of Particles:", mean_diameter, "(pixels)")

# Display a sorted list of particle diameters
sorted_diameters = np.sort(particle_diameters)
print("Sorted Particle Diameters:", sorted_diameters)


# In[24]:


import seaborn as sns
import pandas as pd

# Create a DataFrame for the violin plot
df_violin = pd.DataFrame({'Particle Diameter': particle_diameters})

# Create a violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x=df_violin['Particle Diameter'], inner="quartile")
plt.title('Violin Plot of Particle Diameters')
plt.xlabel('Particle Diameter (pixels)')
plt.show()

