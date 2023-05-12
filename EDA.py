#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python numpy matplotlib seaborn')


# In[24]:


# Define the image directories
lion_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Lions'
cheetah_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Cheetahs'

lion_images = []
for filename in os.listdir(lion_dir):
    img = cv2.imread(os.path.join(lion_dir, filename))
    lion_images.append(img)

cheetah_images = []
for filename in os.listdir(cheetah_dir):
    img = cv2.imread(os.path.join(cheetah_dir, filename))
    cheetah_images.append(img)

# Calculate the number of images in each class
num_lions = len(lion_images)
num_cheetahs = len(cheetah_images)

# Create a bar chart to visualize the distribution of images in each class
fig, ax = plt.subplots()
ax.bar(['Lions', 'Cheetahs'], [num_lions, num_cheetahs])
ax.set_title('Number of Images per Class')
ax.set_xlabel('Class')
ax.set_ylabel('Number of Images')
plt.show()
# Load images and create labels
images = []
labels = []
for filename in os.listdir(cheetah_dir):
    img = cv2.imread(os.path.join(cheetah_dir, filename))
    if img is not None:
        images.append(img)
        labels.append("Cheetah")
for filename in os.listdir(lion_dir):
    img = cv2.imread(os.path.join(lion_dir, filename))
    if img is not None:
        images.append(img)
        labels.append("Lion")

# Show example images of each class
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(images[0])
axs[0].set_title(labels[0])
axs[1].imshow(images[-1])
axs[1].set_title(labels[-1])
plt.show()

# Calculate mean and standard deviation of image intensities for each class
means = []
stds = []
for i in range(len(labels)):
    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    mean = np.mean(img)
    std = np.std(img)
    means.append(mean)
    stds.append(std)

# Create box plots to visualize the distribution of image intensities for each class
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=means[:num_cheetahs], ax=axs[0])
axs[0].set_title('Cheetah Mean Intensity')
sns.boxplot(y=means[num_cheetahs:], ax=axs[1])
axs[1].set_title('Lion Mean Intensity')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=stds[:num_cheetahs], ax=axs[0])
axs[0].set_title('Cheetah Intensity Standard Deviation')
sns.boxplot(y=stds[num_cheetahs:], ax=axs[1])
axs[1].set_title('Lion Intensity Standard Deviation')
plt.show()

# Create a scatter plot to visualize the relationship between mean and standard deviation of intensities for each class
sns.scatterplot(x=means, y=stds, hue=labels)
plt.title("Mean vs. Standard Deviation of Intensities")
plt.xlabel("Mean Intensity")
plt.ylabel("Standard Deviation")
plt.show()



# In[ ]:




