#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from PIL import Image, ImageTk

class ImageEDAApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image EDA")
        
        # Create a canvas to display images/plots
        self.canvas = tk.Canvas(self.master, width=800, height=400)
        self.canvas.pack()
        
        # Create a button to trigger the EDA
        self.button = tk.Button(self.master, text="Visualize Images", command=self.visualize_images)
        self.button.pack()
    
    def visualize_images(self):
        # Define the image directories
        lion_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Lions'
        cheetah_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Cheetahs'

        # Load the images
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
        fig1, ax1 = plt.subplots()
        ax1.bar(['Lions', 'Cheetahs'], [num_lions, num_cheetahs])
        ax1.set_title('Number of Images per Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Images')
        
        # Convert the Matplotlib figure to an image and display it in the GUI
        image1 = self.plot_to_image(fig1)
        self.display_image(image1)
        
        # Calculate mean and standard deviation of image intensities for each class
        means = []
        stds = []
        images = []
        labels = []
        
        for filename in os.listdir(cheetah_dir):
            img = cv2.imread(os.path.join(cheetah_dir, filename))
            if img is not None:
                images.append(img)
                labels.append("Cheetah")
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray_img)
                std = np.std(gray_img)
                means.append(mean)
                stds.append(std)
        
        for filename in os.listdir(lion_dir):
            img = cv2.imread(os.path.join(lion_dir, filename))
            if img is not None:
                images.append(img)
                labels.append("Lion")
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray_img)
                std = np.std(gray_img)
                means.append(mean)
                stds.append(std)
        
        # Create box plots to visualize the distribution of image intensities for each class
        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(y=means[:num_cheetahs], ax=axs2[0])
        axs2[0].set_title('Cheetah Mean Intensity')
        sns.boxplot(y=means[num_cheetahs:], ax=axs2[1])
        axs2[1].set_title('Lion Mean Intensity')
        
        # Convert the Matplotlib figure to an image and display it in the GUI
        image2 = self.plot_to_image(fig2)
        self.display_image(image2)
        
        fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(y=stds[:num_cheetahs], ax=axs3[0])
        axs3[0].set_title('Cheetah Intensity Standard Deviation')
        sns.boxplot(y=stds[num_cheetahs:], ax=axs3[1])
        axs3[1].set_title('Lion Intensity Standard Deviation')
        
        # Convert the Matplotlib figure to an image and display it in the GUI
        image3 = self.plot_to_image(fig3)
        self.display_image(image3)
        
        fig4 = plt.figure(figsize=(8, 6))
        sns.scatterplot(x=means, y=stds, hue=labels)
        plt.title("Mean vs. Standard Deviation of Intensities")
        plt.xlabel("Mean Intensity")
        plt.ylabel("Standard Deviation")
        
        # Convert the Matplotlib figure to an image and display it in the GUI
        image4 = self.plot_to_image(fig4)
        self.display_image(image4)
    
    def plot_to_image(self, fig):
        # Convert Matplotlib figure to a PIL Image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
        image = Image.fromarray(image)
        return image
    
    def display_image(self, image):
        # Convert PIL Image to Tkinter PhotoImage and display it on the canvas
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.image = photo
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEDAApp(root)
    root.mainloop()

