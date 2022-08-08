import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Defining the x axis labels in accordance to the selected keyframes
x_labels = list(range(0, 55, 5))

# Changing the first label to 1
x_labels[0] = 1

# Converting the x axis values to string 
x_labels = [str(i) for i in x_labels]

# Appending the x axis label concerning the average frame
x_labels.append("Avg.")

# Definig the vgg16-based model test set performance regarding the selected keyframe
y_vgg16 = [0.86, 0.86, 0.84, 0.76, 0.72, 0.72, 0.74, 0.72, 0.66, 0.66, 0.76, 0.84]

# Defining a colormap
tab20b_colormap = matplotlib.cm.get_cmap("tab20b")

# Defining the bar and label colors
bar_color = tab20b_colormap(16)
label_color = tab20b_colormap(19)

# Defining the bar width
width = 0.6

# Defining the x axis bars coordinates
x_indexes = np.arange(len(y_vgg16))

# Instantiating the plot objects
fig, ax = plt.subplots()

# Defining the vertical bar plot
vgg16_bars = ax.bar(x_indexes, y_vgg16, width, color=bar_color)

# Iterating over the y axis values
for i, v in enumerate(y_vgg16):
    # Positioning the y axis labels inside the bars
    ax.text(x=i+0.03, y=v-(y_vgg16[i]*0.01), s=y_vgg16[i], 
        fontsize='large', 
        color=label_color,
        rotation='vertical',
        ha='center',
        va='top')

# Defining the y axis label and ticks
ax.set_ylabel("5-fold average accuracy")
ax.set_yticks(np.arange(0.0, 1.20, 0.2))

# Defining the x axis label and ticks
ax.set_xlabel("Keyframes")
ax.set_xticks(np.arange(len(x_labels)), x_labels)

# Defining the plot title
#ax.set_title("Keyframe influence on classification performance")

# Saving the plot
#plt.savefig("vgg16_keyframe_performance.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)

# Showing the vgg16-based model test set performance regarding the selected keyframe
plt.show()
