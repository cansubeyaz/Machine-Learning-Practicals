import cv2
import matplotlib.pyplot as plt

rgb_image = cv2.imread(r"C:\Users\Lenovo\Desktop\PAML_2024\pepper.jpg")
lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)

l_channel, a_channel, b_channel = cv2.split(lab_image)

modified_lab = cv2.merge((l_channel, a_channel, b_channel))

# Optionally, convert back to RGB to display or save the result
# modified_rgb_image = cv2.cvtColor(modified_lab, cv2.COLOR_Lab2RGB)

fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
ax[0].axis("off")
ax[0].set_title("Input Image")
ax[0].imshow(rgb_image)

ax[1].axis("off")
ax[1].set_title("Lab Color Space")
ax[1].imshow(lab_image)

plt.tight_layout()
plt.show()

# Save or display the processed image
cv2.imwrite("processed_pepper.jpg", modified_lab)
print("Image processed and saved successfully.")