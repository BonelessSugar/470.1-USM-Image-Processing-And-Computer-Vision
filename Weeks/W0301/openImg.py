#COS 470 by Xin Zhang
import cv2

# Load the image
image = cv2.imread('dog.jpg')

print(image[5,1])

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
else:
    print("Image loaded successfully.")

    # Display the image shape (height, width, number of channels)
    print(f"Image shape: {image.shape}")

    # Optionally, display the image in a window
    cv2.imshow('Loaded Image', image)

    # Waits indefinitely for a key press
    key = cv2.waitKey(0)
    if key == ord('q'):  # Checks if 'q' was pressed
        print("You pressed 'q'. Exiting...")
    # # if you do not want to detect specific key presses, you can simplify this as:
    # cv2.waitKey(0)

    cv2.destroyAllWindows()  # Closes the image window


