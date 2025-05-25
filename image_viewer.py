import sys
import numpy as np
import matplotlib.pyplot as plt

def show_ubyte_image(img_path):
    with open(img_path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    if arr.size != 28 * 28:
        raise ValueError(f"Expected 784 bytes, got {arr.size}")
    img = arr.reshape((28, 28))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("MNIST image preview")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python show_mnist_image.py path/to/image.ubyte")
        sys.exit(1)
    show_ubyte_image(sys.argv[1])
