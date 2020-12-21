import numpy as np
from scipy import signal as sg
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage


RED = 0
GREEN = 1
BLUE = 2


class TFL:

    def __init__(self, image):

        self.kernel = self.init_kernel_red()
        self.image = image

    def init_kernel_green(self):
        return np.array([[-1 / 9, -1 / 9, -1 / 9],
                  [-1 / 9, 8 / 9, -1 / 9],
                  [-1 / 9, -1 / 9, -1 / 9]])

    def init_kernel_red(self):
        return np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                         [-10 + 0j, 0 + 0j, +10 + 0j],
                         [-3 + 3j, 0 + 10j, +3 + 3j]])


    def print_image(self, image):
        img = Image.fromarray(image, 'RGB')
        # img.save('my.png')
        img.show()

    def convolve_by_color(self, image, color):
        gray_image = image[:, :, color]
        grad = signal.convolve2d(gray_image.T, self.kernel, boundary='symm', mode='same') #TODO check pading
        # self.print_convolve(image, grad)

        return grad

    def filter_by_arg(self, grad, arg):
        green = np.argwhere(grad > arg)

        return np.array(green[:, :1]).ravel(), np.array(green[:, 1:]).ravel()

    def print_convolve(self, image, grad):
        fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))

        ax_orig.imshow(image, cmap='gray')
        ax_orig.set_axis_off()

        ax_mag.imshow(np.absolute(grad.T), cmap='gray')
        ax_mag.set_axis_off()

        fig.show()

    def hight(self):
        return self.image.shape[0]

    def weight(self):
        return self.image.shape[1]

    def pyramid(self):
        pass

    def resize_image(self, num):
        size = self.weight() // num, self.hight()// num
        return np.array(self.open_image.resize(size))


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    tfl = TFL(c_image)

    green_grad = tfl.convolve_by_color(tfl.image, GREEN)

    red_grad = tfl.convolve_by_color(tfl.image, RED)

    print(type(green_grad))


    g = ndimage.maximum_filter(np.real(green_grad), size=20)


    green_x ,green_y = tfl.filter_by_arg(green_grad, 1700)
    red_x, red_y = tfl.filter_by_arg(red_grad, 1700)

    return red_x, red_y, green_x, green_y