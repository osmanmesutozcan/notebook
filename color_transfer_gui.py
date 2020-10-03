import tkinter as tk
import tkinter.filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from computer_vision.color_transfer import color_transfer


def read_image():
    path = tkinter.filedialog.askopenfilename()
    return Image.open(path)


def display_image(image, frame, size):
    image= image.resize(size, Image.ANTIALIAS)
    tk_image = ImageTk.PhotoImage(image)

    label = tk.Label(frame, image=tk_image)
    label.image = tk_image
    label.pack()

    return image


def read_and_display_image(frame, size):
    img = read_image()
    display_image(img, frame, size)
    return img


def pil_to_cv(image):
    pil_image = image.convert('RGB')
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR


def cv_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


class TGui:
    def __init__(self):
        self.source_image = None
        self.target_image = None

        root = tk.Tk()
        root.geometry('1200x850')
        root.resizable(height=False, width=False)
        self.root = root

        self.top_frame = tk.Frame(root, width=1200, height=50)
        self.top_frame.grid(row=0)

        self.center_frame = tk.Frame(root, width=1200, height=800, bg='white')
        self.center_frame.grid(row=1, columnspan=2)

        self.left_frame = tk.Frame(self.center_frame, width=400, height=800, bg='green')
        self.left_frame.grid(row=1, column=0, rowspan=2)

        self.right_frame = tk.Frame(self.center_frame, width=800, height=800, bg='grey')
        self.right_frame.grid(row=1, column=1)

        self.left_top_frame = tk.Frame(self.left_frame, width=400, height=400, bg='red')
        self.left_top_frame.grid(row=0)

        self.left_bottom_frame = tk.Frame(self.left_frame, width=400, height=400, bg='blue')
        self.left_bottom_frame.grid(row=1)

        select_source_btn = tkinter.Button(self.top_frame, text='Select Source', command=self.display_source_image)
        select_source_btn.place(height=50, width=100, x=0)

        select_target_btn = tkinter.Button(self.top_frame, text='Select Target', command=self.display_target_image)
        select_target_btn.place(height=50, width=100, x=100)

        process_btn = tkinter.Button(self.top_frame, text='Process', command=self.process_image)
        process_btn.place(height=50, width=100, x=200)

    def display_source_image(self):
        self.source_image = read_and_display_image(self.left_top_frame, (400, 400))

    def display_target_image(self):
        self.target_image = read_and_display_image(self.left_bottom_frame, (400, 400))

    def process_image(self):
        src = pil_to_cv(self.source_image)
        tgt = pil_to_cv(self.target_image)

        transferred = color_transfer(src, tgt)
        display_image(cv_to_pil(transferred), self.right_frame, (800, 800))


if __name__ == '__main__':
    app_gui = TGui()
    app_gui.root.mainloop()
