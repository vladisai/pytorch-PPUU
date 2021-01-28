import ipywidgets as widgets
import torch
import PIL
from IPython.display import display
import io
import numpy as np
import bqplot
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from pathlib import Path
import subprocess


from torchvision import transforms


class EvalVisualizer:
    def __init__(self, output_dir=None):
        mpl.use("Agg")
        self.output_dir = output_dir
        self.t_data = None
        self.c_data = None
        self.i_data = None
        self.images_history = []

    def update(self, image):
        if image.shape[0] == 4:
            image = image[:3] + image[3]
        image = self.transform(image)
        self.i_data = image

    def update_c(self, overlay):
        big_image = (
            overlay.clone()
            .detach()
            .cpu()
            .mul_(255.0)
            .clamp_(0, 255)
            .type(torch.uint8)
            .permute(1, 2, 0, 3)
            .reshape(3, 117, -1)
        )
        big_image = self.transform(big_image)
        self.c_data = big_image

    def update_t(self, image, data):
        image = (
            image.clone()
            .detach()
            .cpu()
            .mul_(255.0)
            .clamp_(0, 255)
            .type(torch.uint8)
        )
        image = self.transform(image)
        self.t_data = image
        self.t_data_no_traj = data

    def update_values(self, cost, acc, turn, acc_grad=0, turn_grad=0):
        self.costs_history.append(cost)
        self.acc_history.append(acc)
        self.turn_history.append(turn)
        self.acc_grad_history.append(acc_grad)
        self.turn_grad_history.append(turn_grad)

    def update_plot(self):
        if (
            self.i_data is not None
            and self.t_data is not None
            and self.c_data is not None
        ):
            self.costs_plot_output.clear_output()
            with self.costs_plot_output:
                plt.figure(dpi=200, figsize=(10, 10))
                plt.subplot(4, 4, 3)
                plt.plot(self.costs_history)
                plt.title('cost over iterations')
                plt.xlabel('iterations')
                plt.ylabel('cost')

                ax = plt.subplot(4, 4, 7)
                ax.plot(self.acc_history)
                ax.set_xlabel('iterations')
                ax.set_ylabel('steering result')
                ax.set_title('acceleration over iterations')

                ax2 = ax.twinx()
                ax2.tick_params(axis='y', colors='red')
                ax2.plot(self.acc_grad_history, c='red', alpha=0.5)
                ax2.set_ylabel('gradient')

                ax = plt.subplot(4, 4, 8)
                plt.plot(self.turn_history)
                plt.xlabel('iterations')
                plt.ylabel('steering result')
                plt.title('turning over iterations')

                ax2 = ax.twinx()
                ax2.tick_params(axis='y', colors='red')
                ax2.plot(self.turn_grad_history, c='red', alpha=0.5)
                ax2.set_ylabel('gradient')

                plt.subplot(4, 4, 4)
                if self.t_data is not None:
                    plt.title('cost landscape')
                    im = plt.imshow(self.t_data_no_traj)
                    im.axes.get_xaxis().set_visible(False)
                    im.axes.get_yaxis().set_visible(False)
                    plt.gcf().colorbar(im, orientation="vertical", ax=plt.gca())

                plt.subplot(2, 4, 1)
                im = plt.imshow(self.i_data) # show image
                im.axes.get_xaxis().set_visible(False)
                im.axes.get_yaxis().set_visible(False)

                plt.subplot(2, 4, 2)
                im = plt.imshow(self.t_data) # show traj
                im.axes.get_xaxis().set_visible(False)
                im.axes.get_yaxis().set_visible(False)

                plt.subplot(2, 1, 2)
                im = plt.imshow(self.c_data) # show planning images
                im.axes.get_xaxis().set_visible(False)
                im.axes.get_yaxis().set_visible(False)


                # saving to history for replay
                io_buf = io.BytesIO()
                plt.gcf().savefig(io_buf, format='png', dpi=100)
    #             io_buf.seek(0)
    #             img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
    #                                  newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
                io_buf.seek(0)
    #             self.images_history.append(np.frombuffer(io_buf.getvalue(), dtype=np.uint8))
                self.images_history.append(io_buf.getvalue())
                plt.subplots_adjust(left=0.01, bottom=None, right=0.99, top=None, wspace=0.5, hspace=0.4)

                plt.show()
                plt.close()

    def step_reset(self):
        self.costs_history = []
        self.acc_history = []
        self.turn_history = []
        self.acc_grad_history = []
        self.turn_grad_history = []

    def episode_reset(self):
        self.images_history = []
        self.transform = transforms.ToPILImage()
        self.costs_plot_output = widgets.Output()
        self.step_reset()

    def save_video(self, k):
        path = Path(self.output_dir) / "visualizer" / "images" / str(k)
        path.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(self.images_history):
            with open(path / f"{i:0>4d}.png", "wb") as f:
                f.write(img)
        video_path = (
            Path(self.output_dir) / "visualizer" / "videos" / f"{k}.mp4"
        )
        video_path.parent.mkdir(exist_ok=True, parents=True)
        with open("/dev/null", 'w') as f:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-r",
                    "10",
                    "-i",
                    f"{path}/%04d.png",
                    "-vcodec",
                    "mpeg4",
                    "-q:v",
                    "10",
                    "-y",
                    video_path,
                ],
                stdout=f,
                stderr=f,
            )
