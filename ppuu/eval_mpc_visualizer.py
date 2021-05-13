import io
import shutil
import subprocess
from pathlib import Path

import ipywidgets as widgets
import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from torchvision import transforms


class EvalVisualizer:
    def __init__(self, output_dir=None, notebook=False):
        self.setup_mpl()
        self.transform = transforms.ToPILImage()

        self.output_dir = output_dir
        self.cost_profile_and_traj = None
        self.cost_profile = None
        self.i_data = None
        self.notebook = notebook
        self.mask_overlay_image_data = None

        if notebook:
            # In the notebook we create this once and keep it, otherwise we spawn a new one
            # for each episode.
            self.costs_plot_output = widgets.Output()

        self.images_history = []

    def setup_mpl(self):
        mpl.use("Agg")

    def update_main_image(self, image):
        if image.shape[0] == 4:
            image = image[:3] + image[3]
        image = self.transform(image)
        self.i_data = image

    def update_mask_overlay(self, overlay):
        big_image = (
            overlay[:, :3]
            .clone()
            .detach()
            .cpu()
            .mul_(255.0)
            .clamp_(0, 255)
            .type(torch.uint8)
            .permute(1, 2, 0, 3)
            .reshape(3, 117, -1)
        )
        big_image = self.transform(big_image)
        self.mask_overlay_image_data = big_image

    def update_cost_profile_and_traj(
        self, cost_profile_and_traj, cost_profile
    ):
        cost_profile_and_traj = (
            cost_profile_and_traj.clone()
            .detach()
            .cpu()
            .mul_(255.0)
            .clamp_(0, 255)
            .type(torch.uint8)
        )
        self.cost_profile_and_traj = self.transform(cost_profile_and_traj)
        self.cost_profile = cost_profile

    def update_values(self, cost, acc, turn, acc_grad=0, turn_grad=0):
        self.costs_history.append(cost)
        self.acc_history.append(acc)
        self.turn_history.append(turn)
        self.acc_grad_history.append(acc_grad)
        self.turn_grad_history.append(turn_grad)

    def draw(self):
        plt.figure(dpi=200, figsize=(15, 15))
        plt.subplot(4, 4, 3)
        plt.plot(self.costs_history)
        plt.title("cost over iterations", y=1.08)
        plt.xlabel("iterations")
        plt.ylabel("cost")

        ax = plt.subplot(4, 4, 7)
        ax.plot(self.acc_history)
        ax.set_xlabel("iterations")
        ax.set_ylabel("steering result")
        ax.set_title("acceleration over iterations", y=1.08)

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", colors="red")
        ax2.plot(self.acc_grad_history, c="red", alpha=0.5)
        ax2.set_ylabel("gradient")

        ax = plt.subplot(4, 4, 8)
        plt.plot(self.turn_history)
        plt.xlabel("iterations")
        plt.ylabel("steering result")
        plt.title("turning over iterations", y=1.08)

        ax2 = ax.twinx()
        ax2.tick_params(axis="y", colors="red")
        ax2.plot(self.turn_grad_history, c="red", alpha=0.5)
        ax2.set_ylabel("gradient")

        plt.subplot(4, 4, 4)
        if (
            self.cost_profile_and_traj is not None
            and self.cost_profile is not None
        ):
            plt.title("cost landscape", y=1.08)
            im = plt.imshow(self.cost_profile)
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            plt.gcf().colorbar(im, orientation="vertical", ax=plt.gca())

        plt.subplot(2, 4, 1)
        if self.i_data is not None:
            im = plt.imshow(self.i_data)  # show image
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)

        plt.subplot(2, 4, 2)
        if self.cost_profile_and_traj is not None:
            im = plt.imshow(self.cost_profile_and_traj)  # show traj
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)

        if self.mask_overlay_image_data is not None:
            plt.subplot(2, 1, 2)
            im = plt.imshow(
                self.mask_overlay_image_data
            )  # show planning images
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)

        plt.subplots_adjust(
            left=0.01,
            bottom=None,
            right=0.99,
            top=None,
            wspace=0.5,
            hspace=0.5,
        )

    def save_plot_to_history(self):
        io_buf = io.BytesIO()
        plt.gcf().savefig(io_buf, format="png", dpi=100)
        io_buf.seek(0)
        self.images_history.append(io_buf.getvalue())

    def show_mpl(self):
        plt.show()
        plt.close()

    def update_plot(self):
        if (
            self.i_data is not None
            or self.t_data is not None
            or self.c_data is not None
        ):
            self.costs_plot_output.clear_output(wait=True)
            with self.costs_plot_output:
                self.draw()
                self.save_plot_to_history()
                self.show_mpl()

    def step_reset(self):
        self.costs_history = []
        self.acc_history = []
        self.turn_history = []
        self.acc_grad_history = []
        self.turn_grad_history = []

    def episode_reset(self):
        if not self.notebook:
            self.costs_plot_output = widgets.Output()
        self.images_history = []
        self.step_reset()

    def save_video(self, k):
        if self.output_dir is not None:
            path = Path(self.output_dir) / "visualizer" / "images" / str(k)
            path.mkdir(exist_ok=True, parents=True)
            for i, img in enumerate(self.images_history):
                with open(path / f"{i:0>4d}.png", "wb") as f:
                    f.write(img)
            video_path = (
                Path(self.output_dir) / "visualizer" / "videos" / f"{k}.mp4"
            )
            video_path.parent.mkdir(exist_ok=True, parents=True)
            with open("/dev/null", "w") as f:
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
            # Delete images because they take up space in scratch.
            shutil.rmtree(path)
