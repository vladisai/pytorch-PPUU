"""Contains the tabs for different visualizations.
Each tab is a standalone visualization."""
import ipywidgets as widgets
from DataReader import DataReader
from Widgets import DimensionalityReductionPlot  # ExperimentEntryView,
from Widgets import (
    EpisodeReview,
    HeatMap,
    HeatMapComparison,
    Picker,
    PiePlot,
    PolicyComparison,
)


class EpisodeReviewTab(widgets.VBox):
    """A tab for visualizing model's performance on an episode.
    Model is picked with dropdown picker"""

    def __init__(self):
        self.episode_review = EpisodeReview()
        self.picker = Picker(Picker.EPISODE_LEVEL, widget=self.episode_review)
        super(EpisodeReviewTab, self).__init__(
            [self.picker, self.episode_review]
        )


class PiePlotTab(widgets.VBox):
    """A tab for visualizing model's success rate with pie chart.
    Model is picked with dropdown picker"""

    def __init__(self):
        self.pie_plot = PiePlot()
        self.picker = Picker(Picker.MODEL_LEVEL, widget=self.pie_plot)
        super(PiePlotTab, self).__init__([self.picker, self.pie_plot])


class DimensionalityReductionPlotTab(widgets.VBox):
    """A tab for visualizing episodes using with scatter plot and
    dimensionality reduction."""

    def __init__(self):
        self.episode_review = EpisodeReview()
        self.dimensionality_reduction_plot = DimensionalityReductionPlot(
            widget=self.episode_review
        )
        self.picker = Picker(
            Picker.MODEL_LEVEL, widget=self.dimensionality_reduction_plot
        )
        super(DimensionalityReductionPlotTab, self).__init__(
            [
                self.picker,
                self.dimensionality_reduction_plot,
                self.episode_review,
            ]
        )


class HeatMapTab(widgets.VBox):
    """A tabl showing episodes 'difficulty' for a given model
    Each cell in the heatmap represents how hard an episode is"""

    def __init__(self):
        self.heat_map = HeatMap()
        self.picker = Picker(Picker.EXPERIMENT_LEVEL, widget=self.heat_map)

        super(HeatMapTab, self).__init__([self.picker, self.heat_map])


class HeatMapComparisonTab(widgets.VBox):
    """Compares two models in their performance for different
    episodes, enabling us to see which episodes were failing or successful
    for two models.
    Color coding:
        orange - both models failed.
        red - first model succeeded, second failed.
        green - first model failed, second succeeded,
        blue - both models succeeded.
    """

    def __init__(self):

        self.heat_map = HeatMapComparison()
        self.picker0 = Picker(
            Picker.MODEL_LEVEL, callback=self.get_callback(0)
        )
        self.picker1 = Picker(
            Picker.MODEL_LEVEL, callback=self.get_callback(1)
        )

        self.picked_values = [None, None]

        self.pickers_hbox = widgets.HBox([self.picker0, self.picker1])

        super(HeatMapComparisonTab, self).__init__(
            [self.pickers_hbox, self.heat_map]
        )

    def get_callback(self, index):
        def callback(episode, seed, step):
            self.picked_values[index] = (episode, seed, step)
            if (
                self.picked_values[0] is not None
                and self.picked_values[1] is not None
            ):
                self.heat_map.update(
                    self.picked_values[0], self.picked_values[1]
                )

        return callback


class PolicyComparisonTab(widgets.VBox):
    """Tab for comparing success rates across checkpoints of different
    experiments.

    Experiments are chosen using a multiselect widget.
    """

    def __init__(self):
        self.experiment_multiselect = widgets.SelectMultiple(
            options=list(DataReader.find_experiments()),
            description="Experiments:",
            disabled=False,
            value=[],
        )

        self.policy_comparison = PolicyComparison()

        def experiment_multiselect_change_callback(change):
            if change.name == "value" and change.new is not None:
                self.policy_comparison.update(change.new)

        self.experiment_multiselect.observe(
            experiment_multiselect_change_callback, type="change"
        )
        super(PolicyComparisonTab, self).__init__(
            [self.experiment_multiselect, self.policy_comparison]
        )
