#!/usr/bin/python

""" tool for displaying data """

from PyQt4 import QtCore
from PyQt4 import QtGui as qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import itertools
import matplotlib.pyplot as pyplot


class MultiTabPlot(qt.QMainWindow):

    def __init__(self, parent=None, figures=None, labels=None):
        qt.QMainWindow.__init__(self, parent)
        self.main = qt.QWidget()
        self.main_layout = main_layout = qt.QVBoxLayout()
        self.main.setLayout(main_layout)
        self.setCentralWidget(self.main)
        self.tabs = qt.QTabWidget(self.main)
        self.canvases = []
        self.setup_tabs_and_canvases(figures, labels)
        main_layout.addWidget(self.tabs)

    def setup_tabs_and_canvases(self, figures, labels):
        labels = [] if labels is None else labels
        figures = [Figure()] if figures is None else figures
        for (figure, label) in itertools.izip_longest(figures, labels):
            self.setup_tab_and_canvas(figure, label)

    def setup_tab_and_canvas(self, figure=None, label=None):
        if figure is None:
            figure = Figure()
            figure.add_subplot(111)
        canvas = figure.canvas if figure.canvas else FigureCanvas(figure)
        canvas.setParent(self.tab_widget)
        canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        label = 'Tab %i' % (self.tab_widget.count()+1) if label is None else label
        self.tabs.addTab(canvas, label)
        self.canvases.append(canvas)


def create_scatter_plot(data_dict, fx_name, fy_name, highlight = dict()):
    """
    draw bivariate scatterplot
    :param data_dict: data dictionary to plot; must be iterable of dictionary, first level represents
        observations, second (dictionary) level represents features with the dictionary key being the feature name;
    :param fx_name: feature name (second-level dictionary key) to plot on the x axis; must be numerical feature
    :param fy_name: feature name (second-level dictionary key) to plot on the y axis; must be numerical feature
    :param highlight: if set, highlights a number of observations (first-level indices of data_dict) in specific color;
        * if a list is supplied, the respective observations are highlighted in red
        * if a dict is supplied, the key of the dict must be a color, the value must be a list of indices to
          highlight in that color
    :return:
    """
    figure, axes = pyplot.subplots()
    for k in data_dict:
        axes.scatter(data_dict[k][fx_name], data_dict[k][fy_name])
    axes.xlabel(fx_name)
    axes.ylabel(fy_name)
    # TODO add handling of "highlight" parameter
    return axes


def create_hist_plot(data_dict, fx_name):
    figure, axes = pyplot.subplots()
    data_column = {k: data_dict[k][fx_name] for k in data_dict}
    axes.hist(list(data_column))
    return axes