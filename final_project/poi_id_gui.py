#!/usr/bin/python

""" tool for displaying data """
import itertools

import numpy
import inspect
import sys
from PyQt4 import QtCore
from PyQt4 import QtGui as qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as pyplot
import math


class PlotSpinnerWidget(qt.QWidget):

    def __init__(self, plot_drawer, QWidget_parent=None, spinner_value=10):
        qt.QWidget.__init__(self, QWidget_parent)
        self.spinner_value = spinner_value
        self.plot_drawer = plot_drawer
        self.layout = layout = qt.QVBoxLayout()
        self.spinner_widget = spinner_widget = qt.QSpinBox(self)
        spinner_widget.setValue(spinner_value)
        spinner_widget.valueChanged.connect(self.changeEvent)
        layout.addWidget(self.spinner_widget)
        self.figure_widget = plot_drawer.get_widget()
        layout.addWidget(self.figure_widget)
        self.setLayout(layout)
        self.draw_plot()

    def draw_plot(self):
        self.plot_drawer.draw_plot(self.spinner_value)

    def changeEvent(self, QEvent):
        self.spinner_value = self.spinner_widget.value()
        self.draw_plot()


class PlotDrawer():

    def __init__(self):
        self.figure, self.axes = pyplot.subplots()

    @staticmethod
    def create_vals_by_dict(data_dict, f_name, row_condition = (lambda x: True)):
        vals = []
        for k in data_dict:
            if row_condition(data_dict[k]):
                val = float(data_dict[k][f_name])
                vals.append(val if not numpy.isnan(val) else 0.0)
        return vals

    def get_widget(self):
        widget = self.figure.canvas
        return widget


class HistDrawer(PlotDrawer):

    def __init__(self):
        PlotDrawer.__init__(self)
        self.xvals = []
        self.labels = []
        self.xlabel = ""

    def add_xvals_by_dict(self, data_dict, fx_name, row_condition=(lambda x: True), label=""):
        self.xvals.append(PlotDrawer.create_vals_by_dict(data_dict, fx_name, row_condition))
        self.labels.append(label)
        self.xlabel = fx_name

    def draw_plot(self, dynamic_value):
        self.axes.clear()
        self.axes.hist(self.xvals, bins=dynamic_value, stacked=True, label=self.labels)
        self.axes.legend()
        self.axes.set_xlabel(self.xlabel)
        self.figure.canvas.draw()


class ScatterDrawer(PlotDrawer):

    def __init__(self):
        PlotDrawer.__init__(self)
        self.xvals = []
        self.yvals = []
        self.labels = []
        self.colors = []
        self.xlabel = ""
        self.ylabel = ""

    def add_xyvals_by_dict(self, data_dict, fx_name, fy_name, row_condition=(lambda x: True), label="", color=None):
        color = "blue" if color is None else color
        self.xvals.append(PlotDrawer.create_vals_by_dict(data_dict, fx_name, row_condition))
        self.yvals.append(PlotDrawer.create_vals_by_dict(data_dict, fy_name, row_condition))
        self.labels.append(label)
        self.colors.append(color)
        self.xlabel = fx_name
        self.ylabel = fy_name

    def draw_plot(self, dynamic_value):
        self.axes.clear()
        for xrow, yrow, label, color in zip(self.xvals, self.yvals, self.labels, self.colors):
            self.axes.scatter(xrow, yrow, label=label, color=color)
        self.axes.legend()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.figure.canvas.draw()


class MultiFigureWindow(qt.QMainWindow):

    def __init__(self, data_dict, parent=None, figures=None, labels=None):
        qt.QMainWindow.__init__(self, parent)
        self.data_dict = data_dict
        self.main = qt.QWidget()
        self.main_layout = main_layout = qt.QVBoxLayout()
        self.main.setLayout(main_layout)
        self.setCentralWidget(self.main)
        self.tabs = qt.QTabWidget(self.main)
        self.canvases = []
        self._add_figures(figures, labels)
        main_layout.addWidget(self.tabs)

    def _add_figures(self, figures, labels):
        labels = [] if labels is None else labels
        figures = [] if figures is None else figures
        for (figure, label) in zip(figures, labels):
            self._add_figure(figure, label)

    def _add_widget(self, widget=None, label=None):
        widget.setParent(self.tabs)
        widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        label = 'Tab %i' % (self.tab_widget.count()+1) if label is None else label
        self.tabs.addTab(widget, label)
        self.canvases.append(widget)

    def _add_figure(self, figure=None, label=None):
        if figure is None:
            figure = Figure()
            figure.add_subplot(111)
        widget = figure.canvas if figure.canvas else FigureCanvas(figure)
        self._add_widget(self, widget, label)

    def add_hist_drawer(self, fx_name, row_conditions=None):
        row_conditions = {"data":(lambda x: True)} if row_conditions == None else row_conditions
        drawer = HistDrawer()
        for label in row_conditions:
            drawer.add_xvals_by_dict(self.data_dict, fx_name, row_conditions[label], label)
        widget = PlotSpinnerWidget(drawer, self, 10)
        self._add_widget(widget, fx_name)

    def add_scatter_drawer(self, fx_name, fy_name, row_conditions=None, colors=None):
        color = [] if colors is None else colors
        row_conditions = {"data":(lambda x: True)} if row_conditions == None else row_conditions
        drawer = ScatterDrawer()
        for label, color in itertools.izip_longest(row_conditions, colors):
            drawer.add_xyvals_by_dict(self.data_dict, fx_name, fy_name, row_conditions[label], label, color)
        widget = PlotSpinnerWidget(drawer, self, 10)
        self._add_widget(widget, fx_name + "*" + fy_name)


class PoiIdGui():

    def __init__(self, gui_mode):
        self.gui_mode = gui_mode
        self.application = None
        self.window_univariate_analysis = None
        self.window_bivariate_analysis = None
        if self.gui_mode != None:
            self.application = qt.QApplication(sys.argv)
            pyplot.ioff()

    def exec_(self):
        if self.gui_mode != None:
            if self.window_univariate_analysis is not None:
                self.window_univariate_analysis.show()
            if self.window_bivariate_analysis is not None:
                self.window_bivariate_analysis.show()
            self.application.exec_()

    def prepare_univariate_analysis(self, data_dict, x_features, conditions, no_matter_what=False):
        if self.gui_mode == "univariate_analysis" or no_matter_what:
            window = MultiFigureWindow(data_dict)
            for x_feature in x_features:
                window.add_hist_drawer(x_feature, conditions)
            self.window_univariate_analysis = window

    def prepare_bivariate_analysis(self, data_dict, xy_features, conditions, colors=None, no_matter_what=False):
        if self.gui_mode == "bivariate_analysis" or no_matter_what:
            window = MultiFigureWindow(data_dict)
            for xy_feature in xy_features:
                window.add_scatter_drawer(xy_feature[0], xy_feature[1], conditions, colors)
            self.window_bivariate_analysis = window

