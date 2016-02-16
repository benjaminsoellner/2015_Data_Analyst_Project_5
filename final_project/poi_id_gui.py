#!/usr/bin/python
# coding=utf-8

"""
The "Classify POIs from the Enron Scandal" Exploratory Data Analysis GUI

GUI for performing univariate data analysis (histogram plots) and bivariate
data analyis (scatter plots) on numerical data sets in a window with multiple
tabbed graphs and a few controls to modify the appearance of said graph.

Author:
    Benjamin Soellner <post@benkku.com>
    from the "Intro to Machine Learning" Class
    of Udacity's "Data Analyst" Nanodegree
"""

import itertools
import numpy
import sys
from PyQt4 import QtCore
from PyQt4 import QtGui as qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as pyplot


# ------------
# CLASS PlotSpinnerWidget
# ------------

class PlotSpinnerWidget(qt.QWidget):

    def __init__(self, plot_drawer, QWidget_parent=None, spinner_value=10):
        """A widget (QWidget) which shows a graph drawn by a PlotDrawer and a
        spin box (QSpinBox).

        :param plot_drawer: Should be a PlotDrawer object which is responsive
            may change dynamically change/redraw if the value in the spin box
            changes
        :param QWidget_parent: The parent QWidget which will hold this widget.
        :param spinner_value: The initial spin box value
        """
        # Call super class
        qt.QWidget.__init__(self, QWidget_parent)
        # Initialize simple object properties
        self.spinner_value = spinner_value
        self.plot_drawer = plot_drawer
        # Create widgets and add them to object properties, set up layout,
        # default values etc.
        self.layout = layout = qt.QVBoxLayout()
        self.spinner_widget = spinner_widget = qt.QSpinBox(self)
        spinner_widget.setValue(spinner_value)
        layout.addWidget(self.spinner_widget)
        layout.addWidget(plot_drawer.get_widget())
        self.setLayout(layout)
        # If spinner value changes, catch change by this object "changeEvent"
        # socket (method)
        spinner_widget.valueChanged.connect(self.changeEvent)
        # Draw initial plot
        self.draw_plot()

    def draw_plot(self):
        """Redraw the plot in this widget.
        :return: None
        """
        # Delegate to plot_drawer strategy object
        self.plot_drawer.draw_plot(self.spinner_value)

    def changeEvent(self, QEvent):
        """Called if the spin box value of the Widget changed; redraws the plot
        in this case.
        :param QEvent: the change event
        :return: None
        """
        self.spinner_value = self.spinner_widget.value()
        self.draw_plot()


# ------------
# CLASS PlotDrawer
# ------------

class PlotDrawer():

    def __init__(self):
        """ A strategy super-class for drawing a bunch of data in a QWidget
        showing a pyplot with some limited control over a "dynamic_value" how
        this data should be drawn.
        """
        # Initialize pyplot
        self.figure, self.axes = pyplot.subplots()
        # For multiple data rows, here are some variables for data row
        # labels and colors
        self.labels = []
        self.colors = []


    def get_widget(self):
        """Easy access to return the graph canvas widget.
        :return: The graph canvas widget.
        """
        widget = self.figure.canvas
        return widget

    def draw_plot(self, dynamic_value):
        """This should be overridden with code to draw the actual pyplot.
        :param dynamic_value: provides some control about how the pyplot
            should be drawn (e.g., number of histogram butckets, alpha
            value of scatter plot data points etc.)
        :return: None
        """
        pass

    @staticmethod
    def create_vals_by_dict(data_dict, f_name,
                            row_condition = (lambda x: True)):
        """Static method to extract a feature f_name from a data dictionary
        data_dict, but only for those data points which observe a row_condition.
        :param data_dict: the data dictionary in the form
            data_dict['row_key']['feature_name'] = value
        :param f_name: the feature_name you want to focus on for all rows
        :param row_condition: supply a 1-parameter function or lambda expression
            here. Allows to only extract those rows where
            row_condition(data_dict['row_key']) returns True.
        :return: List of data values for given feature and row_condition;
            converts all values to floats and NaN values to 0.0
        """
        vals = []
        for k in data_dict:
            if row_condition(data_dict[k]):
                val = float(data_dict[k][f_name])
                vals.append(val if not numpy.isnan(val) else 0.0)
        return vals


# ------------
# CLASS HistDrawer
# ------------

class HistDrawer(PlotDrawer):

    def __init__(self):
        """A strategy class for drawing a list of values as histogram in a
        QWidget with the number of bins being dynamically controllable and
        the histogram re-drawable in the same QWidget when the number of bins
        needed ("dynamic_value") changes.
        """
        PlotDrawer.__init__(self)
        self.xvals = []
        self.xlabel = ""

    def add_xvals_by_dict(self, data_dict, fx_name,
                          row_condition=(lambda x: True), label="", color=None):
        """Add a list of values to the histogram with given label and color.
        :param data_dict: the data_dict where to draw the data from of form
            data_dict['row_key']['feature_name'] = value
        :param fx_name: the feature_name of the data_dict you want to use.
        :param row_condition: allows filtering the data_dict for certain values.
            supply a one-argument function or lambda expression; all values from
            the data_dict will be used where row_condition(data_dict['row_key'])
            returns True.
        :param label: the label to give this list of values in the histogram
            legend
        :param color: the color to give this list of values in the histogram
            legend
        :return: None
        """
        color = "blue" if color is None else color
        self.xvals.append(PlotDrawer.create_vals_by_dict(data_dict, fx_name,
                                                         row_condition))
        self.labels.append(label)
        self.colors.append(color)
        self.xlabel = fx_name

    def draw_plot(self, dynamic_value):
        """ (Re-)Draws the histogram with a defined number of bins
        :param dynamic_value: Lets you control the number of bins.
        :return: None
        """
        self.axes.clear()
        self.axes.hist(self.xvals, bins=dynamic_value, stacked=True,
                       label=self.labels, color=self.colors)
        self.axes.legend()
        self.axes.set_xlabel(self.xlabel)
        self.figure.canvas.draw()


# ------------
# CLASS ScatterDrawer
# ------------

class ScatterDrawer(PlotDrawer):

    def __init__(self):
        """A strategy class for drawing a list of x- and y-values as scatterplot
        in a QWidget with the alpha-percentage being dynamically controllable
        and the scatter plot re-drawable in the same QWidget when the
        alpha-value ("dynamic_value") changes.
        """
        PlotDrawer.__init__(self)
        self.xvals = []
        self.yvals = []
        self.xlabel = ""
        self.ylabel = ""

    def add_xyvals_by_dict(self, data_dict, fx_name, fy_name,
                           row_condition=(lambda x: True), label="",
                           color=None):
        """Add a list of points to the scatter plot with given label and color.
        :param data_dict: the data_dict where to draw the data from of form
            data_dict['row_key']['feature_name'] = value
        :param fx_name: the feature_name you want to use as x-cordinate
        :param fy_name: the feature_name you want to use as y-cordinate
        :param row_condition: allows filtering the data_dict for certain values.
            supply a one-argument function or lambda expression; all values from
            the data_dict will be used where row_condition(data_dict['row_key'])
            returns True.
        :param label: the label to give this list of values in the histogram
            legend
        :param color: the color to give this list of values in the histogram
            legend
        :return: None
        """
        color = "blue" if color is None else color
        self.xvals.append(PlotDrawer.create_vals_by_dict(data_dict, fx_name,
                                                         row_condition))
        self.yvals.append(PlotDrawer.create_vals_by_dict(data_dict, fy_name,
                                                         row_condition))
        self.labels.append(label)
        self.colors.append(color)
        self.xlabel = fx_name
        self.ylabel = fy_name

    def draw_plot(self, dynamic_value):
        """ (Re-)Draws the scatter plot with a defined alpha-percentage of the
        data points.
        :param dynamic_value: Lets you control the alpha-percentage.
        :return: None
        """
        self.axes.clear()
        for xrow, yrow, label, color in zip(self.xvals, self.yvals,
                                            self.labels, self.colors):
            self.axes.scatter(xrow, yrow, label=label, color=color,
                              alpha=float(dynamic_value)/100)
        self.axes.legend()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.figure.canvas.draw()


# ------------
# CLASS ScatterDrawer
# ------------

class MultiFigureWindow(qt.QMainWindow):

    def __init__(self, data_dict, parent=None):
        """Creates a window for exploratory data analysis with a few tabs
        showing different graphs.
        :param data_dict: The data_dict holding the data in the form
            data_dict['row_key']['feature_name'] = value
        :param parent: The parent QWidget or window (can be "None")
        """
        # Call super
        qt.QMainWindow.__init__(self, parent)
        # Set simple properties
        self.data_dict = data_dict
        # Create tabs
        self.tabs = tabs = qt.QTabWidget(self)
        self.setCentralWidget(tabs)

    def _add_widget(self, widget=None, label=None):
        """Helper function to add a new widget to the tab
        :param widget: the widget to add to the tab
        :param label: the label to give the tab
        :return: None
        """
        widget.setParent(self.tabs)
        widget.setFocusPolicy(QtCore.Qt.ClickFocus)
        if label is None:
            label = 'Tab %i' % (self.tab_widget.count()+1)
        self.tabs.addTab(widget, label)

    def add_hist_drawer(self, fx_name, row_conditions=None, colors=None):
        """Adds a histogram drawer to the window as tab.
        :param fx_name: the feature to explore in the historgram
        :param row_conditions: a dictionary of facetting conditions to
            filter data_dict entries and show them as separate data rows
            in the histogram; this dictionary must contain as key a
            human-readable description of the condition and as value the
            condition itself; the condition must be a one-parameter function.
            The drawer will call row_conditions['key'](data_dict['row']) and
            only plot the value if this evaluates to True.
        :param colors: a list of colors to shade the different data rows
            supplied by row_conditions in.
        """
        # Set default parameters
        colors = [] if colors is None else colors
        if row_conditions is None:
            row_conditions = {"data":(lambda x: True)}
        # Create drawer strategy and add x-values
        drawer = HistDrawer()
        for label, color in itertools.izip_longest(row_conditions, colors):
            drawer.add_xvals_by_dict(self.data_dict, fx_name,
                                     row_conditions[label], label, color)
        # Wrap up in widget and add to tabs
        widget = PlotSpinnerWidget(drawer, self, 10)
        self._add_widget(widget, fx_name)

    def add_scatter_drawer(self, fx_name, fy_name, row_conditions=None,
                           colors=None):
        """Adds a scatter plot drawer to the window as tab.
        :param fx_name: the feature to use as a x-value for data points in the
            scatter plot
        :param fy_name: the feature to use as a y-value for data points in the
            scatter plot
        :param row_conditions: a dictionary of facetting conditions to
            filter data_dict entries and show them as separate data rows
            in the histogram; this dictionary must contain as key a
            human-readable description of the condition and as value the
            condition itself; the condition must be a one-parameter function.
            The drawer will call row_conditions['key'](data_dict['row']) and
            only plot the value if this evaluates to True.
        :param colors: a list of colors to shade the different data rows
            supplied by row_conditions in.
        """
        # Set default parameters
        colors = [] if colors is None else colors
        if row_conditions is None:
            row_conditions = {"data":(lambda x: True)}
        # Create drawer strategy and add x-values and y-values
        drawer = ScatterDrawer()
        for label, color in itertools.izip_longest(row_conditions, colors):
            drawer.add_xyvals_by_dict(self.data_dict, fx_name, fy_name,
                                      row_conditions[label], label, color)
        # Wrap up in widget and add to tabs
        widget = PlotSpinnerWidget(drawer, self, 10)
        self._add_widget(widget, fx_name + "*" + fy_name)


# ------------
# CLASS PoiIdGui
# ------------

class PoiIdGui():

    def __init__(self, gui_mode):
        """The GUI to explore the POI identifier variables in a univariate
        or bi-variate way.
        :param gui_mode: set to 'univariate_analysis' or 'bivariate_analysis'
            depending on the analysis you want to do.
        """
        self.gui_mode = gui_mode
        self.application = None
        self.window_univariate_analysis = None
        self.window_bivariate_analysis = None
        if self.gui_mode != None:
            self.application = qt.QApplication(sys.argv)
            pyplot.ioff()

    def exec_(self, no_matter_what=False):
        """Shows the windows of the gui_mode with which this object was
        created and runs the respective QtApplication. Call this as the last
        method of the script since running the QtApplication will block all
        other code.
        :param no_matter_what: Overrides the gui_mode of the object and just
            show all windows for which the prepare_* method was called.
        :return: None
        """
        if self.gui_mode != None:
            if self.window_univariate_analysis is not None:
                self.window_univariate_analysis.show()
            if self.window_bivariate_analysis is not None:
                self.window_bivariate_analysis.show()
            self.application.exec_()

    def prepare_univariate_analysis(self, data_dict, x_features,
                                    conditions, colors=None,
                                    no_matter_what=False):
        """
        Prepare the univariate feature analysis window if PoiIdGui was
        instantiated with gui_mode set as "univariate_analysis"
        :param data_dict: The data_dict holding the data in the form
            data_dict['row_key']['feature_name'] = value
        :param x_features: list of features to explore in histogram graphs
        :param conditions: a dictionary of facetting conditions to
            filter data_dict entries and show them as separate data rows
            in the histogram; this dictionary must contain as key a
            human-readable description of the condition and as value the
            condition itself; the condition must be a one-parameter function.
            The drawer will call row_conditions['key'](data_dict['row']) and
            only plot the value if this evaluates to True.
        :param colors: a list of colors to shade the different data rows
            supplied by row_conditions in.
        :param no_matter_what: if True, ignore gui_mode and prepare window
            anyway
        :return: None
        """
        if self.gui_mode == "univariate_analysis" or no_matter_what:
            window = MultiFigureWindow(data_dict)
            for x_feature in x_features:
                window.add_hist_drawer(x_feature, conditions, colors)
            self.window_univariate_analysis = window

    def prepare_bivariate_analysis(self, data_dict, xy_features,
                                   conditions, colors=None,
                                   no_matter_what=False):
        """
        Prepare the bivariate feature analysis window if PoiIdGui was
        instantiated with gui_mode set as "bivariate_analysis"
        :param data_dict: The data_dict holding the data in the form
            data_dict['row_key']['feature_name'] = value
        :param xy_features: list of pairs of x- and y-features to explore in
            scatter plot graphs
        :param conditions: a dictionary of facetting conditions to
            filter data_dict entries and show them as separate data rows
            in the histogram; this dictionary must contain as key a
            human-readable description of the condition and as value the
            condition itself; the condition must be a one-parameter function.
            The drawer will call row_conditions['key'](data_dict['row']) and
            only plot the value if this evaluates to True.
        :param colors: a list of colors to shade the different data rows
            supplied by row_conditions in.
        :param no_matter_what: if True, ignore gui_mode and prepare window
            anyway
        :return: None
        """
        if self.gui_mode == "bivariate_analysis" or no_matter_what:
            window = MultiFigureWindow(data_dict)
            for xy_feature in xy_features:
                window.add_scatter_drawer(xy_feature[0], xy_feature[1],
                                          conditions, colors)
            self.window_bivariate_analysis = window