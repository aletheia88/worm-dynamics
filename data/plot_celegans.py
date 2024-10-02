import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt


class Slider(QWidget):

    def __init__(self, name, minimum, maximum, value, parent=None):

        super().__init__(parent)

        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.value = value
        self.label = QLabel()
        self.slider = QSlider(
            Qt.Orientation.Horizontal,
            minimum=self.minimum,
            maximum=self.maximum,
            value=self.value)

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self.update)
        self.update()

    def update(self):
        value = self.slider.value()
        self.label.setText(f'{self.name}: {value}')


class Plot:

    def __init__(self, data):

        self.num_worms, self.num_variables, self.num_frames = data.shape

        print("worms    :", self.num_worms)
        print("variables:", self.num_variables)
        print("frames   :", self.num_frames)

        self.data = data

        self.app = QApplication([])
        self.window = QMainWindow()
        self.central_widget = QWidget()
        self.window.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        plot_widget = pg.PlotWidget()
        self.layout.addWidget(plot_widget)

        self.worm_slider = Slider(
            name="worm",
            minimum=0,
            maximum=self.num_worms - 1,
            value=0)
        self.layout.addWidget(self.worm_slider)

        self.plot = plot_widget.getPlotItem()
        self.plot.setTitle("C. elegans")

        self.worm_slider.slider.valueChanged.connect(self.update_plot)

    def update_plot(self):

        worm = self.worm_slider.slider.value()

        self.plot.clear()
        self.plot.addLegend()

        for variable in range(self.num_variables):
            plot_item = self.plot.plot(
                np.arange(self.num_frames),
                self.data[worm, variable],
                pen=variable)
            self.plot.legend.addItem(plot_item, f"variable {variable}")

    def run(self):

        self.update_plot()
        self.window.show()
        self.app.exec()

if __name__ == "__main__":

    data = np.load("celegans.npy")
    plot = Plot(data)
    plot.run()
