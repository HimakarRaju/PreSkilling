import sys
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EDAApp(QMainWindow):
    def __init__(self):
        super(EDAApp, self).__init__()

        # Load the UI from a .ui file created in Qt Designer
        uic.loadUi("eda_app.ui", self)

        self.dataset = {}
        self.current_dataset_name = None

        # Connect buttons and actions to functions
        self.loadDatasetButton.clicked.connect(self.load_multiple_datasets)
        self.showHeadButton.clicked.connect(self.show_data)
        self.showStatsButton.clicked.connect(self.show_statistics)
        self.corrHeatmapButton.clicked.connect(self.generate_corr_heatmap)
        self.generatePlotButton.clicked.connect(self.generate_plot)

    def load_multiple_datasets(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_paths:
            self.statusBar().showMessage("Loading datasets...")
            for path in file_paths:
                try:
                    dataset_name = path.split("/")[-1]
                    self.dataset[dataset_name] = pd.read_csv(path)
                    self.current_dataset_name = dataset_name
                    self.datasetComboBox.addItem(dataset_name)
                    self.update_columns()
                    self.statusBar().showMessage(f"Loaded {dataset_name}")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error", f"Failed to load {dataset_name}: {str(e)}"
                    )
                    self.statusBar().showMessage(f"Failed to load {dataset_name}")

    def update_columns(self):
        if self.current_dataset_name:
            data = self.dataset[self.current_dataset_name]
            columns = list(data.columns)
            self.xComboBox.addItems(columns)
            self.yComboBox.addItems(columns)

    def show_data(self):
        if self.dataset:
            self.textEdit.clear()
            for name, data in self.dataset.items():
                self.textEdit.append(f"{name}:\n{data.head().to_string()}\n\n")
        else:
            QMessageBox.warning(self, "Error", "No dataset loaded!")

    def show_statistics(self):
        if self.dataset:
            self.textEdit.clear()
            for name, data in self.dataset.items():
                self.textEdit.append(
                    f"{name} Summary:\n{data.describe().to_string()}\n\n"
                )
        else:
            QMessageBox.warning(self, "Error", "No dataset loaded!")

    def generate_corr_heatmap(self):
        if self.dataset:
            data = self.dataset[self.current_dataset_name]
            numerical_cols = data.select_dtypes(include="number")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numerical_cols.corr(), annot=True, fmt=".2f", ax=ax)
            canvas = FigureCanvas(fig)

            # Create new window to display the heatmap
            self.create_plot_window(canvas)
            self.statusBar().showMessage(
                f"Generated heatmap for {self.current_dataset_name}"
            )
        else:
            QMessageBox.warning(self, "Error", "No dataset loaded!")

    def generate_plot(self):
        plot_type = self.plotTypeComboBox.currentText()
        x_col = self.xComboBox.currentText()
        y_col = self.yComboBox.currentText()

        if plot_type and x_col and y_col:
            data = self.dataset[self.current_dataset_name]
            fig, ax = plt.subplots()

            if plot_type == "PCA":
                self.generate_pca_plot()
            elif plot_type == "scatterplot":
                sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax)
            elif plot_type == "histogram":
                sns.histplot(data[x_col], ax=ax)
            elif plot_type == "pairplot":
                sns.pairplot(data)

            canvas = FigureCanvas(fig)
            self.create_plot_window(canvas)
            self.statusBar().showMessage(
                f"Generated {plot_type} plot for {self.current_dataset_name}"
            )
        else:
            QMessageBox.warning(
                self, "Error", "Please select a valid plot type and columns!"
            )

    def generate_pca_plot(self):
        data = self.dataset[self.current_dataset_name].select_dtypes(
            include=[np.number]
        )
        pca = PCA(n_components=2)
        scaled_data = StandardScaler().fit_transform(data)
        principal_components = pca.fit_transform(scaled_data)

        df_pca = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
        fig = px.scatter(df_pca, x="PC1", y="PC2")
        fig.show()

    def create_plot_window(self, canvas):
        plot_window = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        plot_window.setLayout(layout)
        plot_window.setWindowTitle("Plot Window")
        plot_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EDAApp()
    window.show()
    sys.exit(app.exec())
