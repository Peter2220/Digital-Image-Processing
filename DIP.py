# Peter Victor Shawky - Section 2 - 17772

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
          
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Image Processing Toolbox'
        self.left = 8
        self.top = 30
        self.width = 1000
        self.height = 1000
        self.init_Main_Ui()
        self.init_Menu_Ui()
        self.current_img = None  # default value at start
        self.second_img = None
        self.history = []

    def init_Main_Ui(self):
        self.setObjectName("Test")
        self.setEnabled(True)
        self.resize(1200, 700)
        self.setMinimumSize(QtCore.QSize(1000, 600))
        self.setMaximumSize(QtCore.QSize(1000, 600))
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.setCentralWidget(self.image_label)
        self.show()

    def init_Menu_Ui(self):
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu('&File')

        Exit_action = QAction('Exit', self)
        Exit_action.setShortcut('Ctrl+E')
        Exit_action.triggered.connect(self.exit)

        Open_action = QAction('Open', self)
        Open_action.setShortcut('Ctrl+O')
        Open_action.triggered.connect(self.read_file)

        file_menu.addAction(Open_action)
        file_menu.addAction(Exit_action)

        self.Edit_menu = menu_bar.addMenu("&Edit")
        self.Edit_menu.setEnabled(False)

        Reset_action = QAction('Reset to Original', self)
        Reset_action.setShortcut('Ctrl+R')
        Reset_action.triggered.connect(self.reset_to_original_image)
        
        Undo_action = QAction('Undo Filter', self)
        Undo_action.setShortcut('Ctrl+Z')
        Undo_action.triggered.connect(self.Undo_filter)  # Filter Must exist
        
        self.Edit_menu.addAction(Reset_action)
        self.Edit_menu.addAction(Undo_action)
        
        self.Brightness_menu = menu_bar.addMenu("&Brightness")
        self.Brightness_menu.setEnabled(False)

        self.Smoothing_menu = menu_bar.addMenu("&Smoothing")
        self.Smoothing_menu.setEnabled(False)

        self.Segmentation_menu = menu_bar.addMenu("&Segmentation")
        self.Segmentation_menu.setEnabled(False)

        self.EdgeDetection_menu = menu_bar.addMenu("&Edge Detection")
        self.EdgeDetection_menu.setEnabled(False)
        
        self.PointProcessing_menu = menu_bar.addMenu("&Point Processing")
        self.PointProcessing_menu.setEnabled(False)
        
        self.compression_menu = menu_bar.addMenu('&Compression')
        self.compression_menu.setEnabled(False)
        
        self.Flipping_menu = menu_bar.addMenu("&Flipping")
        self.Flipping_menu.setEnabled(False)

        Sobel_action = QAction('Sobel', self)
        Sobel_action.setShortcut('Alt+K')
        Sobel_action.triggered.connect(self.Sobel)

        SobelX_action = QAction('Sobel X', self)
        SobelX_action.setShortcut('Alt+X')
        SobelX_action.triggered.connect(self.SobelX)

        SobelY_action = QAction('Sobel Y', self)
        SobelY_action.setShortcut('Alt+Y')
        SobelY_action.triggered.connect(self.SobelY)

        Prewitt_action = QAction('Prewitt', self)
        Prewitt_action.setShortcut('Alt+P')
        Prewitt_action.triggered.connect(self.Prewitt)

        Laplacian_action = QAction('Laplacian', self)
        Laplacian_action.setShortcut('Alt+7')
        Laplacian_action.triggered.connect(self.Laplacian)

        Gaussian_action = QAction('Gaussian Blur', self)
        Gaussian_action.setShortcut('Alt+G')
        Gaussian_action.triggered.connect(self.Gaussian)

        MedianBlur_action = QAction('Median Blur', self)
        MedianBlur_action.setShortcut('Alt+M')
        MedianBlur_action.triggered.connect(self.MedianBlur)

        Avg_3x3_action = QAction('Average Kernel (3x3)', self)
        Avg_3x3_action.setShortcut('Alt+A')
        Avg_3x3_action.triggered.connect(self.Avg_Kernel_3x3)

        Cone_5x5_action = QAction('Cone Kernel (5x5)', self)
        Cone_5x5_action.setShortcut('Alt+C')
        Cone_5x5_action.triggered.connect(self.Cone_Kernel_5x5)

        Circular_5x5_action = QAction('Circular Kernel (5x5)', self)
        Circular_5x5_action.setShortcut('Alt+T')
        Circular_5x5_action.triggered.connect(self.Circular_Kernel_5x5)

        Pyramidal_5x5_action = QAction('Pyramidal Kernel (5x5)', self)
        Pyramidal_5x5_action.setShortcut('Alt+U')
        Pyramidal_5x5_action.triggered.connect(self.Pyramidal_Kernel_5x5)
        
        Negative_action = QAction('Negative', self)
        Negative_action.setShortcut('Alt+1')
        Negative_action.triggered.connect(self.Negative)

        Gamma_action = QAction('Gamma', self)
        Gamma_action.setShortcut('Alt+2')
        Gamma_action.triggered.connect(self.Gamma)

        Log_action = QAction('Log', self)
        Log_action.setShortcut('Alt+3')
        Log_action.triggered.connect(self.Log)

        HistogramEQ_action = QAction('Histogram Equalization', self)
        HistogramEQ_action.setShortcut('Alt+4')
        HistogramEQ_action.triggered.connect(self.HistogramEQ)

        Thresholding_action = QAction('Thresholding', self)
        Thresholding_action.setShortcut('Alt+5')
        Thresholding_action.triggered.connect(self.Thresholding)

        GrayLevelSlicing_action = QAction('Gray Level Slicing', self)
        GrayLevelSlicing_action.setShortcut('Alt+6')
        GrayLevelSlicing_action.triggered.connect(self.GrayLevelSlicing)

        Rotation_action = QAction('Rotation', self)
        Rotation_action.setShortcut('Ctrl+0')
        Rotation_action.triggered.connect(self.Rotation)

        Translation_action = QAction('Translation', self)
        Translation_action.setShortcut('Ctrl+1')
        Translation_action.triggered.connect(self.Translation)

        Skewing_action = QAction('Skewing', self)
        Skewing_action.setShortcut('Ctrl+2')
        Skewing_action.triggered.connect(self.Skewing)

        Scaling_action = QAction('Scaling', self)
        Scaling_action.setShortcut('Ctrl+3')
        Scaling_action.triggered.connect(self.Scaling)

        Blending_action = QAction('Blending', self)
        Blending_action.setShortcut('Ctrl+4')
        Blending_action.triggered.connect(self.Blending)

        FlipX_action = QAction('Flip Around X-axis', self)
        FlipX_action.setShortcut('Ctrl+5')
        FlipX_action.triggered.connect(self.FlipX)

        FlipY_action = QAction('Flip Around Y-axis', self)
        FlipY_action.setShortcut('Ctrl+6')
        FlipY_action.triggered.connect(self.FlipY)

        Flip_XY_action = QAction('Flip Around Both Axes', self)
        Flip_XY_action.setShortcut('Ctrl+7')
        Flip_XY_action.triggered.connect(self.Flip_XY)

        BitPlaneSlicing_action = QAction('Bit Plane Slicing', self)
        BitPlaneSlicing_action.setShortcut('Alt+B')
        BitPlaneSlicing_action.triggered.connect(self.BitPlaneSlicing)
        
        self.setWindowTitle('Image Processing Toolbox')
        
        self.EdgeDetection_menu.addAction(Sobel_action)
        self.EdgeDetection_menu.addAction(SobelX_action)
        self.EdgeDetection_menu.addAction(SobelY_action)
        self.EdgeDetection_menu.addAction(Prewitt_action)
        
        self.Smoothing_menu.addAction(Gaussian_action)
        self.Smoothing_menu.addAction(MedianBlur_action)
        self.Smoothing_menu.addAction(Avg_3x3_action)
        self.Smoothing_menu.addAction(Cone_5x5_action)
        self.Smoothing_menu.addAction(Circular_5x5_action)
        self.Smoothing_menu.addAction(Pyramidal_5x5_action)
        
        self.Brightness_menu.addAction(Negative_action)
        self.Brightness_menu.addAction(Gamma_action)
        self.Brightness_menu.addAction(Log_action)
        self.Brightness_menu.addAction(HistogramEQ_action)

        self.Segmentation_menu.addAction(Thresholding_action)
        self.Segmentation_menu.addAction(GrayLevelSlicing_action)
        self.Segmentation_menu.addAction(Laplacian_action)
        
        self.PointProcessing_menu.addAction(Rotation_action)
        self.PointProcessing_menu.addAction(Translation_action)
        self.PointProcessing_menu.addAction(Skewing_action)
        self.PointProcessing_menu.addAction(Scaling_action)
        self.PointProcessing_menu.addAction(Blending_action)

        self.Flipping_menu.addAction(FlipX_action)
        self.Flipping_menu.addAction(FlipY_action)
        self.Flipping_menu.addAction(Flip_XY_action)

        self.compression_menu.addAction(BitPlaneSlicing_action)

    def read_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        
        if file_name[0] != None:
            img = cv2.imread(file_name[0], 0)

            self.Edit_menu.setEnabled(True)
            self.Segmentation_menu.setEnabled(True)
            self.PointProcessing_menu.setEnabled(True)
            self.Flipping_menu.setEnabled(True)
            self.Brightness_menu.setEnabled(True)
            self.Smoothing_menu.setEnabled(True)
            self.EdgeDetection_menu.setEnabled(True)
            self.compression_menu.setEnabled(True)
            
            self.original_img = img
            self.current_img = self.original_img.copy()
            self.reshow_image(self.current_img)
        
        else:
            QMessageBox.about(self, "Image", "Please choose an image")
            
            
    def read_second_img(self):
        # This function to be called when the user clicks on Blending
        global img2
        file = QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        
        img2 = cv2.imread(file[0], 0)
        
    def reset_to_original_image(self):
        self.history.append(self.current_img.copy())
        self.current_img = self.original_img.copy()
        self.reshow_image(self.current_img)

    def Negative(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        
        new_img = img
        output_img = (255 - new_img)
        
        res = np.hstack((new_img, output_img))
        cv2.imwrite("Negative Image.png", output_img)
        
        #self.reshow_image(res)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Skewing(self, img):
        self.history.append(self.current_img.copy()) 
        img = self.current_img
        new_img = img
        
        rows, cols = new_img.shape
        
        points1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        points2 = np.float32([[0, 0], [cols - 1, 0], [30, rows - 1]])
        M = cv2.getAffineTransform(points1, points2)
        output_img = cv2.warpAffine(new_img, M, (cols,rows))

        res = np.hstack((new_img, output_img))

        cv2.imwrite("Skewed Image.png", output_img)
        #self.reshow_image(res)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Gamma(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        gamma_Val, g = QInputDialog.getDouble(self, "Enter Gamma Value", "Gamma")
        
        gamma_trans = np.power(np.float32(img), gamma_Val)
        cv2.normalize(gamma_trans, gamma_trans, 0, 255, norm_type = cv2.NORM_MINMAX)
        output_img = cv2.convertScaleAbs(gamma_trans, gamma_trans)
        
        res = np.hstack((new_img, output_img))
        
        new_outp = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("Gamma Corrected Image.png", new_outp)
        #self.reshow_image(res)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Log(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        
        log_trans = np.log(np.float32(new_img) + 1)
        cv2.normalize(log_trans, log_trans, 0, 255, norm_type = cv2.NORM_MINMAX)
        output_img = cv2.convertScaleAbs(log_trans, log_trans)
        
        res = np.hstack((new_img, output_img))
        cv2.imwrite("Log Transformed Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Rotation(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        
        rows, cols = new_img.shape

        Angle, R_angle = QInputDialog.getDouble(self, "Enter Rotation Angle", "Angle")
        Scaling_Factor, Sc_factor = QInputDialog.getDouble(self, "Enter Scaling Factor", "Scaling Factor")
        
        M = cv2.getRotationMatrix2D((cols/2, rows/2), Angle, Scaling_Factor)
        output_img = cv2.warpAffine(new_img, M, img.shape)

        res = np.hstack((new_img, output_img))
        
        cv2.imwrite("Rotated Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Scaling(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        
        x, x_start = QInputDialog.getInt(self, "Starting X", "X:")
        y, y_start = QInputDialog.getInt(self, "Starting Y", "Y:")
        
        f, Height = QInputDialog.getInt(self, "Enter Height", "Height:")
        w, Width = QInputDialog.getInt(self, "Enter Width", "Width:")

        crop_img = new_img[x : w, y : h]
        
        ScaleX, F_x = QInputDialog.getInt(self, "Enter Fx", "Fx:")
        ScaleY, F_y = QInputDialog.getInt(self, "Enter Fy", "Fy:")
        
        interp_dict = {'0': "cv2.INTER_NEAREST", '1': "cv2.INTER_LINEAR", '2': "INTER_CUBIC"}

        QMessageBox.about(self, "Interpolation", "Enter: 0 for Nearest-Neighbour, 1 for Bi-Linear, 2 for Cubic")

        interpolation, val = QInputDialog.getInt(self, "Enter Interpolation", "Interpolation: ")
        
        output_img = cv2.resize(crop_img, None, fx = ScaleX, fy = ScaleY,
                                interpolation = print(interp_dict[str(interpolation)]))

        cv2.imwrite("Scaled Image.png", output_img)
        scaled_img = output_img
        self.current_img = scaled_img
        cv2.imshow("Scaled Image", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def Translation(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        
        rows, cols = new_img.shape

        Tx, t_x = QInputDialog.getDouble(self, "Enter Tx", "Tx")
        Ty, t_y = QInputDialog.getDouble(self, "Enter Ty", "Ty")
        
        M = np.float32([[1, 0, Tx], [0, 1, Ty]])

        output_img = cv2.warpAffine(new_img, M,(rows, cols))

        in_out = np.hstack((new_img, output_img))

        cv2.imwrite("Translated Image.png", output_img)
        cv2.imshow("Input vs Output", in_out)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def Blending(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        img1 = img

        QMessageBox.about(self, "Second Image", "Please open another image")
        self.read_second_img()
        
        rows, cols = img2.shape
        img1 = cv2.resize(img1, (rows, cols))

        Img1_percentage, percentage1 = QInputDialog.getDouble(self, "First Image %", "Percentage")
        Img2_percentage, percentage2 = QInputDialog.getDouble(self, "Second Image %", "Percentage")
        
        output_img = cv2.addWeighted(img1, Img1_percentage, img2, Img2_percentage, 0)
        
        cv2.imwrite("Blended Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def FlipX(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        output_img = cv2.flip(new_img, 0)

        cv2.imwrite("Vertically Flipped Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def FlipY(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        
        output_img = cv2.flip(new_img, 1)

        cv2.imwrite("Horizontally Flipped Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Flip_XY(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        output_img = cv2.flip(new_img, -1)
        
        cv2.imwrite("Flipped X-Y Axes Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)
              
    def HistogramEQ(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        minbef = np.min(new_img)
        maxbef = np.max(new_img)
        QMessageBox.about(self, "Before", "Minimum= " + str(minbef) + ", and Max = " + str(maxbef))

        # Histograms Equalization using cv2.equalizeHist()
        output_img = cv2.equalizeHist(new_img)
        res = np.hstack((new_img, output_img))

        cv2.imwrite("Equalized Image.png", output_img)

        cv2.imshow("Before vs After", res)
        cv2.waitKey(0)

        # Histogram Calculation using Numpy
        hist, bins = np.histogram(new_img.ravel(), 256, [0, 256])

        minimum = np.min(output_img)
        maximum = np.max(output_img)
        QMessageBox.about(self, "After", "Minimum= " + str(minimum) + ", and Max = " + str(maximum))

        # Histogram Calculation using Opencv
        histr = cv2.calcHist([new_img], [0], None, [256], [0, 256])

        plt.subplot(1, 2, 1)
        plt.plot(histr)

        plt.subplot(1, 2, 2)
        plt.hist(output_img.ravel(), 256, [0, 256])
        plt.show()

        self.current_img = output_img
        self.reshow_image(self.current_img)

    def Thresholding(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        Threshold, thresh = QInputDialog.getInt(self, "Enter Threshold", "Threshold: ")
        
        res, output_img = cv2.threshold(img, Threshold, 255, cv2.THRESH_BINARY)

        cv2.imwrite("Thresholded Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def Gaussian(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        kernel, k = QInputDialog.getInt(self, "Kernel", "Enter odd number")
        
        output_img = cv2.GaussianBlur(new_img,(kernel,kernel), 0)

        cv2.imwrite("Gaussian Blurred Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)


    def MedianBlur(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        kern, ker = QInputDialog.getInt(self, "Kernel", "Enter odd number")
        
        output_img = cv2.medianBlur(img, kern)

        cv2.imwrite("Median Blurred Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def Avg_Kernel_3x3(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        Avg_3x3 = (1/9) * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
        
        output_img = cv2.filter2D(new_img, -1, Avg_3x3)

        cv2.imwrite("Average Kernel 3x3 Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Cone_Kernel_5x5(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        Cone_5x5 = (1/25) * np.array([[0, 0, 1, 0, 0],
                                      [0, 2, 2, 2, 0],
                                      [1, 2, 5, 2, 1],
                                      [0, 2, 2, 2, 0],
                                      [0, 0, 1, 0, 0]])
        
        output_img = cv2.filter2D(new_img, -1, Cone_5x5)

        cv2.imwrite("Cone Kernel 5x5 Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Circular_Kernel_5x5(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        Circular_5x5 = (1/21) * np.array([[0, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0, 1, 1, 1, 0]])
        
        output_img = cv2.filter2D(img, -1, Circular_5x5)

        cv2.imwrite("Circular Kernel 5x5 Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Pyramidal_Kernel_5x5(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        Pyramidal_5x5 = (1/81)* np.array([[1, 2, 3, 2, 1],
                                          [2, 4, 6, 4, 2],
                                          [3, 6, 9, 6, 3],
                                          [2, 4, 6, 4, 2],
                                          [1, 2, 3, 2, 1]])
        
        output_img = cv2.filter2D(img, -1, Pyramidal_5x5)

        cv2.imwrite("Pyramidal Kernel 5x5 Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def GrayLevelSlicing(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img

        row, column = new_img.shape

        # Create an zeros array to store the sliced image
        output_img = np.zeros((row, column), dtype = 'uint8')

        Tmin, min_range = QInputDialog.getInt(self, "Minimum Threshold", "Enter Tmin: ")
        Tmax, max_range = QInputDialog.getInt(self, "Maximum Threshold", "Enter Tmax: ")
        
        for i in range(row):
            for j in range(column):
                if new_img[i,j] > Tmin and new_img[i,j] < Tmax:
                    output_img[i,j] = 255
                else:
                    output_img[i,j] = 0
            
        cv2.imwrite("Gray Level Sliced Image.png", output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)


    def Sobel(self, img):
        """IMPORTANT NOTE: if the output datatype is cv2.CV_8U or np.uint8, this will make all negative values 0.
        To prevent this, we specify the output datatype to some higher forms,
        like cv2.CV_16S, cv2.CV_64F etc, take its absolute value and then convert back to cv2.CV_8U. """

        self.history.append(self.current_img.copy())

        img = self.current_img

        new_img = img.copy()

        sobelx_64 = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely_64 = cv2.Sobel(new_img, cv2.CV_64F, 0, 1, ksize=3)

        sobel = new_img.copy()
        height = np.size(new_img, 0)
        width = np.size(new_img, 1)

        for i in range(width):
            for j in range(height):
                sobel[j, i] = np.minimum(255,
                                         np.round(np.sqrt(sobelx_64[j, i] * sobelx_64[j, i] + sobely_64[j, i] * sobely_64[j, i])))

        output_img = sobel
        cv2.imwrite("Sobel Filtered Image.png", output_img)

        self.current_img = output_img
        self.reshow_image(self.current_img)

    def SobelX(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        sobel_X_64 = cv2.Sobel(new_img, cv2.CV_64F, 1, 0, ksize = 3)
        abs_sobel_X64f = np.absolute(sobel_X_64)
        sobelX_8u = np.uint8(abs_sobel_X64f)

        output_img = sobelX_8u
        cv2.imwrite("Sobel X Filtered Image.png", output_img)

        self.current_img = output_img
        self.reshow_image(self.current_img)

    def SobelY(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img.copy()

        sobel_Y_64 = cv2.Sobel(new_img, cv2.CV_64F, 0, 1, ksize = 3)
        abs_sobel_Y64f = np.absolute(sobel_Y_64)
        sobelY_8u = np.uint8(abs_sobel_Y64f)

        output_img = sobelY_8u
        cv2.imwrite("Sobel Y Filtered Image.png", output_img)

        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Prewitt(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img

        new_img = img.copy()

        kernelx = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])

        img_prewittx = cv2.filter2D(new_img, -1, kernelx)
        img_prewitty = cv2.filter2D(new_img, -1, kernely)

        output_img = img_prewittx + img_prewitty
        
        cv2.imwrite("Prewitt filtered Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def Laplacian(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        
        new_img = img.copy()
        
        laplacian = cv2.Laplacian(new_img, cv2.CV_64F)
        output_img = np.uint8(np.absolute(laplacian))

        cv2.imwrite("Laplacian filtered Image.png", output_img)
        
        self.current_img = output_img
        self.reshow_image(self.current_img)

    def BitPlaneSlicing(self, img):
        self.history.append(self.current_img.copy())
        img = self.current_img
        new_img = img
        out = []
        
        plan, p = QInputDialog.getInt(self, "Bit Plane", "Enter Bit Plane (0 to 6): ")
        for k in range(0, 7):
            # Create an image for each k bit plane
            plane = np.full((new_img.shape[0], new_img.shape[1]), 2 ** k, np.uint8)
            # Execute bitwise and operation
            res = cv2.bitwise_and(plane, new_img)
            # Multiply ones (bit plane sliced) with 255 just for better visualization
            x = res * 255
            # Append to the output list
            out.append(x)
            #cv2.imwrite('./Bitplane' + str(7-k) + '.png', out[x])
        
        output_img = out[plan].copy()
        cv2.imwrite("Bit Plane Sliced Image - " + "Plane " + str(plan) + ".png" , output_img)
        self.current_img = output_img
        self.reshow_image(self.current_img)
        
    def Undo_filter(self):
        if self.history:
            self.current_img = self.history.pop(-1)
            self.reshow_image(self.current_img)
            
    def reshow_image(self, cv_img):
        if cv_img is not None:
            self.image_label.resize(cv_img.shape[1], cv_img.shape[0])
            Q_img = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], QImage.Format_Grayscale8)
            self.image_label.setPixmap(QPixmap.fromImage(Q_img))   
        else:
            print("Image load failed")

                
    def exit(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    sys.exit(app.exec_())
