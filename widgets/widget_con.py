from PyQt5 import QtWidgets, QtCore


class ConWidget(QtWidgets.QWidget):
    
    Signal_piezo_move = QtCore.pyqtSignal()
    Signal_deck_up = QtCore.pyqtSignal()
    Signal_deck_down = QtCore.pyqtSignal()
    Signal_deck_move = QtCore.pyqtSignal()
    Signal_deck_move_stop = QtCore.pyqtSignal()
    Signal_setcoordinates = QtCore.pyqtSignal()
    Signal_resetcoordinates = QtCore.pyqtSignal()
    Signal_setbin = QtCore.pyqtSignal()
    Signal_setlaseron_488_0 = QtCore.pyqtSignal()
    Signal_setlaseron_488_1 = QtCore.pyqtSignal()
    Signal_setlaseron_488_2 = QtCore.pyqtSignal()
    Signal_setlaseron_405 = QtCore.pyqtSignal()
    Signal_setlaseroff_488_0 = QtCore.pyqtSignal()
    Signal_setlaseroff_488_1 = QtCore.pyqtSignal()
    Signal_setlaseroff_488_2 = QtCore.pyqtSignal()
    Signal_setlaseroff_405 = QtCore.pyqtSignal()
    Signal_start_video = QtCore.pyqtSignal()
    Signal_stop_video = QtCore.pyqtSignal()
    Signal_run_fft = QtCore.pyqtSignal()
    Signal_stop_fft = QtCore.pyqtSignal()
    Signal_2d_resolft = QtCore.pyqtSignal()
    Signal_3d_resolft = QtCore.pyqtSignal()
    Signal_beadscan_2d = QtCore.pyqtSignal()
    Signal_save_file = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        Group_CCDCamera = QtWidgets.QGroupBox('CCD Camera')
        Group_CCDCamera.setStyleSheet("font: bold Arial 10px")
        Group_PiezoStage = QtWidgets.QGroupBox('Piezo Stage')
        Group_PiezoStage.setStyleSheet("font: bold Arial 10px")
        Group_MadDeck = QtWidgets.QGroupBox('Mad Deck')
        Group_MadDeck.setStyleSheet("font: bold Arial 10px")
        Group_Illumination = QtWidgets.QGroupBox('Laser')
        Group_Illumination.setStyleSheet("font: bold Arial 10px")
        Group_DataAquisition = QtWidgets.QGroupBox('Image Acquisition ')
        Group_DataAquisition.setStyleSheet("font: bold Arial 10px")
        Group_Triggers = QtWidgets.QGroupBox('Triggers')
        Group_Triggers.setStyleSheet("font: bold Arial 10px")
        Group_File = QtWidgets.QGroupBox('File')
        Group_File.setStyleSheet("font: bold Arial 10px")
        layout.addStretch()
        layout.addWidget(Group_CCDCamera)
        layout.addWidget(Group_PiezoStage)
        layout.addWidget(Group_MadDeck)
        layout.addWidget(Group_Illumination)
        layout.addWidget(Group_DataAquisition)
        layout.addWidget(Group_Triggers)
        layout.addWidget(Group_File)
        layout.addStretch()
        self.setLayout(layout)

        Layout_MainCamera = QtWidgets.QGridLayout()
        
        self.QLabel_ccd_tempetature = QtWidgets.QLabel(str('CCD Temperature'))
        self.QLabel_ccd_tempetature.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLabel_ccd_tempetature.resize(10, 20)
        
        self.QLCDNumber_ccd_tempetature = QtWidgets.QLCDNumber()
        self.QLCDNumber_ccd_tempetature.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLCDNumber_ccd_tempetature.setDecMode()
 
        self.QLabel_coordinate_x = QtWidgets.QLabel(str('X'))
        self.QLabel_coordinate_x.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QSpinBox_coordinate_x = QtWidgets.QSpinBox()
        self.QSpinBox_coordinate_x.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QSpinBox_coordinate_x.setRange(0, 2048)
        self.QSpinBox_coordinate_x.setSingleStep(1)
        self.QSpinBox_coordinate_x.setValue(1)
        
        self.QLabel_coordinate_y = QtWidgets.QLabel(str('Y'))
        self.QLabel_coordinate_y.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QSpinBox_coordinate_y = QtWidgets.QSpinBox()
        self.QSpinBox_coordinate_y.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QSpinBox_coordinate_y.setRange(0, 2048)
        self.QSpinBox_coordinate_y.setSingleStep(1)
        self.QSpinBox_coordinate_y.setValue(1)
        
        self.QLabel_coordinate_n = QtWidgets.QLabel(str('N'))
        self.QLabel_coordinate_n.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QSpinBox_coordinate_n = QtWidgets.QSpinBox()
        self.QSpinBox_coordinate_n.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QSpinBox_coordinate_n.setRange(0, 2048)
        self.QSpinBox_coordinate_n.setSingleStep(1)
        self.QSpinBox_coordinate_n.setValue(1024)
        
        self.QLabel_coordinate_bin = QtWidgets.QLabel(str('Bin'))
        self.QLabel_coordinate_bin.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QSpinBox_coordinate_bin = QtWidgets.QSpinBox()
        self.QSpinBox_coordinate_bin.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QSpinBox_coordinate_bin.setRange(0, 1024)
        self.QSpinBox_coordinate_bin.setSingleStep(1)
        self.QSpinBox_coordinate_bin.setValue(1)
        
        self.QPushButton_setcoordinates = QtWidgets.QPushButton('SetCoord', self)
        self.QPushButton_setcoordinates.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_resetcoordinates = QtWidgets.QPushButton('ResetCoord', self)
        self.QPushButton_resetcoordinates.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        Layout_MainCamera.addWidget(self.QLabel_ccd_tempetature, 0, 0, 1, 1)
        Layout_MainCamera.addWidget(self.QLCDNumber_ccd_tempetature, 0, 1, 1, 1)
        Layout_MainCamera.addWidget(self.QLabel_coordinate_x, 2, 0, 1, 1)
        Layout_MainCamera.addWidget(self.QSpinBox_coordinate_x, 3, 0, 1, 1)
        Layout_MainCamera.addWidget(self.QLabel_coordinate_y, 2, 1, 1, 1)
        Layout_MainCamera.addWidget(self.QSpinBox_coordinate_y, 3, 1, 1, 1)
        Layout_MainCamera.addWidget(self.QLabel_coordinate_n, 2, 2, 1, 1)
        Layout_MainCamera.addWidget(self.QSpinBox_coordinate_n, 3, 2, 1, 1)
        Layout_MainCamera.addWidget(self.QLabel_coordinate_bin, 2, 3, 1, 1)
        Layout_MainCamera.addWidget(self.QSpinBox_coordinate_bin, 3, 3, 1, 1)
        Layout_MainCamera.addWidget(self.QPushButton_setcoordinates, 4, 0, 1, 1)
        Layout_MainCamera.addWidget(self.QPushButton_resetcoordinates, 4, 1, 1, 1)
        Group_CCDCamera.setLayout(Layout_MainCamera)
        
        Layout_mad_deck = QtWidgets.QGridLayout()
        
        self.QLabel_deck_position = QtWidgets.QLabel(str('Position (mm)'))
        self.QLabel_deck_position.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLCDNumber_deck_position = QtWidgets.QLCDNumber()
        self.QLCDNumber_deck_position.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLCDNumber_deck_position.setDecMode()
        self.QLCDNumber_deck_position.setDigitCount(10)
        
        self.QLabel_deck_move_fine = QtWidgets.QLabel(str('Fine step'))
        self.QLabel_deck_move_fine.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QPushButton_move_deck_up = QtWidgets.QPushButton('Up', self)
        self.QPushButton_move_deck_up.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_move_deck_down = QtWidgets.QPushButton('Down', self)
        self.QPushButton_move_deck_down.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QLabel_deck_movement = QtWidgets.QLabel(str('Distance (mm)'))
        self.QLabel_deck_movement.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_deck_movement = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_deck_movement.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_deck_movement.setRange(-11.5, 11.5)
        self.QDoubleSpinBox_deck_movement.setSingleStep(0.001)
        self.QDoubleSpinBox_deck_movement.setDecimals(3)
        self.QDoubleSpinBox_deck_movement.setValue(0.01)
        
        self.QLabel_deck_velocity = QtWidgets.QLabel(str('Velocity (mm)'))
        self.QLabel_deck_velocity.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_deck_velocity = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_deck_velocity.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_deck_velocity.setRange(0.02, 1.50)
        self.QDoubleSpinBox_deck_velocity.setSingleStep(0.02)
        self.QDoubleSpinBox_deck_velocity.setDecimals(2)
        self.QDoubleSpinBox_deck_velocity.setValue(0.02)
        
        self.QPushButton_move_deck = QtWidgets.QPushButton('Move', self)
        self.QPushButton_move_deck.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_move_deck_stop = QtWidgets.QPushButton('STOP', self)
        self.QPushButton_move_deck_stop.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        Layout_mad_deck.addWidget(self.QLabel_deck_position, 0, 0, 1, 1)
        Layout_mad_deck.addWidget(self.QLCDNumber_deck_position, 1, 0, 1, 1)
        Layout_mad_deck.addWidget(self.QLabel_deck_move_fine, 0, 2, 1, 1)
        Layout_mad_deck.addWidget(self.QPushButton_move_deck_up, 1, 2, 1, 1)
        Layout_mad_deck.addWidget(self.QPushButton_move_deck_down, 2, 2, 1, 1)
        Layout_mad_deck.addWidget(self.QLabel_deck_movement, 0, 1, 1, 1)
        Layout_mad_deck.addWidget(self.QDoubleSpinBox_deck_movement, 1, 1, 1, 1)
        # Layout_mad_deck.addWidget(self.QLabel_deck_velocity, 0, 1, 1, 1)
        # Layout_mad_deck.addWidget(self.QDoubleSpinBox_deck_velocity, 1, 1, 1, 1)
        Layout_mad_deck.addWidget(self.QPushButton_move_deck, 2, 0, 1, 1)
        Layout_mad_deck.addWidget(self.QPushButton_move_deck_stop, 2, 1, 1, 1)
        
        Group_MadDeck.setLayout(Layout_mad_deck)
        
        Layout_piezo_stage = QtWidgets.QGridLayout()
        
        self.QLabel_stage_x = QtWidgets.QLabel(str('X (um)'))
        self.QLabel_stage_x.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_stage_x = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_stage_x.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_stage_x.setRange(0, 50)
        self.QDoubleSpinBox_stage_x.setSingleStep(0.001)
        self.QDoubleSpinBox_stage_x.setDecimals(3)
        self.QDoubleSpinBox_stage_x.setValue(0.00)
        
        self.QLCDNumber_stage_x = QtWidgets.QLCDNumber()
        self.QLCDNumber_stage_x.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLCDNumber_stage_x.setDecMode()
        
        self.QLabel_stage_y = QtWidgets.QLabel(str('Y (um)'))
        self.QLabel_stage_y.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_stage_y = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_stage_y.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_stage_y.setRange(0, 50)
        self.QDoubleSpinBox_stage_y.setSingleStep(0.001)
        self.QDoubleSpinBox_stage_y.setDecimals(3)
        self.QDoubleSpinBox_stage_y.setValue(0.00)
        
        self.QLCDNumber_stage_y = QtWidgets.QLCDNumber()
        self.QLCDNumber_stage_y.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLCDNumber_stage_y.setDecMode()
        
        self.QLabel_stage_z = QtWidgets.QLabel(str('Z (um)'))
        self.QLabel_stage_z.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_stage_z = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_stage_z.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_stage_z.setRange(0, 50)
        self.QDoubleSpinBox_stage_z.setSingleStep(0.001)
        self.QDoubleSpinBox_stage_z.setDecimals(3)
        self.QDoubleSpinBox_stage_z.setValue(0.00)
        
        self.QLCDNumber_stage_z = QtWidgets.QLCDNumber()
        self.QLCDNumber_stage_z.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QLCDNumber_stage_z.setDecMode()
        
        Layout_piezo_stage.addWidget(self.QLabel_stage_x, 0, 0, 1, 1)
        Layout_piezo_stage.addWidget(self.QDoubleSpinBox_stage_x, 1, 0, 1, 1)
        Layout_piezo_stage.addWidget(self.QLCDNumber_stage_x, 2, 0, 1, 1)
        Layout_piezo_stage.addWidget(self.QLabel_stage_y, 0, 1, 1, 1)
        Layout_piezo_stage.addWidget(self.QDoubleSpinBox_stage_y, 1, 1, 1, 1)
        Layout_piezo_stage.addWidget(self.QLCDNumber_stage_y, 2, 1, 1, 1)
        Layout_piezo_stage.addWidget(self.QLabel_stage_z, 0, 2, 1, 1)
        Layout_piezo_stage.addWidget(self.QDoubleSpinBox_stage_z, 1, 2, 1, 1)
        Layout_piezo_stage.addWidget(self.QLCDNumber_stage_z, 2, 2, 1, 1)
        
        Group_PiezoStage.setLayout(Layout_piezo_stage)
        
        Layout_Illumination = QtWidgets.QGridLayout()
        
        # self.QLabel_laser_488_0 = QtWidgets.QLabel(str('Laser 488 nm #0'))
        # self.QLabel_laser_488_0.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QRadioButton_laser_488_0 = QtWidgets.QRadioButton('Laser 488 nm #0', self)
        self.QRadioButton_laser_488_0.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_laserpower_488_0 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_laserpower_488_0.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_laserpower_488_0.setRange(0, 100)
        self.QDoubleSpinBox_laserpower_488_0.setSingleStep(0.1)
        self.QDoubleSpinBox_laserpower_488_0.setDecimals(1)
        self.QDoubleSpinBox_laserpower_488_0.setValue(0.0)
        
        self.QPushButton_laser_488_0 = QtWidgets.QPushButton('ON', self)
        self.QPushButton_laser_488_0.setCheckable(True)
        self.QPushButton_laser_488_0.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        # self.QLabel_laser_488_1 = QtWidgets.QLabel(str('Laser 488 nm #1'))
        # self.QLabel_laser_488_1.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QRadioButton_laser_488_1 = QtWidgets.QRadioButton('Laser 488 nm #1', self)
        self.QRadioButton_laser_488_1.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_laserpower_488_1 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_laserpower_488_1.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_laserpower_488_1.setRange(0, 100)
        self.QDoubleSpinBox_laserpower_488_1.setSingleStep(0.1)
        self.QDoubleSpinBox_laserpower_488_1.setDecimals(1)
        self.QDoubleSpinBox_laserpower_488_1.setValue(0.0)
        
        self.QPushButton_laser_488_1 = QtWidgets.QPushButton('ON', self)
        self.QPushButton_laser_488_1.setCheckable(True)
        self.QPushButton_laser_488_1.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        # self.QLabel_laser_488_2 = QtWidgets.QLabel(str('Laser 488 nm #2'))
        # self.QLabel_laser_488_2.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QRadioButton_laser_488_2 = QtWidgets.QRadioButton('Laser 488 nm #2', self)
        self.QRadioButton_laser_488_2.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_laserpower_488_2 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_laserpower_488_2.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_laserpower_488_2.setRange(0, 100)
        self.QDoubleSpinBox_laserpower_488_2.setSingleStep(0.1)
        self.QDoubleSpinBox_laserpower_488_2.setDecimals(1)
        self.QDoubleSpinBox_laserpower_488_2.setValue(0.0)
        
        self.QPushButton_laser_488_2 = QtWidgets.QPushButton('ON', self)
        self.QPushButton_laser_488_2.setCheckable(True)
        self.QPushButton_laser_488_2.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        # self.QLabel_laser_405 = QtWidgets.QLabel(str('Laser 405 nm'))
        # self.QLabel_laser_405.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        self.QRadioButton_laser_405 = QtWidgets.QRadioButton('Laser 405 nm', self)
        self.QRadioButton_laser_405.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
                
        self.QDoubleSpinBox_laserpower_405 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_laserpower_405.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_laserpower_405.setRange(0, 100)
        self.QDoubleSpinBox_laserpower_405.setSingleStep(0.1)
        self.QDoubleSpinBox_laserpower_405.setDecimals(1)
        self.QDoubleSpinBox_laserpower_405.setValue(0.0)
        
        self.QPushButton_laser_405 = QtWidgets.QPushButton('ON', self)
        self.QPushButton_laser_405.setCheckable(True)
        self.QPushButton_laser_405.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        # Layout_Illumination.addWidget(self.QLabel_laser_488_0, 0, 0)
        Layout_Illumination.addWidget(self.QRadioButton_laser_488_0, 0, 0, 1, 1)
        Layout_Illumination.addWidget(self.QDoubleSpinBox_laserpower_488_0, 0, 1, 1, 1)
        Layout_Illumination.addWidget(self.QPushButton_laser_488_0, 0, 2, 1, 1)
        # Layout_Illumination.addWidget(self.QLabel_laser_488_1, 1, 0)
        Layout_Illumination.addWidget(self.QRadioButton_laser_488_1, 1, 0, 1, 1)
        Layout_Illumination.addWidget(self.QDoubleSpinBox_laserpower_488_1, 1, 1, 1, 1)
        Layout_Illumination.addWidget(self.QPushButton_laser_488_1, 1, 2, 1, 1)
        # Layout_Illumination.addWidget(self.QLabel_laser_488, 2, 0)
        Layout_Illumination.addWidget(self.QRadioButton_laser_488_2, 2, 0, 1, 1)
        Layout_Illumination.addWidget(self.QDoubleSpinBox_laserpower_488_2, 2, 1, 1, 1)
        Layout_Illumination.addWidget(self.QPushButton_laser_488_2, 2, 2, 1, 1)
        # Layout_Illumination.addWidget(self.QLabel_laser_405, 3, 0)
        Layout_Illumination.addWidget(self.QRadioButton_laser_405, 3, 0, 1, 1)
        Layout_Illumination.addWidget(self.QDoubleSpinBox_laserpower_405, 3, 1, 1, 1)
        Layout_Illumination.addWidget(self.QPushButton_laser_405, 3, 2, 1, 1)
        
        Group_Illumination.setLayout(Layout_Illumination)
        
        Layout_DataAquisition = QtWidgets.QGridLayout()
        
        self.QLabel_exposure_time = QtWidgets.QLabel(str('Exposure Time (s)'))
        self.QLabel_exposure_time.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_exposure_time = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_exposure_time.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_exposure_time.setRange(0, 10)
        self.QDoubleSpinBox_exposure_time.setSingleStep(0.005)
        self.QDoubleSpinBox_exposure_time.setDecimals(3)
        self.QDoubleSpinBox_exposure_time.setValue(0.02)
        
        self.QLabel_emccd_gain = QtWidgets.QLabel(str('EMCCD Gain'))
        self.QLabel_emccd_gain.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
               
        self.QSpinBox_emccd_gain = QtWidgets.QSpinBox()
        self.QSpinBox_emccd_gain.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QSpinBox_emccd_gain.setRange(0, 300)
        self.QSpinBox_emccd_gain.setSingleStep(1)
        self.QSpinBox_emccd_gain.setValue(0)
        
        self.QPushButton_start_video = QtWidgets.QPushButton('Video', self)
        self.QPushButton_start_video.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_stop_video = QtWidgets.QPushButton('Stop', self)
        self.QPushButton_stop_video.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_singlesnap = QtWidgets.QPushButton('SingleSnap', self)
        self.QPushButton_singlesnap.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_start_fft = QtWidgets.QPushButton('Run FFT', self)
        self.QPushButton_start_fft.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_stop_fft = QtWidgets.QPushButton('Stop FFT', self)
        self.QPushButton_stop_fft.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
    
        Layout_DataAquisition.addWidget(self.QLabel_exposure_time, 0, 0, 1, 1)
        Layout_DataAquisition.addWidget(self.QDoubleSpinBox_exposure_time, 0, 1, 1, 1)
        Layout_DataAquisition.addWidget(self.QLabel_emccd_gain, 1, 0, 1, 1)
        Layout_DataAquisition.addWidget(self.QSpinBox_emccd_gain, 1, 1, 1, 1)
        Layout_DataAquisition.addWidget(self.QPushButton_start_video, 2, 0, 1, 1)
        Layout_DataAquisition.addWidget(self.QPushButton_stop_video, 2, 1, 1, 1)
        Layout_DataAquisition.addWidget(self.QPushButton_singlesnap, 2, 2, 1, 1)
        Layout_DataAquisition.addWidget(self.QPushButton_start_fft, 3, 0, 1, 1)
        Layout_DataAquisition.addWidget(self.QPushButton_stop_fft, 3, 1, 1, 1)
        
        Group_DataAquisition.setLayout(Layout_DataAquisition)

        Layout_Triggers = QtWidgets.QGridLayout()
        
        self.QLabel_cycle_period = QtWidgets.QLabel(str('Cycle period / s'))
        self.QLabel_cycle_period.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_cycle_period = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_cycle_period.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_cycle_period.setRange(0, 1)
        self.QDoubleSpinBox_cycle_period.setSingleStep(0.001)
        self.QDoubleSpinBox_cycle_period.setDecimals(3)
        self.QDoubleSpinBox_cycle_period.setValue(0.100)
        
        self.QLabel_piezo_start = QtWidgets.QLabel(str('Piezo_start / s'))
        self.QLabel_piezo_start.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_piezo_start = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_piezo_start.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_piezo_start.setRange(0, 1)
        self.QDoubleSpinBox_piezo_start.setSingleStep(0.001)
        self.QDoubleSpinBox_piezo_start.setDecimals(3)
        self.QDoubleSpinBox_piezo_start.setValue(0.075)
        
        self.QLabel_start_positions = QtWidgets.QLabel(str('Start_positions / um'))
        self.QLabel_start_positions.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_step_sizes = QtWidgets.QLabel(str('Step_sizes / um'))
        self.QLabel_step_sizes.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_scan_ranges = QtWidgets.QLabel(str('Scan_ranges / um'))
        self.QLabel_scan_ranges.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_axis_x = QtWidgets.QLabel(str('X'))
        self.QLabel_axis_x.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_axis_y = QtWidgets.QLabel(str('Y'))
        self.QLabel_axis_y.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_axis_z = QtWidgets.QLabel(str('Z'))
        self.QLabel_axis_z.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_start_x = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_start_x.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_start_x.setRange(0, 50)
        self.QDoubleSpinBox_start_x.setSingleStep(0.001)
        self.QDoubleSpinBox_start_x.setDecimals(3)
        self.QDoubleSpinBox_start_x.setValue(0.000)
        
        self.QDoubleSpinBox_start_y = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_start_y.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_start_y.setRange(0, 50)
        self.QDoubleSpinBox_start_y.setSingleStep(0.001)
        self.QDoubleSpinBox_start_y.setDecimals(3)
        self.QDoubleSpinBox_start_y.setValue(0.000)
        
        self.QDoubleSpinBox_start_z = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_start_z.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_start_z.setRange(0, 50)
        self.QDoubleSpinBox_start_z.setSingleStep(0.001)
        self.QDoubleSpinBox_start_z.setDecimals(3)
        self.QDoubleSpinBox_start_z.setValue(0.000)
        
        self.QDoubleSpinBox_step_x = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_step_x.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_step_x.setRange(0, 50)
        self.QDoubleSpinBox_step_x.setSingleStep(0.001)
        self.QDoubleSpinBox_step_x.setDecimals(3)
        self.QDoubleSpinBox_step_x.setValue(0.040)
        
        self.QDoubleSpinBox_step_y = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_step_y.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_step_y.setRange(0, 50)
        self.QDoubleSpinBox_step_y.setSingleStep(0.001)
        self.QDoubleSpinBox_step_y.setDecimals(3)
        self.QDoubleSpinBox_step_y.setValue(0.040)
        
        self.QDoubleSpinBox_step_z = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_step_z.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_step_z.setRange(0, 50)
        self.QDoubleSpinBox_step_z.setSingleStep(0.001)
        self.QDoubleSpinBox_step_z.setDecimals(3)
        self.QDoubleSpinBox_step_z.setValue(0.080)
        
        self.QDoubleSpinBox_range_x = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_range_x.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_range_x.setRange(0, 50)
        self.QDoubleSpinBox_range_x.setSingleStep(0.001)
        self.QDoubleSpinBox_range_x.setDecimals(3)
        self.QDoubleSpinBox_range_x.setValue(0.400)
        
        self.QDoubleSpinBox_range_y = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_range_y.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_range_y.setRange(0, 50)
        self.QDoubleSpinBox_range_y.setSingleStep(0.001)
        self.QDoubleSpinBox_range_y.setDecimals(3)
        self.QDoubleSpinBox_range_y.setValue(0.400)
        
        self.QDoubleSpinBox_range_z = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_range_z.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_range_z.setRange(0, 50)
        self.QDoubleSpinBox_range_z.setSingleStep(0.001)
        self.QDoubleSpinBox_range_z.setDecimals(3)
        self.QDoubleSpinBox_range_z.setValue(0.800)
        
        self.QLabel_ttl_starts = QtWidgets.QLabel(str('TTL_starts / s'))
        self.QLabel_ttl_starts.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_stops = QtWidgets.QLabel(str('TTL_stops / s'))
        self.QLabel_ttl_stops.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_on_405 = QtWidgets.QLabel(str('On 405'))
        self.QLabel_ttl_on_405.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_off_488_0 = QtWidgets.QLabel(str('Off 488_0'))
        self.QLabel_ttl_off_488_0.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_off_488_1 = QtWidgets.QLabel(str('Off 488_1'))
        self.QLabel_ttl_off_488_1.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_read_488_2 = QtWidgets.QLabel(str('Read 488_2'))
        self.QLabel_ttl_read_488_2.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLabel_ttl_camera = QtWidgets.QLabel(str('Camera'))
        self.QLabel_ttl_camera.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QDoubleSpinBox_ttl_start_on_405 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_start_on_405.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_start_on_405.setRange(0, 1)
        self.QDoubleSpinBox_ttl_start_on_405.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_start_on_405.setDecimals(3)
        self.QDoubleSpinBox_ttl_start_on_405.setValue(0.005)
        
        self.QDoubleSpinBox_ttl_start_off_488_0 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_start_off_488_0.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_start_off_488_0.setRange(0, 1)
        self.QDoubleSpinBox_ttl_start_off_488_0.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_start_off_488_0.setDecimals(3)
        self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.020)
        
        self.QDoubleSpinBox_ttl_start_off_488_1 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_start_off_488_1.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_start_off_488_1.setRange(0, 1)
        self.QDoubleSpinBox_ttl_start_off_488_1.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_start_off_488_1.setDecimals(3)
        self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.020)
        
        self.QDoubleSpinBox_ttl_start_read_488_2 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_start_read_488_2.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_start_read_488_2.setRange(0, 1)
        self.QDoubleSpinBox_ttl_start_read_488_2.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_start_read_488_2.setDecimals(3)
        self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.040)
        
        self.QDoubleSpinBox_ttl_start_camera = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_start_camera.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_start_camera.setRange(0, 1)
        self.QDoubleSpinBox_ttl_start_camera.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_start_camera.setDecimals(3)
        self.QDoubleSpinBox_ttl_start_camera.setValue(0.040)
        
        self.QDoubleSpinBox_ttl_stop_on_405 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_stop_on_405.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_stop_on_405.setRange(0, 1)
        self.QDoubleSpinBox_ttl_stop_on_405.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_stop_on_405.setDecimals(3)
        self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.015)
        
        self.QDoubleSpinBox_ttl_stop_off_488_0 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_stop_off_488_0.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_stop_off_488_0.setRange(0, 1)
        self.QDoubleSpinBox_ttl_stop_off_488_0.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_stop_off_488_0.setDecimals(3)
        self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.035)
        
        self.QDoubleSpinBox_ttl_stop_off_488_1 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_stop_off_488_1.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_stop_off_488_1.setRange(0, 1)
        self.QDoubleSpinBox_ttl_stop_off_488_1.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_stop_off_488_1.setDecimals(3)
        self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.035)
        
        self.QDoubleSpinBox_ttl_stop_read_488_2 = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_stop_read_488_2.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_stop_read_488_2.setRange(0, 1)
        self.QDoubleSpinBox_ttl_stop_read_488_2.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_stop_read_488_2.setDecimals(3)
        self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.060)
        
        self.QDoubleSpinBox_ttl_stop_camera = QtWidgets.QDoubleSpinBox()
        self.QDoubleSpinBox_ttl_stop_camera.setStyleSheet("background-color: dark; color: black; font: bold Arial 10px")
        self.QDoubleSpinBox_ttl_stop_camera.setRange(0, 1)
        self.QDoubleSpinBox_ttl_stop_camera.setSingleStep(0.001)
        self.QDoubleSpinBox_ttl_stop_camera.setDecimals(3)
        self.QDoubleSpinBox_ttl_stop_camera.setValue(0.060)
        
        self.QLabel_trigger_parameter = QtWidgets.QLabel(str('Pre-sets'))
        self.QLabel_trigger_parameter.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QComboBox_trigger_parameter = QtWidgets.QComboBox()
        self.QComboBox_trigger_parameter.addItems(['Default', 'UseDefined_1', 'UseDefined_2', 'UseDefined_3'])
        self.QComboBox_trigger_parameter.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_2d_resolft = QtWidgets.QPushButton('2D_RESOLFT', self)
        self.QPushButton_2d_resolft.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_3d_resolft = QtWidgets.QPushButton('3D_RESOLFT', self)
        self.QPushButton_3d_resolft.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        self.QPushButton_2d_beadscan = QtWidgets.QPushButton('2D_BeadScan', self)
        self.QPushButton_2d_beadscan.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        Layout_Triggers.addWidget(self.QLabel_cycle_period, 0, 0, 1, 2)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_cycle_period, 1, 0, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_piezo_start, 0, 2, 1, 2)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_piezo_start, 1, 2, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_trigger_parameter, 0, 4, 1, 2)
        Layout_Triggers.addWidget(self.QComboBox_trigger_parameter, 1, 4, 1, 2)
        Layout_Triggers.addWidget(self.QLabel_step_sizes, 4, 0, 1, 2)
        Layout_Triggers.addWidget(self.QLabel_scan_ranges, 5, 0, 1, 2)
        Layout_Triggers.addWidget(self.QLabel_axis_x, 2, 2, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_axis_y, 2, 3, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_axis_z, 2, 4, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_start_x, 3, 2, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_start_y, 3, 3, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_start_z, 3, 4, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_step_x, 4, 2, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_step_y, 4, 3, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_step_z, 4, 4, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_range_x, 5, 2, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_range_y, 5, 3, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_range_z, 5, 4, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_ttl_starts, 7, 0, 1, 2)
        Layout_Triggers.addWidget(self.QLabel_ttl_stops, 8, 0, 1, 2)
        Layout_Triggers.addWidget(self.QLabel_ttl_on_405, 6, 2, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_ttl_off_488_0, 6, 3, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_ttl_off_488_1, 6, 4, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_ttl_read_488_2, 6, 5, 1, 1)
        Layout_Triggers.addWidget(self.QLabel_ttl_camera, 6, 6, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_start_on_405, 7, 2, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_start_off_488_0, 7, 3, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_start_off_488_1, 7, 4, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_start_read_488_2, 7, 5, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_start_camera, 7, 6, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_stop_on_405, 8, 2, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_stop_off_488_0, 8, 3, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_stop_off_488_1, 8, 4, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_stop_read_488_2, 8, 5, 1, 1)
        Layout_Triggers.addWidget(self.QDoubleSpinBox_ttl_stop_camera, 8, 6, 1, 1)
        Layout_Triggers.addWidget(self.QPushButton_2d_beadscan, 9, 0, 1, 2)
        Layout_Triggers.addWidget(self.QPushButton_2d_resolft, 9, 2, 1, 2)
        Layout_Triggers.addWidget(self.QPushButton_3d_resolft, 9, 4, 1, 2)
        
        Group_Triggers.setLayout(Layout_Triggers)

        Layout_File = QtWidgets.QGridLayout()
        
        self.QLabel_file_name = QtWidgets.QLabel(str('File name'))
        self.QLabel_file_name.setStyleSheet("background-color: dark; color: white; font: bold Arial 10px")
        
        self.QLineEdit_filename = QtWidgets.QLineEdit()
        self.QLineEdit_filename.setStyleSheet("background-color: dark; font: bold Arial 10px")
        
        self.QPushButton_save = QtWidgets.QPushButton('Save', self)
        self.QPushButton_save.setStyleSheet("background-color: lightgrey; color: dark; font: bold Arial 10px")
        
        Layout_File.addWidget(self.QLabel_file_name, 0, 0, 1, 1)
        Layout_File.addWidget(self.QLineEdit_filename, 0, 1, 1, 4)
        Layout_File.addWidget(self.QPushButton_save, 0, 5, 1, 1)
        
        Group_File.setLayout(Layout_File)
        
        self.QPushButton_setcoordinates.clicked.connect(self.set_coordinates)
        self.QPushButton_resetcoordinates.clicked.connect(self.reset_coordinates)
        self.QDoubleSpinBox_stage_x.valueChanged.connect(self.piezo_move)
        self.QDoubleSpinBox_stage_y.valueChanged.connect(self.piezo_move)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.piezo_move)
        self.QPushButton_move_deck_up.clicked.connect(self.deck_move_up)
        self.QPushButton_move_deck_down.clicked.connect(self.deck_move_down)
        self.QPushButton_move_deck.clicked.connect(self.deck_move)
        self.QPushButton_move_deck_stop.clicked.connect(self.deck_move_stop)
        self.QPushButton_start_video.clicked.connect(self.start_video)
        self.QPushButton_stop_video.clicked.connect(self.stop_video)
        self.QPushButton_start_fft.clicked.connect(self.run_ft)
        self.QPushButton_stop_fft.clicked.connect(self.stop_ft)
        self.QPushButton_save.clicked.connect(self.save)
        self.QPushButton_laser_488_0.clicked.connect(self.set_laser_488_0)
        self.QPushButton_laser_488_1.clicked.connect(self.set_laser_488_1)
        self.QPushButton_laser_488_2.clicked.connect(self.set_laser_488_2)
        self.QPushButton_laser_405.clicked.connect(self.set_laser_405)
        self.QPushButton_2d_resolft.clicked.connect(self.resolft_2d)
        self.QPushButton_3d_resolft.clicked.connect(self.resolft_3d)
        self.QPushButton_2d_beadscan.clicked.connect(self.beadscan_2d)
        self.QComboBox_trigger_parameter.currentIndexChanged.connect(self.update_trigger_parameter_sets)


    def set_laser_488_0(self):
        if self.QPushButton_laser_488_0.isChecked():
            self.QPushButton_laser_488_0.setStyleSheet("background-color : red")
            self.Signal_setlaseron_488_0.emit()
        else:
            self.QPushButton_laser_488_0.setStyleSheet("background-color : lightgrey")
            self.Signal_setlaseroff_488_0.emit()
        
    def set_laser_488_1(self):
        if self.QPushButton_laser_488_1.isChecked():
            self.QPushButton_laser_488_1.setStyleSheet("background-color : red")
            self.Signal_setlaseron_488_1.emit()
        else:
            self.QPushButton_laser_488_1.setStyleSheet("background-color : lightgrey")
            self.Signal_setlaseroff_488_1.emit()

    def set_laser_488_2(self):
        if self.QPushButton_laser_488_2.isChecked():
            self.QPushButton_laser_488_2.setStyleSheet("background-color : red")
            self.Signal_setlaseron_488_2.emit()
        else:
            self.QPushButton_laser_488_2.setStyleSheet("background-color : lightgrey")
            self.Signal_setlaseroff_488_2.emit()
        
    def set_laser_405(self):
        if self.QPushButton_laser_405.isChecked():
            self.QPushButton_laser_405.setStyleSheet("background-color : red")
            self.Signal_setlaseron_405.emit()
        else:
            self.QPushButton_laser_405.setStyleSheet("background-color : lightgrey")
            self.Signal_setlaseroff_405.emit()
    
    def set_coordinates(self):
        self.Signal_setcoordinates.emit()
        
    def reset_coordinates(self):
        self.Signal_resetcoordinates.emit()
        self.QSpinBox_coordinate_x.setValue(1)
        self.QSpinBox_coordinate_y.setValue(1)
        self.QSpinBox_coordinate_n.setValue(1024)
        self.QSpinBox_coordinate_bin.setValue(1)
    
    def deck_move_up(self):
        self.Signal_deck_up.emit()
        
    def deck_move_down(self):
        self.Signal_deck_down.emit()
        
    def deck_move(self):
        self.Signal_deck_move.emit()
    
    def deck_move_stop(self):
        self.Signal_deck_move_stop.emit()
    
    def piezo_move(self):
        self.Signal_piezo_move.emit()
        
    def start_video(self):
        self.Signal_start_video.emit()
        
    def stop_video(self):
        self.Signal_stop_video.emit()
        
    def run_ft(self):
        self.Signal_run_fft.emit()
        
    def stop_ft(self):
        self.Signal_stop_fft.emit()
        
    def save(self):
        self.Signal_save_file.emit()
        
    def resolft_2d(self):
        self.Signal_2d_resolft.emit()
        
    def resolft_3d(self):
        self.Signal_3d_resolft.emit()
        
    def beadscan_2d(self):
        self.Signal_beadscan_2d.emit()
        
    def update_trigger_parameter_sets(self):
        cind = self.QComboBox_trigger_parameter.currentIndex()
        if cind==0:
            self.QDoubleSpinBox_cycle_period.setValue(0.100)
            self.QDoubleSpinBox_piezo_start.setValue(0.050)
            self.QDoubleSpinBox_start_x.setValue(0.000)
            self.QDoubleSpinBox_start_y.setValue(0.000)
            self.QDoubleSpinBox_start_z.setValue(0.000)
            self.QDoubleSpinBox_step_x.setValue(0.040)
            self.QDoubleSpinBox_step_y.setValue(0.040)
            self.QDoubleSpinBox_step_z.setValue(0.040)
            self.QDoubleSpinBox_range_x.setValue(1.000)
            self.QDoubleSpinBox_range_y.setValue(1.000)
            self.QDoubleSpinBox_range_z.setValue(0.600)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.005)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.010)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.012)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.017)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.012)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.017)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.020)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.040)
            self.QDoubleSpinBox_ttl_start_camera.setValue(0.020)
            self.QDoubleSpinBox_ttl_stop_camera.setValue(0.040)
        if cind==1:
            self.QDoubleSpinBox_cycle_period.setValue(0.075)
            self.QDoubleSpinBox_piezo_start.setValue(0.050)
            self.QDoubleSpinBox_start_x.setValue(0.000)
            self.QDoubleSpinBox_start_y.setValue(0.000)
            self.QDoubleSpinBox_start_z.setValue(0.000)
            self.QDoubleSpinBox_step_x.setValue(0.032)
            self.QDoubleSpinBox_step_y.setValue(0.032)
            self.QDoubleSpinBox_step_z.setValue(0.050)
            self.QDoubleSpinBox_range_x.setValue(0.960)
            self.QDoubleSpinBox_range_y.setValue(0.960)
            self.QDoubleSpinBox_range_z.setValue(0.500)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.005)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.006)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.010)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.010)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.041)
            self.QDoubleSpinBox_ttl_start_camera.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_camera.setValue(0.041)
