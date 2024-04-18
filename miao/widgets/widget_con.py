from PyQt5 import QtWidgets, QtCore

from miao.utilities import customized_widgets as cw


class ConWidget(QtWidgets.QWidget):
    Signal_check_emccd_temperature = QtCore.pyqtSignal()
    Signal_switch_emccd_cooler = QtCore.pyqtSignal(bool)
    Signal_piezo_move = QtCore.pyqtSignal(str, float, float, float)
    Signal_deck_read_position = QtCore.pyqtSignal()
    Signal_deck_zero_position = QtCore.pyqtSignal()
    Signal_deck_move_single_step = QtCore.pyqtSignal(bool)
    Signal_deck_move_continuous = QtCore.pyqtSignal(bool, int, float)
    Signal_galvo_set = QtCore.pyqtSignal(float, float)
    Signal_galvo_scan_update = QtCore.pyqtSignal()
    Signal_set_laser = QtCore.pyqtSignal(list, bool, float)
    Signal_daq_update = QtCore.pyqtSignal(int)
    Signal_plot_trigger = QtCore.pyqtSignal()
    Signal_video = QtCore.pyqtSignal(bool, str)
    Signal_fft = QtCore.pyqtSignal(bool)
    Signal_plot_profile = QtCore.pyqtSignal(bool)
    Signal_data_acquire = QtCore.pyqtSignal(str, int)
    Signal_save_file = QtCore.pyqtSignal(str)

    def __init__(self, config, logg, path, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.config = config
        self.logg = logg
        self.data_folder = path
        self._setup_ui()
        self._set_signal_connections()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self._create_docks()
        self._create_widgets()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for name, (dock, group) in self.docks.items():
            splitter.addWidget(dock)
            group.setLayout(self.widgets[name])
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_docks(self):
        self.docks = {
            "camera": cw.create_dock("Camera"),
            "position": cw.create_dock("Position"),
            "laser": cw.create_dock("Laser"),
            "daq": cw.create_dock("Daq"),
            "video": cw.create_dock("Live Imaging"),
            "acquisition": cw.create_dock("Data Acquisition")
        }

    def _create_widgets(self):
        self.widgets = {
            "camera": self._create_camera_widgets(),
            "position": self._create_position_widgets(),
            "laser": self._create_laser_widgets(),
            "daq": self._create_daq_widgets(),
            "video": self._create_video_widgets(),
            "acquisition": self._create_acquisition_widgets()
        }

    def _create_camera_widgets(self):
        layout_camera = QtWidgets.QHBoxLayout()
        self.QLCDNumber_ccd_tempetature = cw.lcdnumber_widget(0, 3)
        self.QPushButton_emccd_cooler_check = cw.pushbutton_widget('Check', False, True)
        self.QPushButton_emccd_cooler_switch = cw.pushbutton_widget('Cooler OFF', True, True, True)
        self.QDoubleSpinBox_emccd_exposure_time = cw.doublespinbox_widget(0, 10, 0.005, 3, 0.02)
        self.QSpinBox_emccd_gain = cw.spinbox_widget(0, 300, 1, 0)
        self.QSpinBox_emccd_coordinate_x = cw.spinbox_widget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_y = cw.spinbox_widget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_n = cw.spinbox_widget(0, 1024, 1, 1024)
        self.QSpinBox_emccd_coordinate_bin = cw.spinbox_widget(0, 1024, 1, 1)
        self.QDoubleSpinBox_ttl_start_emccd = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_emccd = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.emccd_scroll_area, emccd_scroll_layout = cw.create_scroll_area()
        emccd_scroll_layout.addRow(cw.label_widget(str('EMCCD')))
        emccd_scroll_layout.addRow(cw.frame_widget())
        emccd_scroll_layout.addRow(cw.label_widget(str('Temperature')), self.QLCDNumber_ccd_tempetature)
        emccd_scroll_layout.addRow(self.QPushButton_emccd_cooler_check, self.QPushButton_emccd_cooler_switch)
        emccd_scroll_layout.addRow(cw.label_widget(str('EMGain')), self.QSpinBox_emccd_gain)
        emccd_scroll_layout.addRow(cw.label_widget(str('X')), self.QSpinBox_emccd_coordinate_x)
        emccd_scroll_layout.addRow(cw.label_widget(str('Y')), self.QSpinBox_emccd_coordinate_y)
        emccd_scroll_layout.addRow(cw.label_widget(str('N')), self.QSpinBox_emccd_coordinate_n)
        emccd_scroll_layout.addRow(cw.label_widget(str('Bin')), self.QSpinBox_emccd_coordinate_bin)
        emccd_scroll_layout.addRow(cw.label_widget(str('Exposure / s')), self.QDoubleSpinBox_emccd_exposure_time)
        emccd_scroll_layout.addRow(cw.label_widget(str('From / s')), cw.label_widget(str('To / s')))
        emccd_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_emccd, self.QDoubleSpinBox_ttl_stop_emccd)
        self.QDoubleSpinBox_scmos_exposure_time = cw.doublespinbox_widget(0, 10, 0.005, 3, 0.01)
        self.QSpinBox_scmos_coordinate_x = cw.spinbox_widget(0, 2048, 1, 0)
        self.QSpinBox_scmos_coordinate_y = cw.spinbox_widget(0, 2048, 1, 0)
        self.QSpinBox_scmos_coordinate_n = cw.spinbox_widget(0, 2048, 1, 2048)
        self.QSpinBox_scmos_coordinate_bin = cw.spinbox_widget(0, 2048, 1, 1)
        self.QDoubleSpinBox_ttl_start_scmos = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_scmos = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.scmos_scroll_area, scmos_scroll_layout = cw.create_scroll_area()
        scmos_scroll_layout.addRow(cw.label_widget(str('sCMOS')))
        scmos_scroll_layout.addRow(cw.frame_widget())
        scmos_scroll_layout.addRow(cw.label_widget(str('Exposure / s')), self.QDoubleSpinBox_scmos_exposure_time)
        scmos_scroll_layout.addRow(cw.label_widget(str('X')), self.QSpinBox_scmos_coordinate_x)
        scmos_scroll_layout.addRow(cw.label_widget(str('Y')), self.QSpinBox_scmos_coordinate_y)
        scmos_scroll_layout.addRow(cw.label_widget(str('N')), self.QSpinBox_scmos_coordinate_n)
        scmos_scroll_layout.addRow(cw.label_widget(str('Bin')), self.QSpinBox_scmos_coordinate_bin)
        scmos_scroll_layout.addRow(cw.label_widget(str('From / s')), cw.label_widget(str('To / s')))
        scmos_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_scmos, self.QDoubleSpinBox_ttl_stop_scmos)
        self.QDoubleSpinBox_tis_exposure_time = cw.doublespinbox_widget(0, 10, 0.005, 3, 0.01)
        self.QDoubleSpinBox_ttl_start_tis = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_tis = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.tis_scroll_area, tis_scroll_layout = cw.create_scroll_area()
        tis_scroll_layout.addRow(cw.label_widget(str('TIS')))
        tis_scroll_layout.addRow(cw.frame_widget())
        tis_scroll_layout.addRow(cw.label_widget(str('Exposure / s')), self.QDoubleSpinBox_tis_exposure_time)
        tis_scroll_layout.addRow(cw.label_widget(str('From / s')), cw.label_widget(str('To / s')))
        tis_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_tis, self.QDoubleSpinBox_ttl_stop_tis)
        layout_camera.addWidget(self.emccd_scroll_area)
        layout_camera.addWidget(self.scmos_scroll_area)
        layout_camera.addWidget(self.tis_scroll_area)
        return layout_camera

    def _create_position_widgets(self):
        layout_position = QtWidgets.QHBoxLayout()
        self.QLCDNumber_deck_position = cw.lcdnumber_widget()
        self.QPushButton_deck_position = cw.pushbutton_widget('Read')
        self.QPushButton_deck_position_zero = cw.pushbutton_widget('Zero')
        self.QPushButton_move_deck_up = cw.pushbutton_widget('Up')
        self.QPushButton_move_deck_down = cw.pushbutton_widget('Down')
        self.QSpinBox_deck_direction = cw.spinbox_widget(-1, 1, 2, 1)
        self.QDoubleSpinBox_deck_velocity = cw.doublespinbox_widget(0.02, 1.50, 0.02, 2, 0.02)
        self.QPushButton_move_deck = cw.pushbutton_widget('Move', checkable=True)
        self.mad_deck_scroll_area, mad_deck_scroll_layout = cw.create_scroll_area()
        mad_deck_scroll_layout.addRow(cw.label_widget(str('Mad Deck')))
        mad_deck_scroll_layout.addRow(cw.frame_widget())
        mad_deck_scroll_layout.addRow(cw.label_widget(str('Position (mm)')), self.QLCDNumber_deck_position)
        mad_deck_scroll_layout.addRow(self.QPushButton_deck_position, self.QPushButton_deck_position_zero)
        mad_deck_scroll_layout.addRow(cw.label_widget(str('Direction (+up)')), self.QSpinBox_deck_direction)
        mad_deck_scroll_layout.addRow(cw.label_widget(str('Velocity (mm)')), self.QDoubleSpinBox_deck_velocity)
        mad_deck_scroll_layout.addRow(self.QPushButton_move_deck)
        mad_deck_scroll_layout.addRow(cw.label_widget(str('Single step')))
        mad_deck_scroll_layout.addRow(self.QPushButton_move_deck_up, self.QPushButton_move_deck_down)
        self.QDoubleSpinBox_stage_x = cw.doublespinbox_widget(0, 100, 0.02, 2, 50.00)
        self.QLCDNumber_piezo_position_x = cw.lcdnumber_widget()
        self.QDoubleSpinBox_step_x = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.034)
        self.QDoubleSpinBox_range_x = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.850)
        self.QDoubleSpinBox_stage_y = cw.doublespinbox_widget(0, 100, 0.02, 2, 50.00)
        self.QLCDNumber_piezo_position_y = cw.lcdnumber_widget()
        self.QDoubleSpinBox_step_y = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.034)
        self.QDoubleSpinBox_range_y = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.850)
        self.QDoubleSpinBox_stage_z = cw.doublespinbox_widget(0, 100, 0.04, 2, 50.00)
        self.QLCDNumber_piezo_position_z = cw.lcdnumber_widget()
        self.QDoubleSpinBox_step_z = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.160)
        self.QDoubleSpinBox_range_z = cw.doublespinbox_widget(0, 50, 0.001, 3, 4.80)
        self.QDoubleSpinBox_piezo_return_time = cw.doublespinbox_widget(0, 50, 0.01, 2, 0.12)
        self.mcl_piezo_scroll_area, mcl_piezo_scroll_layout = cw.create_scroll_area()
        mcl_piezo_scroll_layout.addRow(cw.label_widget(str('MCL Piezo')))
        mcl_piezo_scroll_layout.addRow(cw.frame_widget())
        hbox_xp = QtWidgets.QHBoxLayout()
        hbox_xp.addWidget(cw.label_widget(str('X (um)')))
        hbox_xp.addWidget(self.QDoubleSpinBox_stage_x)
        hbox_xp.addWidget(self.QLCDNumber_piezo_position_x)
        mcl_piezo_scroll_layout.addRow(hbox_xp)
        mcl_piezo_scroll_layout.addRow(cw.label_widget(str('Step / um')), cw.label_widget(str('Range / um')))
        mcl_piezo_scroll_layout.addRow(self.QDoubleSpinBox_step_x, self.QDoubleSpinBox_range_x)
        mcl_piezo_scroll_layout.addRow(cw.frame_widget())
        hbox_yp = QtWidgets.QHBoxLayout()
        hbox_yp.addWidget(cw.label_widget(str('Y (um)')))
        hbox_yp.addWidget(self.QDoubleSpinBox_stage_y)
        hbox_yp.addWidget(self.QLCDNumber_piezo_position_y)
        mcl_piezo_scroll_layout.addRow(hbox_yp)
        mcl_piezo_scroll_layout.addRow(cw.label_widget(str('Step / um')), cw.label_widget(str('Range / um')))
        mcl_piezo_scroll_layout.addRow(self.QDoubleSpinBox_step_y, self.QDoubleSpinBox_range_y)
        mcl_piezo_scroll_layout.addRow(cw.frame_widget())
        hbox_zp = QtWidgets.QHBoxLayout()
        hbox_zp.addWidget(cw.label_widget(str('Z (um)')))
        hbox_zp.addWidget(self.QDoubleSpinBox_stage_z)
        hbox_zp.addWidget(self.QLCDNumber_piezo_position_z)
        mcl_piezo_scroll_layout.addRow(hbox_zp)
        mcl_piezo_scroll_layout.addRow(cw.label_widget(str('Step / um')), cw.label_widget(str('Range / um')))
        mcl_piezo_scroll_layout.addRow(self.QDoubleSpinBox_step_z, self.QDoubleSpinBox_range_z)
        mcl_piezo_scroll_layout.addRow(cw.frame_widget())
        mcl_piezo_scroll_layout.addRow(cw.label_widget(str('Piezo Return / s')), self.QDoubleSpinBox_piezo_return_time)
        self.QSpinBox_galvo_frequency = cw.spinbox_widget(0, 4000, 1, 147)
        self.QDoubleSpinBox_galvo_x = cw.doublespinbox_widget(-10, 10, 0.0001, 4, 0)
        self.QDoubleSpinBox_galvo_y = cw.doublespinbox_widget(-10, 10, 0.0001, 4, 0)
        self.QDoubleSpinBox_galvo_range_x = cw.doublespinbox_widget(-10, 10, 0.0001, 4, 0.4)
        self.QDoubleSpinBox_galvo_range_y = cw.doublespinbox_widget(-10, 10, 0.0001, 4, 0.4)
        self.QDoubleSpinBox_dot_range_x = cw.doublespinbox_widget(0, 20, 0.0002, 4, 0.2)
        self.QDoubleSpinBox_dot_range_y = cw.doublespinbox_widget(0, 20, 0.0002, 4, 0.2)
        self.QSpinBox_dot_step = cw.spinbox_widget(0, 4000, 1, 30)
        self.QDoubleSpinBox_dot_step = cw.doublespinbox_widget(0, 20, 0.0001, 4, 0.0172)
        self.QSpinBox_galvo_dwell = cw.spinbox_widget(0, 4000, 1, 1)
        self.QSpinBox_laser_delay = cw.spinbox_widget(0, 4000, 1, 0)
        self.galvo_scroll_area, galvo_scroll_layout = cw.create_scroll_area()
        galvo_scroll_layout.addRow(cw.label_widget(str('Galvo Scanner')))
        galvo_scroll_layout.addRow(cw.frame_widget())
        galvo_scroll_layout.addRow(cw.label_widget(str('Frequency / Hz')), self.QSpinBox_galvo_frequency)
        galvo_scroll_layout.addRow(cw.frame_widget())
        galvo_scroll_layout.addRow(cw.label_widget(str('X / v')), self.QDoubleSpinBox_galvo_x)
        galvo_scroll_layout.addRow(cw.label_widget(str('Scan Range / V')), self.QDoubleSpinBox_galvo_range_x)
        galvo_scroll_layout.addRow(cw.label_widget(str('Dot Range / V')), self.QDoubleSpinBox_dot_range_x)
        galvo_scroll_layout.addRow(cw.label_widget(str('Dot Step / sample')), self.QSpinBox_dot_step)
        galvo_scroll_layout.addRow(cw.label_widget(str('Dot Step / volt')), self.QDoubleSpinBox_dot_step)
        galvo_scroll_layout.addRow(cw.label_widget(str('Dot Dwell / sample')), self.QSpinBox_galvo_dwell)
        galvo_scroll_layout.addRow(cw.label_widget(str('Offset / sample')), self.QSpinBox_laser_delay)
        galvo_scroll_layout.addRow(cw.frame_widget())
        galvo_scroll_layout.addRow(cw.label_widget(str('Y / v')), self.QDoubleSpinBox_galvo_y)
        galvo_scroll_layout.addRow(cw.label_widget(str('Range / V')), self.QDoubleSpinBox_galvo_range_y)
        galvo_scroll_layout.addRow(cw.label_widget(str('Dot Range / V')), self.QDoubleSpinBox_dot_range_y)
        layout_position.addWidget(self.mad_deck_scroll_area)
        layout_position.addWidget(self.mcl_piezo_scroll_area)
        layout_position.addWidget(self.galvo_scroll_area)
        return layout_position

    def _create_laser_widgets(self):
        layout_illumination = QtWidgets.QHBoxLayout()
        self.QRadioButton_laser_405 = cw.radiobutton_widget('405 nm')
        self.QDoubleSpinBox_laserpower_405 = cw.doublespinbox_widget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_405 = cw.pushbutton_widget('ON', checkable=True)
        self.QRadioButton_laser_488_0 = cw.radiobutton_widget('488 nm #0')
        self.QDoubleSpinBox_laserpower_488_0 = cw.doublespinbox_widget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488_0 = cw.pushbutton_widget('ON', checkable=True)
        self.QRadioButton_laser_488_1 = cw.radiobutton_widget('488 nm #1')
        self.QDoubleSpinBox_laserpower_488_1 = cw.doublespinbox_widget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488_1 = cw.pushbutton_widget('ON', checkable=True)
        self.QRadioButton_laser_488_2 = cw.radiobutton_widget('488 nm #2')
        self.QDoubleSpinBox_laserpower_488_2 = cw.doublespinbox_widget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488_2 = cw.pushbutton_widget('ON', checkable=True)
        self.QLabel_laser_405_starts = cw.label_widget(str('From / s'))
        self.QLabel_laser_405_stops = cw.label_widget(str('To / s'))
        self.QDoubleSpinBox_ttl_start_on_405 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_on_405 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.QLabel_laser_488_0_starts = cw.label_widget(str('From / s'))
        self.QLabel_laser_488_0_stops = cw.label_widget(str('To / s'))
        self.QDoubleSpinBox_ttl_start_off_488_0 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_off_488_0 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.QLabel_laser_488_1_starts = cw.label_widget(str('From / s'))
        self.QLabel_laser_488_1_stops = cw.label_widget(str('To / s'))
        self.QDoubleSpinBox_ttl_start_off_488_1 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_off_488_1 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.QLabel_laser_488_2_starts = cw.label_widget(str('From / s'))
        self.QLabel_laser_488_2_stops = cw.label_widget(str('To / s'))
        self.QDoubleSpinBox_ttl_start_read_488_2 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.008)
        self.QDoubleSpinBox_ttl_stop_read_488_2 = cw.doublespinbox_widget(0, 50, 0.001, 3, 0.032)
        self.laser_405_scroll_area, laser_405_scroll_layout = cw.create_scroll_area()
        self.laser_488_0_scroll_area, laser_488_0_scroll_layout = cw.create_scroll_area()
        self.laser_488_1_scroll_area, laser_488_1_scroll_layout = cw.create_scroll_area()
        self.laser_488_2_scroll_area, laser_488_2_scroll_layout = cw.create_scroll_area()
        laser_405_scroll_layout.addRow(self.QRadioButton_laser_405, self.QDoubleSpinBox_laserpower_405)
        laser_405_scroll_layout.addRow(self.QPushButton_laser_405)
        laser_405_scroll_layout.addRow(self.QLabel_laser_405_starts, self.QLabel_laser_405_stops)
        laser_405_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_on_405, self.QDoubleSpinBox_ttl_stop_on_405)
        laser_488_0_scroll_layout.addRow(self.QRadioButton_laser_488_0, self.QDoubleSpinBox_laserpower_488_0)
        laser_488_0_scroll_layout.addRow(self.QPushButton_laser_488_0)
        laser_488_0_scroll_layout.addRow(self.QLabel_laser_488_0_starts, self.QLabel_laser_488_0_stops)
        laser_488_0_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_off_488_0,
                                         self.QDoubleSpinBox_ttl_stop_off_488_0)
        laser_488_1_scroll_layout.addRow(self.QRadioButton_laser_488_1, self.QDoubleSpinBox_laserpower_488_1)
        laser_488_1_scroll_layout.addRow(self.QPushButton_laser_488_1)
        laser_488_1_scroll_layout.addRow(self.QLabel_laser_488_1_starts, self.QLabel_laser_488_1_stops)
        laser_488_1_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_off_488_1,
                                         self.QDoubleSpinBox_ttl_stop_off_488_1)
        laser_488_2_scroll_layout.addRow(self.QRadioButton_laser_488_2, self.QDoubleSpinBox_laserpower_488_2)
        laser_488_2_scroll_layout.addRow(self.QPushButton_laser_488_2)
        laser_488_2_scroll_layout.addRow(self.QLabel_laser_488_2_starts, self.QLabel_laser_488_2_stops)
        laser_488_2_scroll_layout.addRow(self.QDoubleSpinBox_ttl_start_read_488_2,
                                         self.QDoubleSpinBox_ttl_stop_read_488_2)
        layout_illumination.addWidget(self.laser_405_scroll_area)
        layout_illumination.addWidget(self.laser_488_0_scroll_area)
        layout_illumination.addWidget(self.laser_488_1_scroll_area)
        layout_illumination.addWidget(self.laser_488_2_scroll_area)
        return layout_illumination

    def _create_daq_widgets(self):
        layout_daq = QtWidgets.QHBoxLayout()
        self.QSpinBox_daq_sample_rate = cw.spinbox_widget(100, 1250, 1, 100)
        layout_daq.addWidget(cw.label_widget(str('DAQ Sample Rate / KS/s')))
        layout_daq.addWidget(self.QSpinBox_daq_sample_rate)
        return layout_daq

    def _create_video_widgets(self):
        layout_video = QtWidgets.QHBoxLayout()
        self.QComboBox_imaging_camera_selection = cw.combobox_widget(list_items=["EMCCD", "SCMOS", "TIS"])
        self.QComboBox_live_modes = cw.combobox_widget(list_items=["Wide Field", "Dot Scan"]) #"Line Scan"
        self.QPushButton_video = cw.pushbutton_widget("Video", checkable=True)
        self.QPushButton_fft = cw.pushbutton_widget("FFT", checkable=True, enable=False)
        self.QComboBox_profile_axis = cw.combobox_widget(list_items=["X", "Y"])
        self.QPushButton_plot_profile = cw.pushbutton_widget("Plot Profile", checkable=True, enable=False)
        self.QPushButton_plot_trigger = cw.pushbutton_widget("Plot Triggers")
        layout_video.addWidget(self.QComboBox_imaging_camera_selection)
        layout_video.addWidget(self.QComboBox_live_modes)
        layout_video.addWidget(self.QPushButton_video)
        layout_video.addWidget(self.QPushButton_fft)
        layout_video.addWidget(self.QComboBox_profile_axis)
        layout_video.addWidget(self.QPushButton_plot_profile)
        layout_video.addWidget(self.QPushButton_plot_trigger)
        return layout_video

    def _create_acquisition_widgets(self):
        layout_acquisition = QtWidgets.QGridLayout()
        self.QComboBox_acquisition_modes = cw.combobox_widget(list_items=["Wide Field 2D", "Wide Field 3D",
                                                                          "Monalisa Scan 2D", "Monalisa Scan 3D",
                                                                          "Dot Scan 2D", "Dot Scan 3D"])
        self.QSpinBox_acquisition_number = cw.spinbox_widget(1, 50000, 1, 1)
        self.QPushButton_acquire = cw.pushbutton_widget('Acquire')
        self.QPushButton_save = cw.pushbutton_widget('Save')
        layout_acquisition.addWidget(cw.label_widget(str('Acq Modes')), 0, 0, 1, 1)
        layout_acquisition.addWidget(self.QComboBox_acquisition_modes, 1, 0, 1, 1)
        layout_acquisition.addWidget(cw.label_widget(str('Acq Number')), 0, 1, 1, 1)
        layout_acquisition.addWidget(self.QSpinBox_acquisition_number, 1, 1, 1, 1)
        layout_acquisition.addWidget(self.QPushButton_acquire, 1, 2, 1, 1)
        layout_acquisition.addWidget(self.QPushButton_save, 1, 3, 1, 1)
        return layout_acquisition

    def _set_signal_connections(self):
        self.QPushButton_emccd_cooler_check.clicked.connect(self.check_emccd_temperature)
        self.QPushButton_emccd_cooler_switch.clicked.connect(self.switch_emccd_cooler)
        self.QDoubleSpinBox_stage_x.valueChanged.connect(self.set_piezo_x)
        self.QDoubleSpinBox_stage_y.valueChanged.connect(self.set_piezo_y)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.set_piezo_z)
        self.QPushButton_deck_position.clicked.connect(self.read_deck)
        self.QPushButton_deck_position_zero.clicked.connect(self.zero_deck)
        self.QPushButton_move_deck_up.clicked.connect(self.deck_move_up)
        self.QPushButton_move_deck_down.clicked.connect(self.deck_move_down)
        self.QPushButton_move_deck.clicked.connect(self.deck_move_range)
        self.QDoubleSpinBox_galvo_x.valueChanged.connect(self.set_galvo_x)
        self.QDoubleSpinBox_galvo_y.valueChanged.connect(self.set_galvo_y)
        self.QSpinBox_galvo_frequency.valueChanged.connect(self.update_galvo_scan)
        self.QSpinBox_dot_step.valueChanged.connect(self.update_galvo_scan)
        self.QPushButton_laser_488_0.clicked.connect(self.set_laser_488_0)
        self.QPushButton_laser_488_1.clicked.connect(self.set_laser_488_1)
        self.QPushButton_laser_488_2.clicked.connect(self.set_laser_488_2)
        self.QPushButton_laser_405.clicked.connect(self.set_laser_405)
        self.QSpinBox_daq_sample_rate.valueChanged.connect(self.update_daq)
        self.QPushButton_plot_trigger.clicked.connect(self.plot_trigger_sequence)
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_fft.clicked.connect(self.run_fft)
        self.QPushButton_plot_profile.clicked.connect(self.run_plot_profile)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)
        self.QPushButton_save.clicked.connect(self.save)
        self.QComboBox_live_modes.currentIndexChanged[str].connect(self.update_live_parameter_sets)
        self.QComboBox_acquisition_modes.currentIndexChanged[str].connect(self.update_acquisition_parameter_sets)

    @QtCore.pyqtSlot()
    def check_emccd_temperature(self):
        self.Signal_check_emccd_temperature.emit()

    @QtCore.pyqtSlot(bool)
    def switch_emccd_cooler(self, checked: bool):
        self.Signal_switch_emccd_cooler.emit(checked)
        if checked:
            self.QPushButton_emccd_cooler_switch.setText("Cooler ON")
        else:
            self.QPushButton_emccd_cooler_switch.setText("Cooler OFF")

    @QtCore.pyqtSlot(float)
    def set_piezo_x(self, pos_x: float):
        pos_y = self.QDoubleSpinBox_stage_y.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move.emit("x", pos_x, pos_y, pos_z)

    @QtCore.pyqtSlot(float)
    def set_piezo_y(self, pos_y: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move.emit("y", pos_x, pos_y, pos_z)

    @QtCore.pyqtSlot(float)
    def set_piezo_z(self, pos_z: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_y = self.QDoubleSpinBox_stage_y.value()
        self.Signal_piezo_move.emit("z", pos_x, pos_y, pos_z)

    @QtCore.pyqtSlot()
    def read_deck(self):
        self.Signal_deck_read_position.emit()

    @QtCore.pyqtSlot()
    def zero_deck(self):
        self.Signal_deck_zero_position.emit()

    @QtCore.pyqtSlot()
    def deck_move_up(self):
        self.Signal_deck_move_single_step.emit(True)

    @QtCore.pyqtSlot()
    def deck_move_down(self):
        self.Signal_deck_move_single_step.emit(False)

    @QtCore.pyqtSlot(bool)
    def deck_move_range(self, checked: bool):
        distance = self.QSpinBox_deck_direction.value()
        velocity = self.QDoubleSpinBox_deck_velocity.value()
        self.Signal_deck_move_continuous.emit(checked, distance, velocity)

    @QtCore.pyqtSlot(float)
    def set_galvo_x(self, value: float):
        vy = self.QDoubleSpinBox_galvo_y.value()
        self.Signal_galvo_set.emit(value, vy)

    @QtCore.pyqtSlot(float)
    def set_galvo_y(self, value: float):
        vx = self.QDoubleSpinBox_galvo_x.value()
        self.Signal_galvo_set.emit(vx, value)

    @QtCore.pyqtSlot()
    def update_galvo_scan(self):
        self.Signal_galvo_scan_update.emit()

    @QtCore.pyqtSlot(bool)
    def set_laser_488_0(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488_0.value()
        self.Signal_set_laser.emit(["488_0"], checked, power)

    @QtCore.pyqtSlot(bool)
    def set_laser_488_1(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488_1.value()
        self.Signal_set_laser.emit(["488_1"], checked, power)

    @QtCore.pyqtSlot(bool)
    def set_laser_488_2(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488_2.value()
        self.Signal_set_laser.emit(["488_2"], checked, power)

    @QtCore.pyqtSlot(bool)
    def set_laser_405(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_405.value()
        self.Signal_set_laser.emit(["405"], checked, power)

    @QtCore.pyqtSlot(int)
    def update_daq(self, sample_rate: int):
        self.Signal_daq_update.emit(sample_rate)

    @QtCore.pyqtSlot()
    def plot_trigger_sequence(self):
        self.Signal_plot_trigger.emit()

    @QtCore.pyqtSlot()
    def run_video(self):
        vm = self.QComboBox_live_modes.currentText()
        if self.QPushButton_video.isChecked():
            self.Signal_video.emit(True, vm)
            self.QPushButton_fft.setEnabled(True)
            self.QPushButton_plot_profile.setEnabled(True)
        else:
            self.Signal_video.emit(False, vm)
            if self.QPushButton_fft.isChecked():
                self.Signal_fft.emit(False)
            self.QPushButton_fft.setEnabled(False)
            self.QPushButton_fft.setChecked(False)
            if self.QPushButton_plot_profile.isChecked():
                self.Signal_plot_profile.emit(False)
            self.QPushButton_plot_profile.setEnabled(False)
            self.QPushButton_plot_profile.setChecked(False)

    @QtCore.pyqtSlot()
    def run_fft(self):
        if self.QPushButton_fft.isChecked():
            self.Signal_fft.emit(True)
        else:
            self.Signal_fft.emit(False)

    @QtCore.pyqtSlot(bool)
    def run_plot_profile(self, checked: bool):
        self.Signal_plot_profile.emit(checked)

    @QtCore.pyqtSlot()
    def run_acquisition(self):
        acq_mode = self.QComboBox_acquisition_modes.currentText()
        acq_num = self.QSpinBox_acquisition_number.value()
        self.Signal_data_acquire.emit(acq_mode, acq_num)

    @QtCore.pyqtSlot()
    def save(self):
        dialog = cw.create_file_dialogue(name="Save File", file_filter="All Files (*)",
                                         default_dir=str(self.data_folder))
        if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
            selected_file = dialog.selectedFiles()
            if selected_file:
                self.Signal_save_file.emit(selected_file[0])

    @QtCore.pyqtSlot(str)
    def update_live_parameter_sets(self, text: str):
        if text == "Wide Field":
            self.QDoubleSpinBox_range_x.setValue(0.000)
            self.QDoubleSpinBox_range_y.setValue(0.000)
            self.QDoubleSpinBox_range_z.setValue(0.000)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.032)
        if text == "Dot Scan":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.000)
            self.QDoubleSpinBox_range_y.setValue(0.000)
            self.QDoubleSpinBox_range_z.setValue(0.000)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.032)

    @QtCore.pyqtSlot(str)
    def update_acquisition_parameter_sets(self, text: str):
        if text == "Wide Field 2D":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.000)
            self.QDoubleSpinBox_range_y.setValue(0.000)
            self.QDoubleSpinBox_range_z.setValue(0.000)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.032)
        if text == "Wide Field 3D":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.000)
            self.QDoubleSpinBox_range_y.setValue(0.000)
            self.QDoubleSpinBox_range_z.setValue(4.800)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.032)
        if text == "Dot Scan 2D":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.850)
            self.QDoubleSpinBox_range_y.setValue(0.850)
            self.QDoubleSpinBox_range_z.setValue(0.000)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.012)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.064)
        if text == "Dot Scan 3D":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.850)
            self.QDoubleSpinBox_range_y.setValue(0.850)
            self.QDoubleSpinBox_range_z.setValue(3.200)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.012)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.064)
        if text == "Monalisa Scan 2D":
            self.QDoubleSpinBox_step_x.setValue(0.034)
            self.QDoubleSpinBox_step_y.setValue(0.034)
            self.QDoubleSpinBox_step_z.setValue(0.160)
            self.QDoubleSpinBox_range_x.setValue(0.850)
            self.QDoubleSpinBox_range_y.setValue(0.850)
            self.QDoubleSpinBox_range_z.setValue(0.000)
            self.QDoubleSpinBox_ttl_start_on_405.setValue(0.008)
            self.QDoubleSpinBox_ttl_stop_on_405.setValue(0.012)
            self.QDoubleSpinBox_ttl_start_off_488_0.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_0.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_off_488_1.setValue(0.016)
            self.QDoubleSpinBox_ttl_stop_off_488_1.setValue(0.032)
            self.QDoubleSpinBox_ttl_start_read_488_2.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_read_488_2.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_emccd.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_emccd.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_scmos.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_scmos.setValue(0.064)
            self.QDoubleSpinBox_ttl_start_tis.setValue(0.040)
            self.QDoubleSpinBox_ttl_stop_tis.setValue(0.064)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ConWidget(None, None, None)
    window.show()
    sys.exit(app.exec_())
