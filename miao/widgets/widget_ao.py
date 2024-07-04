from PyQt5 import QtWidgets, QtCore

from miao.utilities import customized_widgets as cw


class AOWidget(QtWidgets.QWidget):
    Signal_img_shwfs_base = QtCore.pyqtSignal()
    Signal_img_wfs = QtCore.pyqtSignal(bool)
    Signal_img_shwfr_run = QtCore.pyqtSignal()
    Signal_img_shwfs_compute_wf = QtCore.pyqtSignal()
    Signal_img_shwfs_correct_wf = QtCore.pyqtSignal(int)
    Signal_img_shwfs_save_wf = QtCore.pyqtSignal()
    Signal_img_shwfs_acquisition = QtCore.pyqtSignal()
    Signal_dm_selection = QtCore.pyqtSignal(str)
    Signal_push_actuator = QtCore.pyqtSignal(int, float)
    Signal_influence_function = QtCore.pyqtSignal()
    Signal_set_zernike = QtCore.pyqtSignal()
    Signal_set_dm = QtCore.pyqtSignal()
    Signal_update_cmd = QtCore.pyqtSignal()
    Signal_load_dm = QtCore.pyqtSignal()
    Signal_save_dm = QtCore.pyqtSignal()
    Signal_sensorlessAO_run = QtCore.pyqtSignal()
    Signal_sensorlessAO_save = QtCore.pyqtSignal()

    def __init__(self, config, logg, path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.logg = logg
        self.data_folder = path
        self._setup_ui()
        self._set_signal_connections()
        self._set_initial_values()

    def closeEvent(self, event):
        self.save_spinbox_values()
        event.accept()

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
            "image": cw.create_dock("SH Image"),
            "parameters": cw.create_dock("SH Parameters"),
            "commands": cw.create_dock("Wavefront Sensor"),
            "deformable mirror": cw.create_dock("Deformable Mirror"),
            "dwfs": cw.create_dock("Wavefront Sensing AO"),
            "sensorless ao": cw.create_dock("Sensorless AO")
        }

    def _create_widgets(self):
        self.widgets = {
            "image": self._create_image_widgets(),
            "parameters": self._create_parameters_widgets(),
            "commands": self._create_shwfs_widgets(),
            "deformable mirror": self._create_dm_widgets(),
            "dwfs": self._create_dwfs_widgets(),
            "sensorless ao": self._create_sensorless_widgets()
        }

    def _create_image_widgets(self):
        layout_image = QtWidgets.QHBoxLayout()

        self.lcdNumber_wfmax_img = cw.lcdnumber_widget()
        self.lcdNumber_wfmin_img = cw.lcdnumber_widget()
        self.lcdNumber_wfrms_img = cw.lcdnumber_widget()
        self.image_shwfs_scroll_area, image_shwfs_scroll_layout = cw.create_scroll_area()
        image_shwfs_scroll_layout.addRow(cw.label_widget(str('Wavefront MAX')), self.lcdNumber_wfmax_img)
        image_shwfs_scroll_layout.addRow(cw.label_widget(str('Wavefront MIN')), self.lcdNumber_wfmin_img)
        image_shwfs_scroll_layout.addRow(cw.label_widget(str('Wavefront RMS')), self.lcdNumber_wfrms_img)

        layout_image.addWidget(self.image_shwfs_scroll_area)
        return layout_image

    def _create_parameters_widgets(self):
        layout_parameters = QtWidgets.QHBoxLayout()

        self.QLabel_wfrmd_foc = cw.label_widget(str('Method'))
        self.QComboBox_wfrmd_foc = cw.combobox_widget(list_items=['correlation', 'centerofmass'])
        self.QSpinBox_base_xcenter_foc = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_base_ycenter_foc = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_offset_xcenter_foc = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_offset_ycenter_foc = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_n_lenslets_x_foc = cw.spinbox_widget(0, 64, 1, 14)
        self.QSpinBox_n_lenslets_y_foc = cw.spinbox_widget(0, 64, 1, 14)
        self.QSpinBox_spacing_foc = cw.spinbox_widget(0, 64, 1, 26)
        self.QSpinBox_radius_foc = cw.spinbox_widget(0, 64, 1, 12)
        self.QDoubleSpinBox_foc_background = cw.doublespinbox_widget(0, 1, 0.01, 2, 0.1)
        self.confocal_shwfs_parameters_scroll_area, confocal_shwfs_parameters_scroll_layout = cw.create_scroll_area()
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Illumination')))
        confocal_shwfs_parameters_scroll_layout.addRow(cw.frame_widget())
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Method')), self.QComboBox_wfrmd_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('X_center (Base)')),
                                                       self.QSpinBox_base_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Y_center (Base)')),
                                                       self.QSpinBox_base_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('X_center (Offset)')),
                                                       self.QSpinBox_offset_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Y_center (Offset)')),
                                                       self.QSpinBox_offset_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Lenslet X')),
                                                       self.QSpinBox_n_lenslets_x_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Lenslet Y')),
                                                       self.QSpinBox_n_lenslets_y_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Spacing')), self.QSpinBox_spacing_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Radius')), self.QSpinBox_radius_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Background')),
                                                       self.QDoubleSpinBox_foc_background)

        self.QComboBox_wfrmd_img = cw.combobox_widget(list_items=['correlation', 'centerofmass'])
        self.QSpinBox_base_xcenter_img = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_base_ycenter_img = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_offset_xcenter_img = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_offset_ycenter_img = cw.spinbox_widget(0, 2048, 1, 1024)
        self.QSpinBox_n_lenslets_x_img = cw.spinbox_widget(0, 64, 1, 14)
        self.QSpinBox_n_lenslets_y_img = cw.spinbox_widget(0, 64, 1, 14)
        self.QSpinBox_spacing_img = cw.spinbox_widget(0, 64, 1, 26)
        self.QSpinBox_radius_img = cw.spinbox_widget(0, 64, 1, 12)
        self.QDoubleSpinBox_img_background = cw.doublespinbox_widget(0, 1, 0.01, 2, 0.1)
        self.image_shwfs_parameters_scroll_area, image_shwfs_parameters_scroll_layout = cw.create_scroll_area()
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Imaging')))
        image_shwfs_parameters_scroll_layout.addRow(cw.frame_widget())
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Method')), self.QComboBox_wfrmd_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('X_center (Base)')),
                                                    self.QSpinBox_base_xcenter_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Y_center (Base)')),
                                                    self.QSpinBox_base_ycenter_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('X_center (Offset)')),
                                                    self.QSpinBox_offset_xcenter_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Y_center (Offset)')),
                                                    self.QSpinBox_offset_ycenter_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Lenslet X')), self.QSpinBox_n_lenslets_x_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Lenslet Y')), self.QSpinBox_n_lenslets_y_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Spacing')), self.QSpinBox_spacing_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Radius')), self.QSpinBox_radius_img)
        image_shwfs_parameters_scroll_layout.addRow(cw.label_widget(str('Background')),
                                                    self.QDoubleSpinBox_img_background)

        layout_parameters.addWidget(self.confocal_shwfs_parameters_scroll_area)
        layout_parameters.addWidget(self.image_shwfs_parameters_scroll_area)
        return layout_parameters

    def _create_shwfs_widgets(self):
        layout_shwfs = QtWidgets.QHBoxLayout()

        self.QComboBox_wfs_camera_selection = cw.combobox_widget(list_items=["EMCCD", "SCMOS", "Thorlabs"])
        self.QPushButton_img_shwfs_base = cw.pushbutton_widget('SetBase', enable=True)
        self.QPushButton_run_img_wfs = cw.pushbutton_widget('RunWFS', checkable=True)
        self.QPushButton_run_img_wfr = cw.pushbutton_widget('RunWFR', enable=True)
        self.QPushButton_img_shwfs_compute_wf = cw.pushbutton_widget('ComputeWF', enable=True)
        self.QPushButton_img_shwfs_save_wf = cw.pushbutton_widget('SaveWF', enable=True)
        self.QPushButton_img_shwfs_acquisition = cw.pushbutton_widget('ACQ')
        self.image_shwfs_scroll_area, image_shwfs_scroll_layout = cw.create_scroll_area()
        image_shwfs_scroll_layout.addRow(cw.label_widget(str('Camera')), self.QComboBox_wfs_camera_selection)
        image_shwfs_scroll_layout.addRow(self.QPushButton_run_img_wfs, self.QPushButton_img_shwfs_base)
        image_shwfs_scroll_layout.addRow(self.QPushButton_run_img_wfr, self.QPushButton_img_shwfs_acquisition)
        image_shwfs_scroll_layout.addRow(self.QPushButton_img_shwfs_compute_wf, self.QPushButton_img_shwfs_save_wf)

        layout_shwfs.addWidget(self.image_shwfs_scroll_area)
        return layout_shwfs

    def _create_dm_widgets(self):
        layout_deformablemirror = QtWidgets.QGridLayout()

        self.QComboBox_dms = cw.combobox_widget(list_items=[])
        self.QComboBox_wfsmd = cw.combobox_widget(list_items=['phase', 'zonal', 'modal'])
        self.QSpinBox_actuator = cw.spinbox_widget(0, 96, 1, 0)
        self.QDoubleSpinBox_actuator_push = cw.doublespinbox_widget(-1, 1, 0.005, 3, 0)
        self.QPushButton_push_actuator = cw.pushbutton_widget('Push')
        self.QPushButton_influence_fuction_laser = cw.pushbutton_widget('InfluFunc')
        self.QSpinBox_zernike_mode = cw.spinbox_widget(0, 100, 1, 0)
        self.QDoubleSpinBox_zernike_mode_amp = cw.doublespinbox_widget(-20, 20, 0.01, 2, 0)
        self.QPushButton_set_zernike_mode = cw.pushbutton_widget('Set Zernike')
        self.QComboBox_cmd = cw.combobox_widget(list_items=['0', '1'])
        self.QComboBox_cmd.setCurrentIndex(1)
        self.QPushButton_setDM = cw.pushbutton_widget('Set DM')
        self.QPushButton_load_dm = cw.pushbutton_widget('Load DM')
        self.QPushButton_update_cmd = cw.pushbutton_widget('Add DM')
        self.QPushButton_save_dm = cw.pushbutton_widget('Save DM')

        layout_deformablemirror.addWidget(cw.label_widget(str('DM')), 0, 0, 1, 1)
        layout_deformablemirror.addWidget(self.QComboBox_dms, 0, 1, 1, 1)
        layout_deformablemirror.addWidget(cw.label_widget(str('Method')), 0, 2, 1, 1)
        layout_deformablemirror.addWidget(self.QComboBox_wfsmd, 0, 3, 1, 1)
        layout_deformablemirror.addWidget(cw.label_widget(str('Actuator')), 1, 0, 1, 1)
        layout_deformablemirror.addWidget(self.QSpinBox_actuator, 1, 1, 1, 1)
        layout_deformablemirror.addWidget(cw.label_widget(str('Push')), 2, 0, 1, 1)
        layout_deformablemirror.addWidget(self.QDoubleSpinBox_actuator_push, 2, 1, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_push_actuator, 3, 0, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_influence_fuction_laser, 3, 1, 1, 1)
        layout_deformablemirror.addWidget(cw.label_widget(str('Zernike Mode')), 1, 2, 1, 1)
        layout_deformablemirror.addWidget(self.QSpinBox_zernike_mode, 1, 3, 1, 1)
        layout_deformablemirror.addWidget(cw.label_widget(str('Amplitude')), 2, 2, 1, 1)
        layout_deformablemirror.addWidget(self.QDoubleSpinBox_zernike_mode_amp, 2, 3, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_set_zernike_mode, 3, 2, 1, 1)
        layout_deformablemirror.addWidget(self.QComboBox_cmd, 4, 0, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_setDM, 4, 1, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_load_dm, 3, 3, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_update_cmd, 4, 2, 1, 1)
        layout_deformablemirror.addWidget(self.QPushButton_save_dm, 4, 3, 1, 1)
        return layout_deformablemirror

    def _create_dwfs_widgets(self):
        layout_dwfs = QtWidgets.QGridLayout()

        self.QSpinBox_close_loop_number = cw.spinbox_widget(0, 100, 1, 1)
        self.QPushButton_dwfs_cl_correction = cw.pushbutton_widget('Close Loop Correction')

        layout_dwfs.addWidget(cw.label_widget(str('Loop #   (0 - infinite)')), 0, 0, 1, 1)
        layout_dwfs.addWidget(self.QSpinBox_close_loop_number, 0, 1, 1, 1)
        layout_dwfs.addWidget(self.QPushButton_dwfs_cl_correction, 0, 2, 1, 1)
        return layout_dwfs

    def _create_sensorless_widgets(self):
        layout_sensorless = QtWidgets.QGridLayout()

        self.QSpinBox_zernike_mode_start = cw.spinbox_widget(1, 64, 1, 4)
        self.QSpinBox_zernike_mode_stop = cw.spinbox_widget(1, 64, 1, 10)
        self.QDoubleSpinBox_zernike_mode_amps_start = cw.doublespinbox_widget(-50, 50, 0.005, 3, -0.01)
        self.QSpinBox_zernike_mode_amps_stepnum = cw.spinbox_widget(0, 50, 2, 3)
        self.QDoubleSpinBox_zernike_mode_amps_step = cw.doublespinbox_widget(-50, 50, 0.005, 3, 0.01)
        self.QDoubleSpinBox_lpf = cw.doublespinbox_widget(0, 1, 0.05, 2, 0.1)
        self.QDoubleSpinBox_hpf = cw.doublespinbox_widget(0, 1, 0.05, 2, 0.6)
        self.QComboBox_metric = cw.combobox_widget(list_items=['Max(Intensity)', 'SNR(FFT)', 'HighPass(FFT)'])
        self.QPushButton_sensorless_run = cw.pushbutton_widget('Run AO')
        self.QPushButton_sensorless_save = cw.pushbutton_widget('Save Results')

        layout_sensorless.addWidget(cw.label_widget(str('Zernike Modes')), 0, 0, 1, 2)
        layout_sensorless.addWidget(cw.label_widget(str('From')), 1, 0, 1, 1)
        layout_sensorless.addWidget(self.QSpinBox_zernike_mode_start, 1, 1, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('To')), 2, 0, 1, 1)
        layout_sensorless.addWidget(self.QSpinBox_zernike_mode_stop, 2, 1, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('Amplitudes')), 0, 2, 1, 2)
        layout_sensorless.addWidget(cw.label_widget(str('From')), 1, 2, 1, 1)
        layout_sensorless.addWidget(self.QDoubleSpinBox_zernike_mode_amps_start, 1, 3, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('StepNum')), 2, 2, 1, 1)
        layout_sensorless.addWidget(self.QSpinBox_zernike_mode_amps_stepnum, 2, 3, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('StepSize')), 3, 2, 1, 1)
        layout_sensorless.addWidget(self.QDoubleSpinBox_zernike_mode_amps_step, 3, 3, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('LPF')), 0, 4, 1, 1)
        layout_sensorless.addWidget(self.QDoubleSpinBox_lpf, 1, 4, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('HPF')), 2, 4, 1, 1)
        layout_sensorless.addWidget(self.QDoubleSpinBox_hpf, 3, 4, 1, 1)
        layout_sensorless.addWidget(cw.label_widget(str('Image Metric')), 0, 5, 1, 1)
        layout_sensorless.addWidget(self.QComboBox_metric, 1, 5, 1, 1)
        layout_sensorless.addWidget(self.QPushButton_sensorless_run, 2, 5, 1, 1)
        layout_sensorless.addWidget(self.QPushButton_sensorless_save, 3, 5, 1, 1)
        return layout_sensorless

    def _set_signal_connections(self):
        self.QPushButton_img_shwfs_base.clicked.connect(self.img_wfs_base)
        self.QPushButton_run_img_wfs.clicked.connect(self.run_img_wfs)
        self.QPushButton_run_img_wfr.clicked.connect(self.run_img_wfr)
        self.QPushButton_img_shwfs_compute_wf.clicked.connect(self.compute_img_wf)
        self.QPushButton_img_shwfs_save_wf.clicked.connect(self.save_img_wf)
        self.QPushButton_img_shwfs_acquisition.clicked.connect(self.wfs_acq)
        self.QComboBox_dms.currentIndexChanged.connect(self.select_dm)
        self.QPushButton_push_actuator.clicked.connect(self.push_dm_actuator)
        self.QPushButton_influence_fuction_laser.clicked.connect(self.run_influence_function)
        self.QPushButton_set_zernike_mode.clicked.connect(self.set_dm_zernike)
        self.QPushButton_setDM.clicked.connect(self.set_dm_acts)
        self.QPushButton_update_cmd.clicked.connect(self.update_dm_cmd)
        self.QPushButton_load_dm.clicked.connect(self.load_dm_file)
        self.QPushButton_save_dm.clicked.connect(self.save_dm_cmd)
        self.QPushButton_dwfs_cl_correction.clicked.connect(self.run_close_loop_correction)
        self.QPushButton_sensorless_run.clicked.connect(self.run_sensorless_correction)
        self.QPushButton_sensorless_save.clicked.connect(self.save_sensorless_correction)

    def _set_initial_values(self):
        self.QComboBox_wfs_camera_selection.setCurrentIndex(1)
        self.QSpinBox_base_xcenter_img.setValue(983)
        self.QSpinBox_base_ycenter_img.setValue(1081)
        self.QSpinBox_offset_xcenter_img.setValue(983)
        self.QSpinBox_offset_ycenter_img.setValue(1081)
        self.QSpinBox_n_lenslets_x_img.setValue(19)
        self.QSpinBox_n_lenslets_y_img.setValue(18)
        self.QSpinBox_spacing_img.setValue(61)
        self.QSpinBox_radius_img.setValue(24)
        self.load_spinbox_values()

    @QtCore.pyqtSlot()
    def img_wfs_base(self):
        self.Signal_img_shwfs_base.emit()

    @QtCore.pyqtSlot()
    def run_img_wfs(self):
        if self.QPushButton_run_img_wfs.isChecked():
            self.Signal_img_wfs.emit(True)
        else:
            self.Signal_img_wfs.emit(False)

    @QtCore.pyqtSlot()
    def run_img_wfr(self):
        self.Signal_img_shwfr_run.emit()

    @QtCore.pyqtSlot()
    def compute_img_wf(self):
        self.Signal_img_shwfs_compute_wf.emit()

    @QtCore.pyqtSlot()
    def save_img_wf(self):
        self.Signal_img_shwfs_save_wf.emit()

    @QtCore.pyqtSlot()
    def wfs_acq(self):
        self.Signal_img_shwfs_acquisition.emit()

    @QtCore.pyqtSlot()
    def select_dm(self):
        dn = self.QComboBox_dms.currentText()
        self.Signal_dm_selection.emit(dn)

    @QtCore.pyqtSlot()
    def push_dm_actuator(self):
        n = self.QSpinBox_actuator.value()
        a = self.QDoubleSpinBox_actuator_push.value()
        self.Signal_push_actuator.emit(n, a)

    @QtCore.pyqtSlot()
    def run_influence_function(self):
        self.Signal_influence_function.emit()

    @QtCore.pyqtSlot()
    def set_dm_zernike(self):
        self.Signal_set_zernike.emit()

    @QtCore.pyqtSlot()
    def set_dm_acts(self):
        self.Signal_set_dm.emit()

    @QtCore.pyqtSlot()
    def update_dm_cmd(self):
        self.Signal_update_cmd.emit()

    @QtCore.pyqtSlot()
    def load_dm_file(self):
        self.Signal_load_dm.emit()

    @QtCore.pyqtSlot()
    def save_dm_cmd(self):
        self.Signal_save_dm.emit()

    @QtCore.pyqtSlot()
    def run_close_loop_correction(self):
        n = self.QSpinBox_close_loop_number.value()
        self.Signal_img_shwfs_correct_wf.emit(n)

    @QtCore.pyqtSlot()
    def run_sensorless_correction(self):
        self.Signal_sensorlessAO_run.emit()

    @QtCore.pyqtSlot()
    def save_sensorless_correction(self):
        self.Signal_sensorlessAO_save.emit()

    def save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                values[name] = obj.value()
        self.config.write_config(values, self.config.configs["AOWidget Path"])

    def load_spinbox_values(self):
        try:
            values = self.config.load_config(self.config.configs["AOWidget Path"])
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = AOWidget(None, None, None)
    window.show()
    sys.exit(app.exec_())
