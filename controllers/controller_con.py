class ConController:

    def __init__(self, view):
        self.v = view

    def get_camera_coordinates(self):
        return self.v.QSpinBox_coordinate_x.value(), self.v.QSpinBox_coordinate_y.value(), \
            self.v.QSpinBox_coordinate_n.value()

    def get_camera_bin(self):
        return self.v.QSpinBox_coordinate_bin.value()

    def get_deck_movement(self):
        return self.v.QDoubleSpinBox_deck_movement.value()

    def get_piezo_positions(self):
        return self.v.QDoubleSpinBox_stage_x.value(), self.v.QDoubleSpinBox_stage_y.value(), \
            self.v.QDoubleSpinBox_stage_z.value()

    def get_exposure_time(self):
        return self.v.QDoubleSpinBox_exposure_time.value()

    def get_galvo_scan(self):
        return self.v.QDoubleSpinBox_galvo_x.value(), self.v.QDoubleSpinBox_galvo_y.value()

    def get_emccd_gain(self):
        return self.v.QSpinBox_emccd_gain.value()

    def get_lasers(self):
        lasers = []
        if self.v.QRadioButton_laser_405.isChecked():
            lasers.append(0)
        if self.v.QRadioButton_laser_488_0.isChecked():
            lasers.append(1)
        if self.v.QRadioButton_laser_488_1.isChecked():
            lasers.append(2)
        if self.v.QRadioButton_laser_488_2.isChecked():
            lasers.append(3)
        return lasers

    def get_cobolt_laser_power(self):
        return self.v.QDoubleSpinBox_laserpower_405.value(), self.v.QDoubleSpinBox_laserpower_488_0.value(), \
            self.v.QDoubleSpinBox_laserpower_488_1.value(), self.v.QDoubleSpinBox_laserpower_488_2.value()

    def get_trigger_parameters(self):
        detection_device = self.v.QComboBox_camera_selection.currentIndex()
        sequence_time = self.v.QDoubleSpinBox_cycle_period.value()
        axis_lengths = [self.v.QDoubleSpinBox_range_x.value(), self.v.QDoubleSpinBox_range_y.value(),
                        self.v.QDoubleSpinBox_range_z.value()]
        step_sizes = [self.v.QDoubleSpinBox_step_x.value(), self.v.QDoubleSpinBox_step_y.value(),
                      self.v.QDoubleSpinBox_step_z.value()]
        axis_start_pos = [self.v.QDoubleSpinBox_start_x.value(), self.v.QDoubleSpinBox_start_y.value(),
                          self.v.QDoubleSpinBox_start_z.value()]
        analog_start = self.v.QDoubleSpinBox_piezo_start.value()
        digital_starts = [self.v.QDoubleSpinBox_ttl_start_on_405.value(),
                          self.v.QDoubleSpinBox_ttl_start_off_488_0.value(),
                          self.v.QDoubleSpinBox_ttl_start_off_488_1.value(),
                          self.v.QDoubleSpinBox_ttl_start_read_488_2.value(),
                          self.v.QDoubleSpinBox_ttl_start_camera_main.value(),
                          self.v.QDoubleSpinBox_ttl_start_camera_wfs.value()]
        digital_ends = [self.v.QDoubleSpinBox_ttl_stop_on_405.value(),
                        self.v.QDoubleSpinBox_ttl_stop_off_488_0.value(),
                        self.v.QDoubleSpinBox_ttl_stop_off_488_1.value(),
                        self.v.QDoubleSpinBox_ttl_stop_read_488_2.value(),
                        self.v.QDoubleSpinBox_ttl_stop_camera_main.value(),
                        self.v.QDoubleSpinBox_ttl_stop_camera_wfs.value()]
        return detection_device, sequence_time, axis_lengths, step_sizes, axis_start_pos, \
            analog_start, digital_starts, digital_ends

    def get_galvo_scan_parameters(self):
        detection_device = self.v.QComboBox_camera_selection.currentIndex()
        galvo_starts = [self.v.QDoubleSpinBox_galvo_start_x.value(), self.v.QDoubleSpinBox_galvo_start_y.value()]
        galvo_stops = [self.v.QDoubleSpinBox_galvo_stop_x.value(), self.v.QDoubleSpinBox_galvo_stop_y.value()]
        galvo_step_sizes = [self.v.QDoubleSpinBox_galvo_step_x.value(), self.v.QDoubleSpinBox_galvo_step_y.value()]
        digital_starts = [self.v.QDoubleSpinBox_ttl_start_on_405.value(),
                          self.v.QDoubleSpinBox_ttl_start_off_488_0.value(),
                          self.v.QDoubleSpinBox_ttl_start_off_488_1.value(),
                          self.v.QDoubleSpinBox_ttl_start_read_488_2.value(),
                          self.v.QDoubleSpinBox_ttl_start_camera_main.value(),
                          self.v.QDoubleSpinBox_ttl_start_camera_wfs.value()]
        digital_ends = [self.v.QDoubleSpinBox_ttl_stop_on_405.value(),
                        self.v.QDoubleSpinBox_ttl_stop_off_488_0.value(),
                        self.v.QDoubleSpinBox_ttl_stop_off_488_1.value(),
                        self.v.QDoubleSpinBox_ttl_stop_read_488_2.value(),
                        self.v.QDoubleSpinBox_ttl_stop_camera_main.value(),
                        self.v.QDoubleSpinBox_ttl_stop_camera_wfs.value()]
        return detection_device, galvo_starts, galvo_stops, galvo_step_sizes, digital_starts, digital_ends

    def get_profile_axis(self):
        return self.v.QComboBox_profile_axis.currentText()

    def display_camera_temperature(self, temperature):
        self.v.QLCDNumber_ccd_tempetature.display(temperature)

    def display_deck_position(self, mdposz):
        self.v.QLCDNumber_deck_position.display(mdposz)

    def display_piezo_position_x(self, ps):
        self.v.QLCDNumber_piezo_position_x.display(ps)

    def display_piezo_position_y(self, ps):
        self.v.QLCDNumber_piezo_position_y.display(ps)

    def display_piezo_position_z(self, ps):
        self.v.QLCDNumber_piezo_position_z.display(ps)

    def get_file_name(self):
        return self.v.QLineEdit_filename.text()
