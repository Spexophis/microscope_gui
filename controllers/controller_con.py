class ConController():

    def __init__(self, view):
        self.view = view

    def get_camera_coordinates(self):
        return self.view.QSpinBox_coordinate_x.value(), self.view.QSpinBox_coordinate_y.value(), \
            self.view.QSpinBox_coordinate_n.value()

    def get_camera_bin(self):
        return self.view.QSpinBox_coordinate_bin.value()

    def get_deck_movement(self):
        return self.view.QDoubleSpinBox_deck_movement.value()

    def get_piezo_positions(self):
        return self.view.QDoubleSpinBox_stage_x.value(), self.view.QDoubleSpinBox_stage_y.value(), \
            self.view.QDoubleSpinBox_stage_z.value()

    def get_exposure_time(self):
        return self.view.QDoubleSpinBox_exposure_time.value()

    def get_galvo_scan(self):
        return self.view.QDoubleSpinBox_galvo_x.value(), self.view.QDoubleSpinBox_galvo_y.value()

    def get_emccd_gain(self):
        return self.view.QSpinBox_emccd_gain.value()

    def get_cobolt_laser_power(self):
        return self.view.QDoubleSpinBox_laserpower_405.value(), self.view.QDoubleSpinBox_laserpower_488_0.value(), \
            self.view.QDoubleSpinBox_laserpower_488_1.value(), self.view.QDoubleSpinBox_laserpower_488_2.value()

    def get_trigger_parameters(self):
        illumination_source = self.view.QComboBox_laser_selection.currentIndex()
        detection_device = self.view.QComboBox_camera_selection.currentIndex()
        sequence_time = self.view.QDoubleSpinBox_cycle_period.value()
        axis_lengths = [self.view.QDoubleSpinBox_range_x.value(), self.view.QDoubleSpinBox_range_y.value(),
                        self.view.QDoubleSpinBox_range_z.value()]
        step_sizes = [self.view.QDoubleSpinBox_step_x.value(), self.view.QDoubleSpinBox_step_y.value(),
                      self.view.QDoubleSpinBox_step_z.value()]
        axis_start_pos = [self.view.QDoubleSpinBox_start_x.value(), self.view.QDoubleSpinBox_start_y.value(),
                          self.view.QDoubleSpinBox_start_z.value()]
        analog_start = self.view.QDoubleSpinBox_piezo_start.value()
        digital_starts = [self.view.QDoubleSpinBox_ttl_start_on_405.value(),
                          self.view.QDoubleSpinBox_ttl_start_off_488_0.value(),
                          self.view.QDoubleSpinBox_ttl_start_off_488_1.value(),
                          self.view.QDoubleSpinBox_ttl_start_read_488_2.value(),
                          self.view.QDoubleSpinBox_ttl_start_camera_main.value(),
                          self.view.QDoubleSpinBox_ttl_start_camera_wfs.value()]
        digital_ends = [self.view.QDoubleSpinBox_ttl_stop_on_405.value(),
                        self.view.QDoubleSpinBox_ttl_stop_off_488_0.value(),
                        self.view.QDoubleSpinBox_ttl_stop_off_488_1.value(),
                        self.view.QDoubleSpinBox_ttl_stop_read_488_2.value(),
                        self.view.QDoubleSpinBox_ttl_stop_camera_main.value(),
                        self.view.QDoubleSpinBox_ttl_stop_camera_wfs.value()]
        return illumination_source, detection_device, sequence_time, axis_lengths, step_sizes, axis_start_pos, \
            analog_start, digital_starts, digital_ends

    def get_profile_axis(self):
        return self.view.QComboBox_profile_axis.currentText()

    def display_camera_temperature(self, temperature):
        self.view.QLCDNumber_ccd_tempetature.display(temperature)

    def display_deck_position(self, mdposz):
        self.view.QLCDNumber_deck_position.display(mdposz)

    def get_file_name(self):
        return self.view.QLineEdit_filename.text()
