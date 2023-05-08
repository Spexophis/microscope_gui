import os
import time
from getpass import getuser

import numpy as np
import tifffile as tf
from PyQt5 import QtCore

from controllers import controller_ao, controller_con, controller_view


datapath = r'C:\Users\ruizhe.lin\Documents\data'


class MainController:

    def __init__(self, view, module, process):

        self.view = view
        self.om = module
        self.p = process
        self.view_controller = controller_view.ViewController(self.view.get_view_widget())
        self.con_controller = controller_con.ConController(self.view.get_control_widget())
        self.ao_controller = controller_ao.AOController(self.view.get_ao_widget())

        t = time.strftime("%Y%m%d")
        dpth = t + '_' + getuser()
        self.path = os.path.join(datapath, dpth)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        self.stack_params = {'Date/Time': 0, 'User': '', 'X': 0, 'Y': 0, 'Z': 0, 'Xstep': 0, 'Ystep': 0, 'Zstep': 0,
                             'Exposure(s)': 0, 'CCD Temperature': 0, 'Pixel Size(nm)': 13 / (3 * 63), 'CCD setting': ''}

        # video thread    
        self.thread_video = QtCore.QThread()
        self.videoWorker = VideoWorker(parent=None)
        self.videoWorker.moveToThread(self.thread_video)
        self.thread_video.started.connect(self.videoWorker.run)
        self.thread_video.finished.connect(self.videoWorker.stop)
        self.videoWorker.signal_imshow.connect(self.imshow_main)
        # image process thread
        self.thread_fft = QtCore.QThread()
        self.fftWorker = FFTWorker(parent=None)
        self.fftWorker.moveToThread(self.thread_fft)
        self.thread_fft.started.connect(self.fftWorker.run)
        self.thread_fft.finished.connect(self.fftWorker.stop)
        self.fftWorker.signal_fft.connect(self.imshow_fft)
        # plot thread
        self.thread_plot = QtCore.QThread()
        self.plotWorker = PlotWorker(parent=None)
        self.plotWorker.moveToThread(self.thread_plot)
        self.thread_plot.started.connect(self.plotWorker.run)
        self.thread_plot.finished.connect(self.plotWorker.stop)
        self.plotWorker.signal_plot.connect(self.profile_plot)
        # wavefront sensor thread
        self.thread_wfs = QtCore.QThread()
        self.wfsWorker = WFSWorker(parent=None)
        self.wfsWorker.moveToThread(self.thread_wfs)
        self.thread_wfs.started.connect(self.wfsWorker.run)
        self.thread_wfs.finished.connect(self.wfsWorker.stop)
        self.wfsWorker.signal_wfsshow.connect(self.imshow_wfs)
        # Main
        self.view.get_control_widget().Signal_setcoordinates.connect(self.set_camera_coordinates)
        self.view.get_control_widget().Signal_resetcoordinates.connect(self.reset_camera_coordinates)
        self.view.get_control_widget().Signal_piezo_move_x.connect(self.set_piezo_position_x)
        self.view.get_control_widget().Signal_piezo_move_y.connect(self.set_piezo_position_y)
        self.view.get_control_widget().Signal_piezo_move_z.connect(self.set_piezo_position_z)
        self.view.get_control_widget().Signal_deck_up.connect(self.move_deck_up)
        self.view.get_control_widget().Signal_deck_down.connect(self.move_deck_down)
        self.view.get_control_widget().Signal_deck_move.connect(self.move_deck)
        self.view.get_control_widget().Signal_deck_move_stop.connect(self.move_deck_stop)
        self.view.get_control_widget().Signal_galvo_scan.connect(self.scan_galvo)
        self.view.get_control_widget().Signal_galvo_reset.connect(self.reset_galvo)
        self.view.get_control_widget().Signal_setlaseron_488_0.connect(self.set_laseron_488_0)
        self.view.get_control_widget().Signal_setlaseron_488_1.connect(self.set_laseron_488_1)
        self.view.get_control_widget().Signal_setlaseron_488_2.connect(self.set_laseron_488_2)
        self.view.get_control_widget().Signal_setlaseron_405.connect(self.set_laseron_405)
        self.view.get_control_widget().Signal_setlaseroff_488_0.connect(self.set_laseroff_488_0)
        self.view.get_control_widget().Signal_setlaseroff_488_1.connect(self.set_laseroff_488_1)
        self.view.get_control_widget().Signal_setlaseroff_488_2.connect(self.set_laseroff_488_2)
        self.view.get_control_widget().Signal_setlaseroff_405.connect(self.set_laseroff_405)
        self.view.get_control_widget().Signal_plot_trigger.connect(self.plot_trigger)
        self.view.get_control_widget().Signal_start_video.connect(self.start_video)
        self.view.get_control_widget().Signal_stop_video.connect(self.stop_video)
        self.view.get_control_widget().Signal_run_fft.connect(self.run_fft)
        self.view.get_control_widget().Signal_stop_fft.connect(self.stop_fft)
        self.view.get_control_widget().Signal_run_plot_profile.connect(self.start_plot_live)
        self.view.get_control_widget().Signal_stop_plot_profile.connect(self.stop_plot_live)
        self.view.get_control_widget().Signal_3d_resolft.connect(self.record_3d_resolft)
        self.view.get_control_widget().Signal_2d_resolft.connect(self.record_2d_resolft)
        self.view.get_control_widget().Signal_beadscan_2d.connect(self.record_beadscan_2d)
        self.view.get_control_widget().Signal_save_file.connect(self.save_data)
        # DM
        self.view.get_ao_widget().Signal_push_actuator.connect(self.push_actuator)
        self.view.get_ao_widget().Signal_influence_function.connect(self.influence_function)
        self.view.get_ao_widget().Signal_set_zernike.connect(self.set_zernike)
        self.view.get_ao_widget().Signal_set_dm.connect(self.set_dm)
        self.view.get_ao_widget().Signal_load_dm.connect(self.load_dm)
        self.view.get_ao_widget().Signal_update_cmd.connect(self.update_dm)
        self.view.get_ao_widget().Signal_save_dm.connect(self.save_dm)
        # AO
        self.view.get_ao_widget().Signal_shwfs_initiate.connect(self.set_wfs_base)
        self.view.get_ao_widget().Signal_wfs_start.connect(self.start_wfs)
        self.view.get_ao_widget().Signal_wfs_stop.connect(self.stop_wfs)
        self.view.get_ao_widget().Signal_shwfs_run.connect(self.run_wfr)
        self.view.get_ao_widget().Signal_shwfs_savewf.connect(self.save_wf)
        self.view.get_ao_widget().Signal_shwfs_correctwf.connect(self.correct_wf)
        self.view.get_ao_widget().Signal_sensorlessAO_run.connect(self.ao_optimize)

        temperature = self.om.cam.get_ccd_temperature()
        self.con_controller.display_camera_temperature(temperature)
        p = self.om.md.getPositionStepsTakenAxis(3)
        self.con_controller.display_deck_position(p)
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[1])

    def set_camera_coordinates(self):
        x, y, n = self.con_controller.get_camera_coordinates()
        b = self.con_controller.get_camera_bin()
        self.om.cam.set_image(b, b, x, x + n - 1, y, y + n - 1)

    def reset_camera_coordinates(self):
        self.om.cam.set_image(1, 1, 1, 1024, 1, 1024)

    def move_deck_up(self):
        if not self.om.md.isMoving():
            self.om.md.moveRelativeAxis(3, 0.001524, velocity=1.5)
            self.om.md.wait()
            p = self.om.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def move_deck_down(self):
        if not self.om.md.isMoving():
            self.om.md.moveRelativeAxis(3, -0.001524, velocity=1.5)
            self.om.md.wait()
            p = self.om.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def move_deck(self):
        if not self.om.md.isMoving():
            d = self.con_controller.get_deck_movement()
            self.om.md.moveRelativeAxis(3, d, velocity=self.om.md.velocityMin)

    def move_deck_stop(self):
        if self.om.md.isMoving():
            self.om.md.stopMoving()
            p = self.om.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def set_piezo_position_x(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        self.om.pz.move_position(0, pos_x)
        self.con_controller.display_piezo_position_x(self.om.pz.read_position(0))

    def set_piezo_position_y(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        self.om.pz.move_position(1, pos_y)
        self.con_controller.display_piezo_position_y(self.om.pz.read_position(1))

    def set_piezo_position_z(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        self.om.pz.move_position(2, pos_z)
        self.con_controller.display_piezo_position_z(self.om.pz.read_position(2))

    def reset_galvo(self):
        self.om.daq.set_galvo(0, 0)

    def scan_galvo(self):
        voltx, volty = self.con_controller.get_galvo_scan()
        self.om.daq.set_galvo(voltx, volty)

    def set_laseron_488_0(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.om.laser.constant_power_488_0(p488_0)
        self.om.laser.laserON_488_0()

    def set_laseron_488_1(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.om.laser.constant_power_488_1(p488_1)
        self.om.laser.laserON_488_1()

    def set_laseron_488_2(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.om.laser.constant_power_488_2(p488_2)
        self.om.laser.laserON_488_2()

    def set_laseron_405(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.om.laser.constant_power_405(p405)
        self.om.laser.laserON_405()

    def set_laseroff_488_0(self):
        self.om.laser.laserOFF_488_0()

    def set_laseroff_488_1(self):
        self.om.laser.laserOFF_488_1()

    def set_laseroff_488_2(self):
        self.om.laser.laserOFF_488_2()

    def set_laseroff_405(self):
        self.om.laser.laserOFF_405()

    def set_lasers(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.om.laser.modulation_mode_488_0(0)
        self.om.laser.laserON_488_0()
        self.om.laser.modulation_mode_488_1(0)
        self.om.laser.laserON_488_1()
        self.om.laser.modulation_mode_488_2(0)
        self.om.laser.laserON_488_2()
        self.om.laser.modulation_mode_405(0)
        self.om.laser.laserON_405()
        self.om.laser.modulation_mode_488_1(p488_1)
        self.om.laser.modulation_mode_488_2(p488_2)
        self.om.laser.modulation_mode_488_0(p488_0)
        self.om.laser.modulation_mode_405(p405)

    def lasers_off(self):
        self.om.laser.laserOFF_488_0()
        self.om.laser.laserOFF_488_1()
        self.om.laser.laserOFF_488_2()
        self.om.laser.laserOFF_405()

    def imshow_main(self):
        if self.om.cam.get_image_live():
            self.view_controller.plot_main(self.om.cam.data)
        else:
            print('No Camera Data')

    def imshow_fft(self):
        self.view_controller.plot_fft(self.p.imgprocess.fourier_transform(self.om.cam.data))

    def profile_plot(self):
        ax = self.con_controller.get_profile_axis()
        self.view_controller.plot_update(self.p.imgprocess.get_profile(self.om.cam.data, ax))

    def start_plot_live(self):
        self.thread_plot.start()

    def stop_plot_live(self):
        self.thread_plot.quit()
        self.thread_plot.wait()

    def save_data(self):
        t = time.strftime("%Y%m%d_%H%M%S_")
        slide_name = self.con_controller.get_file_name()
        tf.imwrite(self.path + '/' + t + slide_name + '.tif', self.om.cam.data)
        self.stack_params['Slide Name'] = slide_name
        fnt = self.path + '/' + t + slide_name + '_info.txt'
        self.save_text(fnt)
        print('Data saved')

    def save_text(self, fn=None):
        if fn is None:
            return False
        s = []
        for parts in self.stack_params:
            s.append('%s : %s \n' % (parts, self.stack_params[parts]))
        s.sort()
        fid = open(fn, 'w')
        fid.writelines(s)
        fid.close()

    def stack_tags(self, function):
        self.stack_params.clear()
        self.stack_params['00 User'] = getuser()
        self.stack_params['01 Date/Time'] = time.asctime()
        self.stack_params['02 function'] = function
        self.stack_params['03 CCD Temperature'] = self.om.cam.get_ccd_temperature()
        self.stack_params['04 EMCCDGain'] = self.om.cam.get_emccdgain()
        self.stack_params['05 Pixel size'] = 13 / (63 * 2.8)
        self.stack_params['06 Camera Coordinates'] = self.om.cam.G
        # self.stack_params['07 X'] = xx
        # self.stack_params['08 Y'] = yy
        # self.stack_params['09 Z'] = zz
        # self.stack_params['10 Xstep'] = zs
        # self.stack_params['11 Ystep'] = zs
        # self.stack_params['12 Zstep'] = zs

    def set_camera(self):
        self.set_camera_coordinates()
        # expo = self.con_controller.get_exposure_time()
        gain = self.con_controller.get_emccd_gain()
        # self.om.cam.set_exposure(expo)
        self.om.cam.set_emccd_gain(gain)

    def prepare_video(self):
        self.set_lasers()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.trig_open(dgtr)
        self.om.cam.set_trigger_mode(7)
        self.set_camera()
        self.om.cam.prepare_live()

    def start_video(self):
        self.prepare_video()
        self.om.cam.start_live()
        self.om.daq.trig_run()
        time.sleep(0.1)
        self.thread_video.start()

    def stop_video(self):
        self.thread_video.quit()
        self.thread_video.wait()
        self.om.daq.trig_stop()
        self.lasers_off()
        self.om.cam.stop_live()
        temperature = self.om.cam.get_ccd_temperature()
        self.con_controller.display_camera_temperature(temperature)

    def run_fft(self):
        self.thread_fft.start()

    def stop_fft(self):
        self.thread_fft.quit()
        self.thread_fft.wait()

    def plot_trigger(self):
        sample_rate = 100000
        return_time = 0.001
        conv_factors = [10, 10, 10.]
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.update_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, conv_factors, analog_start, digital_starts, digital_ends)
        dgtr = self.p.trigger.generate_digital_triggers_sw(lasers, camera)
        self.view_controller.plot_update(dgtr[0])
        for i in range(len(digital_starts) - 1):
            self.view_controller.plot(dgtr[i + 1] + i + 1)

    def generate_digital_trigger_sw(self):
        sample_rate = 100000
        return_time = 0.001
        conv_factors = [10, 10, 10.]
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.update_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, conv_factors, analog_start, digital_starts, digital_ends)
        dgtr = self.p.trigger.generate_digital_triggers_sw(lasers, camera)
        return dgtr

    def write_trigger_2d(self):
        sample_rate = 100000
        return_time = 0.05
        conv_factors = [10, 10, 10.]
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.update_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, conv_factors, analog_start, digital_starts, digital_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_2d()
        self.om.daq.trigger_sequence(atr, dtr)

    def write_trigger_3d(self):
        sample_rate = 100000
        return_time = 0.05
        conv_factors = [10, 10, 10.]
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.update_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, conv_factors, analog_start, digital_starts, digital_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_3d()
        self.om.daq.trigger_sequence(atr, dtr)

    def write_trigger_beadscan_2d(self):
        sample_rate = 100000
        return_time = 0.05
        conv_factors = [10, 10, 10.]
        l = self.con_controller.select_laser()
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.update_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, conv_factors, analog_start, digital_starts, digital_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_beadscan_2d(l)
        self.om.daq.trigger_sequence(atr, dtr)

    def prepare_resolft_recording(self):
        self.set_camera()
        self.om.cam.set_trigger_mode(7)
        self.om.cam.prepare_kinetic_acquisition(self.npos)
        self.set_lasers()

    def record_2d_resolft(self):
        self.write_trigger_2d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasers_off()

    def record_3d_resolft(self):
        self.write_trigger_3d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasers_off()

    def record_beadscan_2d(self):
        self.write_trigger_beadscan_2d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasers_off()
        self.reconstruct_beadscan_2d()

    def reconstruct_beadscan_2d(self):
        camera, sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, digital_starts, digital_ends = self.con_controller.get_trigger_parameters()
        step_size = step_sizes[0]
        self.p.bsrecon.reconstruct_all_beads(self.om.cam.data, step_size)
        t = time.strftime("%Y%m%d_%H%M%S_")
        fn = self.con_controller.get_file_name()
        tf.imwrite(self.path + '/' + t + fn + '.tif', self.om.cam.data)
        tf.imwrite(self.path + '/' + t + fn + '_recon_stack.tif', self.p.bsrecon.result)
        tf.imwrite(self.path + '/' + t + fn + '_final_image.tif', self.p.bsrecon.final_image)
        self.view_controller.plot_main(self.p.bsrecon.final_image)
        print('Data saved')

    def push_actuator(self):
        n, a = self.ao_controller.get_acturator()
        values = [0.] * self.om.dm.nbAct
        values[n] = a
        self.om.dm.SetDM(self.p.shwfsr._cmd_add(values, self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))

    def set_zernike(self):
        indz, amp = self.ao_controller.get_zernike_mode()
        # self.om.dm.SetDM(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(indz, amp),
        #                                         self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))
        self.om.dm.SetDM(self.p.shwfsr._cmd_add([i * amp for i in self.om.dm.z2c[indz]],
                                                self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))

    def set_dm(self):
        i = int(self.ao_controller.get_cmd_index())
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[i])
        self.p.shwfsr.current_cmd = i

    def update_dm(self):
        self.p.shwfsr._dm_cmd.append(self.p.shwfsr._temp_cmd[-1])
        self.ao_controller.update_cmd_index()
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])

    def load_dm(self, filename):
        self.p.shwfsr._dm_cmd.append(self.p.shwfsr._read_cmd(filename))
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
        print('New DM cmd loaded')

    def save_dm(self):
        t = time.strftime("%Y%m%d_%H%M%S_")
        self.p.shwfsr._write_cmd(self.path, t, flatfile=False)
        print('DM cmd saved')

    def set_shcam(self):
        self.set_lasers()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.trig_open(dgtr)
        # expo = self.ao_controller.get_exposuretime()
        # self.om.tiscam.setPropertyValue('exposure', expo)
        # self.om.thocam.set_exposure(expo)
        # self.om.hacam.setPropertyValue('exposure_time', expo)

    def set_wfs(self):
        parameters = self.ao_controller.get_parameters()
        self.p.shwfsr.update_parameters(parameters)
        print('SHWFS parameter updated')

    def set_wfs_base(self):
        # self.p.shwfsr.base = self.om.tiscam.grabFrame()
        # self.p.shwfsr.base = self.om.thocam.snap_image()
        self.p.shwfsr.base = self.view_controller.get_image_data('ShackHartmann')
        print('wfs base set')

    def imshow_wfs(self):
        # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
        # self.p.shwfsr.offset = self.om.thocam.get_last_image()
        self.view_controller.plot_sh(self.om.hacam.getLastFrame())

    def start_wfs(self):
        self.set_shcam()
        # self.om.tiscam.prepare_live()
        # self.om.tiscam.start_live()
        # self.om.thocam.start_acquire()
        self.om.hacam.startAcquisition()
        self.om.daq.trig_run()
        time.sleep(0.1)
        self.thread_wfs.start()

    def stop_wfs(self):
        self.thread_wfs.quit()
        self.thread_wfs.wait()
        # self.om.tiscam.stop_live()
        # self.om.thocam.stop_acquire()
        self.om.daq.trig_stop()
        self.om.hacam.stopAcquisition()
        self.lasers_off()

    def run_wfr(self):
        self.p.shwfsr.offset = self.view_controller.get_image_data('ShackHartmann')
        self.p.shwfsr.wavefront_reconstruction(self.p.shwfsr.base, self.p.shwfsr.offset,
                                               self.ao_controller.get_gradient_method())
        self.view_controller.plot_wf(self.p.shwfsr.wf)
        self.ao_controller.display_wf_properties(self.p.imgprocess.wf_properties(self.p.shwfsr.wf))

    def save_wf(self):
        t = time.strftime("%Y%m%d_%H%M%S_")
        slideName = self.ao_controller.get_file_name()
        try:
            # tf.imwrite(self.path + '/' + t + slideName + '_shimg_raw.tif', self.om.tiscam.grabFrame())
            # tf.imwrite(self.path + '/' + t + slideName + '_shimg_raw.tif', self.om.thocam.img)
            tf.imwrite(self.path + '/' + t + slideName + '_shimg_base_raw.tif', self.p.shwfsr.base)
        except:
            print("NO SH Image")
        try:
            tf.imwrite(self.path + '/' + t + slideName + '_shimg_processed.tif', self.p.shwfsr.im)
        except:
            print("NO SH Image")
        try:
            tf.imwrite(self.path + '/' + t + slideName + '_reconstruted_wf.tif', self.p.shwfsr.wf)
        except:
            print("NO WF Image")
        print('WF Data saved')

    def correct_wf(self):
        self.set_wfs()
        self.set_lasers()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.trig_open_ao(dgtr)
        # self.om.tiscam.prepare_live()
        # self.om.tiscam.start_live()
        # self.p.shwfsr.base = self.om.tiscam.grabFrame()
        # self.p.shwfsr.base = self.om.thocam.snap_image()
        self.om.hacam.startAcquisition()
        time.sleep(0.05)
        self.om.daq.trig_run()
        time.sleep(0.05)
        self.p.shwfsr.get_correction(self.om.hacam.getLastFrame(), self.ao_controller.get_wfs_method())
        self.p.shwfsr.correct_cmd()
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.p.shwfsr.current_cmd = i
        self.run_wfr()
        self.om.daq.trig_stop()
        self.om.hacam.stopAcquisition()

    def influence_function(self):
        t = time.strftime("%Y%m%d_%H%M%S")
        newfold = self.path + '/' + t + '_influence_function' + '/'
        try:
            os.mkdir(newfold)
        except:
            print('Directory already exists')
        n, amp = self.ao_controller.get_acturator()
        self.set_wfs()
        self.set_lasers()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.trig_open_ao(dgtr)
        # self.om.tiscam.start_live()
        self.om.hacam.startAcquisition()
        for i in range(self.om.dm.nbAct):
            shimg = []
            print(i)
            values = [0.] * self.om.dm.nbAct
            self.om.dm.SetDM(values)
            time.sleep(0.04)
            self.om.daq.trig_run()
            time.sleep(0.04)
            # self.p.shwfsr.base = self.om.tiscam.grabFrame()
            # self.p.shwfsr.base = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            self.om.daq.trig_stop()
            values[i] = amp
            self.om.dm.SetDM(values)
            time.sleep(0.04)
            self.om.daq.trig_run()
            time.sleep(0.04)
            # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
            # self.p.shwfsr.offset = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            self.om.daq.trig_stop()
            values = [0.] * self.om.dm.nbAct
            self.om.dm.SetDM(values)
            time.sleep(0.04)
            self.om.daq.trig_run()
            time.sleep(0.04)
            # self.p.shwfsr.base = self.om.tiscam.grabFrame()
            # self.p.shwfsr.base = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            self.om.daq.trig_stop()
            values[i] = - amp
            self.om.dm.SetDM(values)
            time.sleep(0.04)
            self.om.daq.trig_run()
            time.sleep(0.04)
            # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
            # self.p.shwfsr.offset = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            self.om.daq.trig_stop()
            tf.imwrite(newfold + t + '_actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        self.om.hacam.stopAcquisition()
        # self.om.tiscam.stop_live()
        influfunc = self.p.shwfsr.generate_influence_matrix(newfold, self.ao_controller.get_wfs_method())
        ctrlmat = self.p.shwfsr.get_control_matrix(influfunc)
        tf.imwrite(newfold + t + '_control_matrix.tif', ctrlmat)

    def start_ao_iteration(self):
        self.set_lasers()
        self.om.cam.set_trigger_mode(7)
        self.set_camera()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.trig_open_ao(dgtr)
        self.om.cam.start_live()

    def stop_ao_iteration(self):
        self.om.daq.trig_stop()
        self.lasers_off()
        self.om.cam.stop_live()
        temperature = self.om.cam.get_ccd_temperature()
        self.con_controller.display_camera_temperature(temperature)

    def ao_optimize(self):
        mode_start, mode_stop, amp_start, amp_step, amp_step_number = self.ao_controller.get_ao_iteration()
        lpr, hpr, mindex, metric = self.ao_controller.get_ao_parameters()
        t = time.strftime("%Y%m%d_%H%M%S_")
        newfold = self.path + '/' + t + '_ao_iteration_' + metric + '/'
        try:
            os.mkdir(newfold)
        except:
            print('Directory already exists')
        results = [('Mode', 'Amp', 'Metric')]
        za = []
        mv = []
        zp = [0] * self.p.shwfsr._n_zernikes
        cmd = self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]
        self.start_ao_iteration()
        self.om.dm.SetDM(cmd)
        time.sleep(0.1)
        self.om.daq.trig_run()
        time.sleep(0.1)
        self.om.cam.get_image_live()
        fn = os.path.join(newfold, 'original.tif')
        tf.imwrite(fn, self.om.cam.data)
        self.om.daq.trig_stop()
        for mode in range(mode_start, mode_stop):
            amprange = []
            dt = []
            for stnm in range(amp_step_number):
                amp = amp_start + stnm * amp_step
                amprange.append(amp)
                # self.om.dm.SetDM(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(mode, amp), cmd))
                self.om.dm.SetDM(self.p.shwfsr._cmd_add([i * amp for i in self.om.dm.z2c[mode]], cmd))
                time.sleep(0.1)
                self.om.daq.trig_run()
                time.sleep(0.1)
                self.om.cam.get_image_live()
                fn = "zm%0.2d_amp%.4f" % (mode, amp)
                fn1 = os.path.join(newfold, fn + '.tif')
                tf.imwrite(fn1, self.om.cam.data)
                if mindex == 0:
                    dt.append(self.p.imgprocess.snr(self.om.cam.data, hpr))
                if mindex == 1:
                    dt.append(self.p.imgprocess.peakv(self.om.cam.data))
                if mindex == 2:
                    dt.append(self.p.imgprocess.hpf(self.om.cam.data, hpr))
                results.append((mode, amp, dt[stnm]))
                print('--', stnm, amp, dt[stnm])
                self.om.daq.trig_stop()
            pmax = self.p.imgprocess.peak(amprange, dt)
            za.extend(amprange)
            mv.extend(dt)
            if pmax != 0.0:
                zp[mode] = pmax
                print('--setting mode %d at value of %.4f--' % (mode, pmax))
                cmd = self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(mode, pmax), cmd)
                self.om.dm.SetDM(cmd)
            else:
                print('----------------mode %d value equals %.4f----' % (mode, pmax))
        self.om.dm.SetDM(cmd)
        time.sleep(0.1)
        self.om.daq.trig_run()
        time.sleep(0.1)
        self.om.cam.get_image_live()
        fn = os.path.join(newfold, 'final.tif')
        tf.imwrite(fn, self.om.cam.data)
        self.stop_ao_iteration()
        self.p.shwfsr._dm_cmd.append(cmd)
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.p.shwfsr.current_cmd = i
        self.p.shwfsr._write_cmd(newfold, '_')
        self.p.shwfsr._save_sensorless_results(os.path.join(newfold, 'results.xlsx'), za, mv, zp)


class VideoWorker(QtCore.QObject):
    signal_imshow = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.signal_imshow.emit)
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.stop()


class FFTWorker(QtCore.QObject):
    signal_fft = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.signal_fft.emit)
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.stop()


class WFSWorker(QtCore.QObject):
    signal_wfsshow = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.signal_wfsshow.emit)
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.stop()


class PlotWorker(QtCore.QObject):
    signal_plot = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.signal_plot.emit)
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.stop()
