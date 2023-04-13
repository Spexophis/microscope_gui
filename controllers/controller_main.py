import os
import time
from getpass import getuser

import numpy as np
import tifffile as tf
from PyQt5 import QtCore

from controllers import controller_ao
from controllers import controller_con
from controllers import controller_plot
from controllers import controller_view

datapath = r'C:\Users\ruizhe.lin\Documents\data'


class MainController:

    def __init__(self, view, module, process):

        self.view = view
        self.om = module
        self.p = process
        self.con_controller = controller_con.ConController(self.view.getControlWidget())
        self.view_controller = controller_view.ViewController(self.view.getViewWidget())
        self.plot_controller = controller_plot.PlotController(self.view.getPlotWidget())
        self.ao_controller = controller_ao.AOController(self.view.getAOWidget())

        t = time.strftime("%Y%m%d")
        dpth = t + '_' + getuser()
        self.path = os.path.join(datapath, dpth)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        self.stackparams = {'Date/Time': 0, 'User': '', 'X': 0, 'Y': 0, 'Z': 0, 'Xstep': 0, 'Ystep': 0, 'Zstep': 0,
                            'Exposure(s)': 0, 'CCD Temperature': 0, 'Pixel Size(nm)': 13 / (3 * 63), 'CCD setting': ''}

        # video thread    
        self.thread_video = QtCore.QThread()
        self.videoWorker = VideoWorker(parent=None)
        self.videoWorker.moveToThread(self.thread_video)
        self.thread_video.started.connect(self.videoWorker.run)
        self.thread_video.finished.connect(self.videoWorker.stop)
        self.videoWorker.signal_imshow.connect(self.imshow_main)
        # image process thread
        self.thread_impro = QtCore.QThread()
        self.improWorker = ImgProcessWorker(parent=None)
        self.improWorker.moveToThread(self.thread_impro)
        self.thread_impro.started.connect(self.improWorker.run)
        self.thread_impro.finished.connect(self.improWorker.stop)
        self.improWorker.signal_fft.connect(self.imshow_fft)
        # plot thread
        self.thread_plot = QtCore.QThread()
        self.plotWorker = PlotWorker(parent=None)
        self.plotWorker.moveToThread(self.thread_plot)
        self.thread_plot.started.connect(self.plotWorker.run)
        self.thread_plot.finished.connect(self.plotWorker.stop)
        self.plotWorker.signal_plot.connect(self.profile_update)
        # wavefront sensor thread
        self.thread_wfs = QtCore.QThread()
        self.wfsWorker = WFSWorker(parent=None)
        self.wfsWorker.moveToThread(self.thread_wfs)
        self.thread_wfs.started.connect(self.wfsWorker.run)
        self.thread_wfs.finished.connect(self.wfsWorker.stop)
        self.wfsWorker.signal_wfsshow.connect(self.imshow_wfs)
        # Main
        self.view.getControlWidget().Signal_setcoordinates.connect(self.set_camera_coordinates)
        self.view.getControlWidget().Signal_resetcoordinates.connect(self.reset_camera_coordinates)
        self.view.getControlWidget().Signal_piezo_move.connect(self.set_piezo_positions)
        self.view.getControlWidget().Signal_deck_up.connect(self.move_deck_up)
        self.view.getControlWidget().Signal_deck_down.connect(self.move_deck_down)
        self.view.getControlWidget().Signal_deck_move.connect(self.move_deck)
        self.view.getControlWidget().Signal_deck_move_stop.connect(self.move_deck_stop)
        self.view.getControlWidget().Signal_setlaseron_488_0.connect(self.set_laseron_488_0)
        self.view.getControlWidget().Signal_setlaseron_488_1.connect(self.set_laseron_488_1)
        self.view.getControlWidget().Signal_setlaseron_488_2.connect(self.set_laseron_488_2)
        self.view.getControlWidget().Signal_setlaseron_405.connect(self.set_laseron_405)
        self.view.getControlWidget().Signal_setlaseroff_488_0.connect(self.set_laseroff_488_0)
        self.view.getControlWidget().Signal_setlaseroff_488_1.connect(self.set_laseroff_488_1)
        self.view.getControlWidget().Signal_setlaseroff_488_2.connect(self.set_laseroff_488_2)
        self.view.getControlWidget().Signal_setlaseroff_405.connect(self.set_laseroff_405)
        self.view.getControlWidget().Signal_start_video.connect(self.start_video)
        self.view.getControlWidget().Signal_stop_video.connect(self.stop_video)
        self.view.getControlWidget().Signal_run_fft.connect(self.run_fft)
        self.view.getControlWidget().Signal_stop_fft.connect(self.stop_fft)
        self.view.getControlWidget().Signal_3d_resolft.connect(self.record_3d_resolft)
        self.view.getControlWidget().Signal_2d_resolft.connect(self.record_2d_resolft)
        self.view.getControlWidget().Signal_beadscan_2d.connect(self.record_beadscan_2d)
        self.view.getControlWidget().Signal_save_file.connect(self.save_data)
        self.view.getPlotWidget().Signal_plot_static.connect(self.profile_plot)
        self.view.getPlotWidget().Signal_plot_update.connect(self.profile_update)
        # DM
        self.view.getAOWidget().Signal_push_actuator.connect(self.push_actuator)
        self.view.getAOWidget().Signal_influence_function.connect(self.influence_function)
        self.view.getAOWidget().Signal_set_zernike.connect(self.set_zernike)
        self.view.getAOWidget().Signal_set_dm.connect(self.set_dm)
        self.view.getAOWidget().Signal_load_dm.connect(self.load_dm)
        self.view.getAOWidget().Signal_update_cmd.connect(self.update_dm)
        self.view.getAOWidget().Signal_save_dm.connect(self.save_dm)
        # AO
        self.view.getAOWidget().Signal_shwfs_initiate.connect(self.initiate_wfs)
        self.view.getAOWidget().Signal_wfs_start.connect(self.start_wfs)
        self.view.getAOWidget().Signal_wfs_stop.connect(self.stop_wfs)
        self.view.getAOWidget().Signal_shwfs_run.connect(self.run_wfr)
        self.view.getAOWidget().Signal_shwfs_savewf.connect(self.save_wf)
        self.view.getAOWidget().Signal_shwfs_correctwf.connect(self.correct_wf)
        self.view.getAOWidget().Signal_sensorlessAO_run.connect(self.ao_optimize)

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

    def set_piezo_positions(self):
        convFactors = [10, 10, 10.]
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        value_x = pos_x * convFactors[0]
        value_y = pos_y * convFactors[1]
        value_z = pos_z * convFactors[2]
        self.om.daq.set_xyz(value_x, value_y, value_z)

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

    def imshow_main(self):
        if self.om.cam.getImage_live():
            self.view_controller.plot_main(self.om.cam.data)
        else:
            print('No Camera Data')

    def imshow_fft(self):
        self.view_controller.plot_fft(self.p.imgprocess.fourier_transform(self.om.cam.data))

    def profile_plot(self):
        h, v = self.plot_controller.get_plot_axis()
        self.plot_controller.plot_profile(self.om.cam.data, h=h, v=v)

    def profile_update(self):
        h, v = self.plot_controller.get_plot_axis()
        self.plot_controller.updata_plot(self.om.cam.data, h=h, v=v)

    def start_plot_live(self):
        self.thread_plot.start()

    def stop_plot_live(self):
        self.thread_plot.quit()
        self.thread_plot.wait()

    def save_data(self):
        t = time.strftime("%Y%m%d_%H%M%S_")
        slideName = self.con_controller.get_file_name()
        tf.imwrite(self.path + '/' + t + slideName + '.tif', self.om.cam.data)
        self.stackparams['Slide Name'] = slideName
        fnt = self.path + '/' + t + slideName + '_info.txt'
        self._save_text(fnt)
        print('Data saved')

    def _save_text(self, fn=None):
        if fn == None:
            return False
        s = []
        for parts in self.stackparams:
            s.append('%s : %s \n' % (parts, self.stackparams[parts]))
        s.sort()
        fid = open(fn, 'w')
        fid.writelines(s)
        fid.close()

    def stackTags(self, function):
        self.stackparams.clear()
        self.stackparams['00 User'] = getuser()
        self.stackparams['01 Date/Time'] = time.asctime()
        self.stackparams['02 function'] = function
        self.stackparams['03 CCD Temperature'] = self.om.cam.get_ccd_temperature()
        self.stackparams['04 EMCCDGain'] = self.om.cam.get_emccdgain()
        self.stackparams['05 Pixel size'] = 13 / (63 * 2.8)
        self.stackparams['06 Camera Coordinates'] = self.om.cam.G
        # self.stackparams['07 X'] = xx
        # self.stackparams['08 Y'] = yy
        # self.stackparams['09 Z'] = zz
        # self.stackparams['10 Xstep'] = zs
        # self.stackparams['11 Ystep'] = zs
        # self.stackparams['12 Zstep'] = zs

    def set_camera(self):
        self.set_camera_coordinates()
        # expo = self.con_controller.get_exposure_time()
        gain = self.con_controller.get_emccd_gain()
        # self.om.cam.set_exposure(expo)
        self.om.cam.set_emccd_gain(gain)

    def prepare_video(self):
        self.setlasers()
        dgtr = self.generate_digital_trigger_sw()
        self.om.daq.Trig_open(dgtr)
        self.om.cam.set_trigger_mode(7)
        self.set_camera()
        self.om.cam.prepare_live()

    def start_video(self):
        self.prepare_video()
        self.om.cam.start_live()
        self.om.daq.Trig_run()
        time.sleep(0.1)
        self.thread_video.start()

    def stop_video(self):
        self.thread_video.quit()
        self.thread_video.wait()
        self.om.daq.Trig_stop()
        self.lasersoff()
        self.om.cam.stop_live()
        temperature = self.om.cam.get_ccd_temperature()
        self.con_controller.display_camera_temperature(temperature)

    def lasersoff(self):
        self.om.laser.laserOFF_488_0()
        self.om.laser.laserOFF_488_1()
        self.om.laser.laserOFF_488_2()
        self.om.laser.laserOFF_405()

    def run_fft(self):
        self.thread_impro.start()

    def stop_fft(self):
        self.thread_impro.quit()
        self.thread_impro.wait()

    def generate_digital_trigger(self, l):
        sample_rate = 100000
        return_time = 0.001
        convFactors = [10, 10, 10.]
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        dgtr = self.p.trigger.generate_digital_triggers(l)
        return dgtr

    def generate_digital_trigger_sw(self):
        sample_rate = 100000
        return_time = 0.001
        convFactors = [10, 10, 10.]
        l = 1
        l = self.con_controller.select_laser()
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        dgtr = self.p.trigger.generate_digital_triggers_sw(l)
        return dgtr

    def write_trigger_2d(self):
        sample_rate = 100000
        return_time = 0.05
        convFactors = [10, 10, 10.]
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_2d()
        self.om.daq.Trigger_sequence(atr, dtr)

    def write_trigger_3d(self):
        sample_rate = 100000
        return_time = 0.05
        convFactors = [10, 10, 10.]
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_3d()
        self.om.daq.Trigger_sequence(atr, dtr)

    def write_trigger_beadscan_2d(self):
        sample_rate = 100000
        return_time = 0.05
        convFactors = [10, 10, 10.]
        l = 3
        l = self.con_controller.select_laser()
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_beadscan_2d(l)
        self.om.daq.Trigger_sequence(atr, dtr)

    def prepare_resolft_recording(self):
        self.set_camera()
        self.om.cam.set_trigger_mode(7)
        self.om.cam.prepare_kinetic_acquisition(self.npos)
        self.setlasers()

    def setlasers(self):
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

    def record_2d_resolft(self):
        self.write_trigger_2d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.Run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasersoff()

    def record_3d_resolft(self):
        self.write_trigger_3d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.Run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasersoff()

    def record_beadscan_2d(self):
        self.write_trigger_beadscan_2d()
        self.prepare_resolft_recording()
        self.om.cam.start_kinetic_acquisition()
        self.om.daq.Run_sequence()
        self.om.cam.get_data(self.npos)
        print('Acquisition Done')
        self.lasersoff()
        self.reconstruct_beadscan_2d()

    def reconstruct_beadscan_2d(self):
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        stepsize = step_sizes[0]
        self.p.bsrecon.reconstruct_all_beads(self.om.cam.data, stepsize)
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
        self.om.dm.SetDM(values + self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd])

    def set_zernike(self):
        indz, amp = self.ao_controller.get_zernike_mode()
        self.om.dm.SetDM(self.p.shwfsr.get_zernike_cmd(indz, amp) + self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd])

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
        expo = self.ao_controller.get_exposuretime()
        # self.om.tiscam.setPropertyValue('exposure', expo)
        # self.om.thocam.set_exposure(expo)
        # self.om.hacam.setPropertyValue('', )
        self.om.hacam.setPropertyValue('exposure_time', expo)

    def set_wfs(self):
        parameters = self.ao_controller.get_parameters()
        self.p.shwfsr.update_parameters(parameters)
        print('SHWFS parameter updated')

    def initiate_wfs(self):
        self.set_shcam()
        self.set_wfs()
        # self.om.tiscam.prepare_live()
        # self.om.tiscam.start_live()
        # self.p.shwfsr.base = self.om.tiscam.grabFrame()
        # self.p.shwfsr.base = self.om.thocam.snap_image()
        self.om.hacam.startAcquisition()
        time.sleep(0.2)
        self.p.shwfsr.base = self.om.hacam.getFrames(verbose=True, avg=True)
        self.view_controller.plot_sh(self.p.shwfsr.base)
        print('wfs base set')
        # self.om.tiscam.stop_live()
        self.om.hacam.stopAcquisition()

    def imshow_wfs(self):
        # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
        # self.p.shwfsr.offset = self.om.thocam.get_last_image()
        self.p.shwfsr.offset = self.om.hacam.getLastFrame()
        self.view_controller.plot_sh(self.p.shwfsr.offset)

    def start_wfs(self):
        # self.om.tiscam.prepare_live()
        # self.om.tiscam.start_live()
        # self.om.thocam.start_acquire()
        self.om.hacam.startAcquisition()
        time.sleep(0.05)
        self.thread_wfs.start()

    def stop_wfs(self):
        self.thread_wfs.quit()
        self.thread_wfs.wait()
        # self.om.tiscam.stop_live()
        # self.om.thocam.stop_acquire()
        self.om.hacam.stopAcquisition()

    def run_wfr(self):
        self.p.shwfsr.offset = self.om.hacam.getLastFrame()
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
        except ValueError:
            print("NO SH Image")
        try:
            tf.imwrite(self.path + '/' + t + slideName + '_shimg_processed.tif', self.p.shwfsr.im)
        except ValueError:
            print("NO SH Image")
        try:
            tf.imwrite(self.path + '/' + t + slideName + '_reconstruted_wf.tif', self.p.shwfsr.wf)
        except ValueError:
            print("NO WF Image")
        print('WF Data saved')

    def correct_wf(self):
        self.set_shcam()
        self.set_wfs()
        # self.om.tiscam.prepare_live()
        # self.om.tiscam.start_live()
        # self.p.shwfsr.base = self.om.tiscam.grabFrame()
        # self.p.shwfsr.base = self.om.thocam.snap_image()
        self.om.hacam.startAcquisition()
        time.sleep(0.2)
        self.p.shwfsr.get_correction(self.om.hacam.getLastFrame(), self.ao_controller.get_wfs_method())
        # self.om.hacam.getFrames(verbose=True, avg=True)
        self.p.shwfsr.correct_cmd()
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.p.shwfsr.current_cmd = i
        self.run_wfr()
        self.om.hacam.stopAcquisition()

    def influence_function(self):
        t = time.strftime("%Y%m%d_%H%M%S")
        newfold = self.path + '/' + t + '_influence_function' + '/'
        try:
            os.mkdir(newfold)
        except:
            print('Directory already exists')
        n, amp = self.ao_controller.get_acturator()
        self.set_shcam()
        # self.om.tiscam.start_live()
        self.om.hacam.startAcquisition()
        for i in range(self.om.dm.nbAct):
            shimg = []
            print(i)
            values = [0.] * self.om.dm.nbAct
            self.om.dm.SetDM(values)
            time.sleep(0.05)
            # self.p.shwfsr.base = self.om.tiscam.grabFrame()
            # self.p.shwfsr.base = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            values[i] = amp
            self.om.dm.SetDM(values)
            time.sleep(0.05)
            # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
            # self.p.shwfsr.offset = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            values = [0.] * self.om.dm.nbAct
            self.om.dm.SetDM(values)
            time.sleep(0.05)
            # self.p.shwfsr.base = self.om.tiscam.grabFrame()
            # self.p.shwfsr.base = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            values[i] = - amp
            self.om.dm.SetDM(values)
            time.sleep(0.05)
            # self.p.shwfsr.offset = self.om.tiscam.grabFrame()
            # self.p.shwfsr.offset = self.om.thocam.snap_image()
            shimg.append(self.om.hacam.getLastFrame())
            tf.imwrite(newfold + t + '_actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        self.om.hacam.stopAcquisition()
        # self.om.tiscam.stop_live()
        influfunc = self.p.shwfsr.generate_influence_matrix(newfold, self.ao_controller.get_wfs_method())
        ctrlmat = self.p.shwfsr.get_control_matrix(influfunc)
        tf.imwrite(newfold + t + '_control_matrix.tif', ctrlmat)

    def generate_digital_trigger_ao(self):
        sample_rate = 100000
        return_time = 0.001
        convFactors = [10, 10, 10.]
        l = 1
        l = self.con_controller.select_laser()
        sequence_time, axis_lengths, step_sizes, axis_start_pos, analog_start, ttl_starts, ttl_ends = self.con_controller.get_trigger_parameters()
        self.p.trigger.updata_parameters(sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos,
                                         return_time, convFactors, analog_start, ttl_starts, ttl_ends)
        dgtr = self.p.trigger.generate_digital_triggers_ao(l)
        return dgtr

    def start_ao_iteration(self):
        self.setlasers()
        dgtr = self.generate_digital_trigger_ao()
        self.om.daq.Trig_open_ao(dgtr)
        self.om.cam.set_trigger_mode(7)
        self.set_camera()
        self.om.cam.prepare_single_acquisition()

    def stop_ao_iteration(self):
        self.om.daq.Trig_stop()
        self.lasersoff()
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
        dt = []
        amprange = []
        self.p.shwfsr._dm_cmd.append(self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd])
        self.start_ao_iteration()
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
        time.sleep(0.1)
        self.om.cam.single_acquisition()
        self.om.daq.Trig_run()
        time.sleep(0.1)
        self.om.cam.get_acquired_image()
        fn = os.path.join(newfold, 'original.tif')
        tf.imwrite(fn, self.om.cam.data)
        self.om.daq.Trig_stop()
        for mode in range(mode_start, mode_stop):
            for stnm in range(amp_step_number):
                amp = amp_start + stnm * amp_step
                amprange.append(amp)
                self.om.dm.SetDM(
                    self.p.shwfsr.get_zernike_cmd(mode, amp) + self.p.shwfsr._dm_cmd[-1])
                time.sleep(0.1)
                self.om.cam.single_acquisition()
                self.om.daq.Trig_run()
                time.sleep(0.1)
                self.om.cam.get_acquired_image()
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
                self.om.daq.Trig_stop()
            pmax = self.p.imgprocess.peak(amprange, dt)
            if pmax != 0.0:
                self.p.shwfsr.zmv[mode] += pmax
                print('--setting mode %d at value of %.4f--' % (mode, pmax))
                self.p.shwfsr.get_zernike_cmd(mode, pmax) + self.p.shwfsr._dm_cmd[-1]
                self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
            else:
                print('----------------mode %d value equals %.4f----' % (mode, pmax))
        self.om.dm.SetDM(self.p.shwfsr._dm_cmd[-1])
        time.sleep(0.1)
        self.om.cam.single_acquisition()
        self.om.daq.Trig_run()
        time.sleep(0.1)
        self.om.cam.get_acquired_image()
        fn = os.path.join(newfold, 'final.tif')
        tf.imwrite(fn, self.om.cam.data)
        # self.om.dm.writeDMfile(newfold, t, self.p.shwfsr.cmd_best, self.p.shwfsr.mod, self.p.shwfsr.zmv, results)
        self.stop_ao_iteration()


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


class ImgProcessWorker(QtCore.QObject):
    signal_fft = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.timer = None

    def run(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(150)
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
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.signal_plot.emit)
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.stop()
