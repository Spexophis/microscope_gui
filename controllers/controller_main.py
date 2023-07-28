import os
import time
import traceback

import numpy as np
import tifffile as tf
from PyQt5 import QtCore

from controllers import controller_ao, controller_con, controller_view


class MainController:

    def __init__(self, view, module, process, config, logg, path):

        self.v = view
        self.m = module
        self.p = process
        self.config = config
        self.logg = logg
        self.data_folder = path
        self.view_controller = controller_view.ViewController(self.v.get_view_widget())
        self.con_controller = controller_con.ConController(self.v.get_control_widget())
        self.ao_controller = controller_ao.AOController(self.v.get_ao_widget())

        self.stack_params = None

        # video thread
        self.videoWorker = LoopWorker(dt=100)
        self.videoWorker.signal_loop.connect(self.imshow_main)
        self.thread_video = QtCore.QThread()
        self.videoWorker.moveToThread(self.thread_video)
        self.thread_video.started.connect(self.videoWorker.start)
        self.thread_video.finished.connect(self.videoWorker.stop)
        # image process thread
        self.fftWorker = LoopWorker(dt=200)
        self.fftWorker.signal_loop.connect(self.imshow_fft)
        self.thread_fft = QtCore.QThread()
        self.fftWorker.moveToThread(self.thread_fft)
        self.thread_fft.started.connect(self.fftWorker.start)
        self.thread_fft.finished.connect(self.fftWorker.stop)
        # plot thread
        self.plotWorker = LoopWorker(dt=200)
        self.plotWorker.signal_loop.connect(self.profile_plot)
        self.thread_plot = QtCore.QThread()
        self.plotWorker.moveToThread(self.thread_plot)
        self.thread_plot.started.connect(self.plotWorker.start)
        self.thread_plot.finished.connect(self.plotWorker.stop)
        # wavefront sensor thread
        self.wfsWorker = LoopWorker(dt=100)
        self.wfsWorker.signal_loop.connect(self.imshow_img_wfs)
        self.thread_wfs = QtCore.QThread()
        self.wfsWorker.moveToThread(self.thread_wfs)
        self.thread_wfs.started.connect(self.wfsWorker.start)
        self.thread_wfs.finished.connect(self.wfsWorker.stop)
        # dedicated thread pool for tasks
        self.task_threadpool = QtCore.QThreadPool()
        self.task_worker = None
        # MCL Piezo
        self.v.get_control_widget().Signal_piezo_move_x.connect(self.set_piezo_position_x)
        self.v.get_control_widget().Signal_piezo_move_y.connect(self.set_piezo_position_y)
        self.v.get_control_widget().Signal_piezo_move_z.connect(self.set_piezo_position_z)
        # MCL Mad Deck
        self.v.get_control_widget().Signal_deck_up.connect(self.move_deck_up)
        self.v.get_control_widget().Signal_deck_down.connect(self.move_deck_down)
        self.v.get_control_widget().Signal_deck_move.connect(self.move_deck)
        self.v.get_control_widget().Signal_deck_move_stop.connect(self.move_deck_stop)
        # Galvo Scanners
        self.v.get_control_widget().Signal_galvo_set.connect(self.set_galvo)
        self.v.get_control_widget().Signal_galvo_reset.connect(self.reset_galvo)
        # Cobolt Lasers
        self.v.get_control_widget().Signal_setlaseron_488_0.connect(self.set_laseron_488_0)
        self.v.get_control_widget().Signal_setlaseron_488_1.connect(self.set_laseron_488_1)
        self.v.get_control_widget().Signal_setlaseron_488_2.connect(self.set_laseron_488_2)
        self.v.get_control_widget().Signal_setlaseron_405.connect(self.set_laseron_405)
        self.v.get_control_widget().Signal_setlaseroff_488_0.connect(self.set_laseroff_488_0)
        self.v.get_control_widget().Signal_setlaseroff_488_1.connect(self.set_laseroff_488_1)
        self.v.get_control_widget().Signal_setlaseroff_488_2.connect(self.set_laseroff_488_2)
        self.v.get_control_widget().Signal_setlaseroff_405.connect(self.set_laseroff_405)
        # Main Image Control
        self.v.get_control_widget().Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.v.get_control_widget().Signal_switch_emccd_cooler_on.connect(self.switch_emdccd_cooler_on)
        self.v.get_control_widget().Signal_switch_emccd_cooler_off.connect(self.switch_emdccd_cooler_off)
        self.v.get_control_widget().Signal_plot_trigger.connect(self.plot_trigger)
        self.v.get_control_widget().Signal_start_video.connect(self.start_video)
        self.v.get_control_widget().Signal_stop_video.connect(self.stop_video)
        self.v.get_control_widget().Signal_run_fft.connect(self.run_fft)
        self.v.get_control_widget().Signal_stop_fft.connect(self.stop_fft)
        self.v.get_control_widget().Signal_run_plot_profile.connect(self.start_plot_live)
        self.v.get_control_widget().Signal_stop_plot_profile.connect(self.stop_plot_live)
        # Main Data Recording

        self.v.get_control_widget().Signal_save_file.connect(self.save_data)
        # DM
        self.v.get_ao_widget().Signal_push_actuator.connect(self.push_actuator)
        self.v.get_ao_widget().Signal_set_zernike.connect(self.set_zernike)
        self.v.get_ao_widget().Signal_set_dm.connect(self.set_dm)
        self.v.get_ao_widget().Signal_load_dm.connect(self.load_dm)
        self.v.get_ao_widget().Signal_update_cmd.connect(self.update_dm)
        self.v.get_ao_widget().Signal_save_dm.connect(self.save_dm)
        self.v.get_ao_widget().Signal_influence_function.connect(self.run_influence_function)
        # WFS
        self.v.get_ao_widget().Signal_img_shwfs_initiate.connect(self.set_img_wfs_base)
        self.v.get_ao_widget().Signal_img_wfs_start.connect(self.start_img_wfs)
        self.v.get_ao_widget().Signal_img_wfs_stop.connect(self.stop_img_wfs)
        self.v.get_ao_widget().Signal_img_shwfr_run.connect(self.run_img_wfr)
        self.v.get_ao_widget().Signal_img_shwfs_compute_wf.connect(self.compute_img_wf)
        self.v.get_ao_widget().Signal_img_shwfs_save_wf.connect(self.save_img_wf)
        # AO
        self.v.get_ao_widget().Signal_img_shwfs_correct_wf.connect(self.run_close_loop_correction)
        self.v.get_ao_widget().Signal_sensorlessAO_run.connect(self.run_ao_optimize)

        p = self.m.md.getPositionStepsTakenAxis(3)
        self.con_controller.display_deck_position(p)
        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd])

        self.set_piezo_position_z()

        self.close_loop_thread = None

        # self.main_cam, self.wfs_cam = self.con_controller.get_camera_selections()
        self.main_camera = "EMCCD"
        self.wfs_camera = "sCMOS"
        if "EMCCD" == self.main_camera:
            self.main_cam = self.m.ccdcam
        elif "sCMOS" == self.main_camera:
            self.main_cam = self.m.scmoscam
        if "sCMOS" == self.wfs_camera:
            self.wfs_cam = self.m.scmoscam
        elif "EMCCD" == self.wfs_camera:
            self.wfs_cam = self.m.ccdcam

    def start_task_thread(self, task, callback, iteration):
        self.task_worker = TaskWorker(task, callback, nl=iteration)
        self.task_threadpool.start(self.task_worker)

    def move_deck_up(self):
        if not self.m.md.isMoving():
            self.m.md.moveRelativeAxis(3, 0.001524, velocity=1.5)
            self.m.md.wait()
            p = self.m.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def move_deck_down(self):
        if not self.m.md.isMoving():
            self.m.md.moveRelativeAxis(3, -0.001524, velocity=1.5)
            self.m.md.wait()
            p = self.m.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def move_deck(self):
        if not self.m.md.isMoving():
            d = self.con_controller.get_deck_movement()
            self.m.md.moveRelativeAxis(3, d, velocity=self.m.md.velocityMin)

    def move_deck_stop(self):
        if self.m.md.isMoving():
            self.m.md.stopMoving()
            p = self.m.md.getPositionStepsTakenAxis(3)
            self.con_controller.display_deck_position(p)

    def set_piezo_position_x(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        # x = self.m.pz.move_position(0, pos_x)
        self.con_controller.display_piezo_position_x(self.m.pz.read_position(0))

    def set_piezo_position_y(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        # y = self.m.pz.move_position(1, pos_y)
        self.con_controller.display_piezo_position_y(self.m.pz.read_position(1))

    def set_piezo_position_z(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        z = self.m.pz.move_position(2, pos_z)
        self.con_controller.display_piezo_position_z(z)

    def reset_galvo(self):
        self.m.daq.set_galvo(0, 0)

    def set_galvo(self):
        voltx, volty = self.con_controller.get_galvo_scan()
        self.m.daq.set_galvo(voltx, volty)

    def set_laseron_488_0(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.m.laser.constant_power_488_0(p488_0)
        self.m.laser.laserON_488_0()

    def set_laseron_488_1(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.m.laser.constant_power_488_1(p488_1)
        self.m.laser.laserON_488_1()

    def set_laseron_488_2(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.m.laser.constant_power_488_2(p488_2)
        self.m.laser.laserON_488_2()

    def set_laseron_405(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.m.laser.constant_power_405(p405)
        self.m.laser.laserON_405()

    def set_laseroff_488_0(self):
        self.m.laser.laserOFF_488_0()

    def set_laseroff_488_1(self):
        self.m.laser.laserOFF_488_1()

    def set_laseroff_488_2(self):
        self.m.laser.laserOFF_488_2()

    def set_laseroff_405(self):
        self.m.laser.laserOFF_405()

    def set_lasers(self):
        p405, p488_0, p488_1, p488_2 = self.con_controller.get_cobolt_laser_power()
        self.m.laser.modulation_mode_488_0(0)
        self.m.laser.laserON_488_0()
        self.m.laser.modulation_mode_488_1(0)
        self.m.laser.laserON_488_1()
        self.m.laser.modulation_mode_488_2(0)
        self.m.laser.laserON_488_2()
        self.m.laser.modulation_mode_405(0)
        self.m.laser.laserON_405()
        self.m.laser.modulation_mode_488_1(p488_1)
        self.m.laser.modulation_mode_488_2(p488_2)
        self.m.laser.modulation_mode_488_0(p488_0)
        self.m.laser.modulation_mode_405(p405)

    def lasers_off(self):
        self.m.laser.laserOFF_488_0()
        self.m.laser.laserOFF_488_1()
        self.m.laser.laserOFF_488_2()
        self.m.laser.laserOFF_405()

    def set_main_camera_roi(self):
        if "EMCCD" == self.main_camera:
            x, y, n, b = self.con_controller.get_emccd_roi()
            self.main_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
            gain = self.con_controller.get_emccd_gain()
            self.main_cam.set_gain(gain)
        elif "sCMOS" == self.main_camera:
            x, y, n, b = self.con_controller.get_scmos_roi()
            self.main_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
        else:
            print("Invalid Main Camera")

    def set_wfs_camera_roi(self):
        if "sCMOS" == self.wfs_camera:
            x, y, n, b = self.con_controller.get_scmos_roi()
            self.wfs_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
        elif "EMCCD" == self.wfs_camera:
            x, y, n, b = self.con_controller.get_emccd_roi()
            self.wfs_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
            gain = self.con_controller.get_emccd_gain()
            self.main_cam.set_gain(gain)
        else:
            print("Invalid WFS Camera")

    # def reset_main_camera_roi(self):
    #     if self.main_cam == "EMCCD":
    #         self.main_cam.set_roi(1, 1, 1, 1024, 1, 1024)
    #     if self.main_cam == "sCMOS":
    #         self.main_cam.set_roi(1, 1, 1, 2048, 1, 2048)
    #
    # def reset_wfs_camera_roi(self):
    #     if self.wfs_cam == "sCMOS":
    #         self.wfs_cam.set_roi(1, 1, 1, 2048, 1, 2048)
    #     elif self.wfs_cam == "EMCCD":
    #         self.wfs_cam.set_roi(1, 1, 1, 1024, 1, 1024)

    def check_emdccd_temperature(self):
        self.con_controller.display_camera_temperature(self.m.ccdcam.get_ccd_temperature())

    def switch_emdccd_cooler_on(self):
        self.m.ccdcam.cooler_on()

    def switch_emdccd_cooler_off(self):
        self.m.ccdcam.cooler_off()

    def generate_digital_trigger_sw(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        return self.p.trigger.generate_digital_triggers_sw(lasers, camera)

    def start_video(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            self.m.daq.trig_open(self.generate_digital_trigger_sw())
            self.main_cam.start_live()
            self.m.daq.trig_run()
        except Exception as e:
            self.logg.error(f"Error starting main camera video: {e}")
        try:
            self.thread_video.start()
        except Exception as e:
            self.logg.error(f"Error starting imshow: {e}")

    def stop_video(self):
        try:
            self.thread_video.quit()
            self.thread_video.wait()
        except Exception as e:
            self.logg.error(f"Error stopping imshow: {e}")
        try:
            self.m.daq.trig_stop()
            self.main_cam.stop_live()
            self.lasers_off()
        except Exception as e:
            self.logg.error(f"Error stopping main camera video: {e}")

    def imshow_main(self):
        self.view_controller.plot_main(self.main_cam.get_last_image())

    def run_fft(self):
        try:
            self.thread_fft.start()
        except Exception as e:
            self.logg.error(f"Error starting fft: {e}")

    def stop_fft(self):
        try:
            self.thread_fft.quit()
            self.thread_fft.wait()
        except Exception as e:
            self.logg.error(f"Error stopping fft: {e}")

    def imshow_fft(self):
        self.view_controller.plot_fft(self.p.imgprocess.fourier_transform(self.main_cam.get_last_image()))

    def start_plot_live(self):
        try:
            self.thread_plot.start()
        except Exception as e:
            self.logg.error(f"Error starting plot: {e}")

    def stop_plot_live(self):
        try:
            self.thread_plot.quit()
            self.thread_plot.wait()
        except Exception as e:
            self.logg.error(f"Error stopping plot: {e}")

    def profile_plot(self):
        ax = self.con_controller.get_profile_axis()
        self.view_controller.plot_update(self.p.imgprocess.get_profile(self.main_cam.get_last_image(), ax))

    def plot_trigger(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        dgtr = self.p.trigger.generate_digital_triggers_sw(lasers, camera)
        self.view_controller.plot_update(dgtr[0])
        for i in range(len(digital_starts) - 1):
            self.view_controller.plot(dgtr[i + 1] + i + 1)

    def write_trigger_2d(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        axis_lengths, step_sizes, analog_start = self.con_controller.get_piezo_scan_parameters()
        axis_start_pos
        self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, axis_start_pos, analog_start)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_2d()
        self.m.daq.trigger_sequence(atr, dtr)

    def write_trigger_3d(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        axis_lengths, step_sizes, axis_start_pos, analog_start = self.con_controller.get_piezo_scan_parameters()
        self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, axis_start_pos, analog_start)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_3d()
        self.m.daq.trigger_sequence(atr, dtr)

    def write_trigger_beadscan_2d(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        axis_lengths, step_sizes, axis_start_pos, analog_start = self.con_controller.get_piezo_scan_parameters()
        self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, axis_start_pos, analog_start)
        atr, dtr, self.npos = self.p.trigger.generate_trigger_sequence_beadscan_2d(lasers)
        self.m.daq.trigger_sequence(atr, dtr)

    def prepare_resolft_recording(self):
        self.main_cam.prepare_data_acquisition(self.npos)
        self.set_lasers()

    def record_2d_resolft(self):
        self.write_trigger_2d()
        self.prepare_resolft_recording()
        self.main_cam.start_data_acquisition()
        self.m.daq.run_sequence()
        self.main_cam.get_images(self.npos)
        print('Acquisition Done')
        self.lasers_off()

    def record_3d_resolft(self):
        self.write_trigger_3d()
        self.prepare_resolft_recording()
        self.main_cam.start_data_acquisition()
        self.m.daq.run_sequence()
        self.main_cam.get_images(self.npos)
        print('Acquisition Done')
        self.lasers_off()

    def record_beadscan_2d(self):
        self.write_trigger_beadscan_2d()
        self.prepare_resolft_recording()
        self.main_cam.start_data_acquisition()
        self.m.daq.run_sequence()
        self.main_cam.get_images(self.npos)
        print('Acquisition Done')
        self.lasers_off()
        self.reconstruct_beadscan_2d()

    def reconstruct_beadscan_2d(self):
        axis_lengths, step_sizes, axis_start_pos, analog_start = self.con_controller.get_piezo_scan_parameters()
        step_size = step_sizes[0]
        self.p.bsrecon.reconstruct_all_beads(self.main_cam.data, step_size)
        t = time.strftime("%Y%m%d_%H%M%S_")
        fn = self.con_controller.get_file_name()
        tf.imwrite(self.data_folder + '/' + t + fn + '.tif', self.main_cam.data)
        tf.imwrite(self.data_folder + '/' + t + fn + '_recon_stack.tif', self.p.bsrecon.result)
        tf.imwrite(self.data_folder + '/' + t + fn + '_final_image.tif', self.p.bsrecon.final_image)
        self.view_controller.plot_main(self.p.bsrecon.final_image)
        print('Data saved')

    def write_trigger_gs(self):
        gv_starts, gv_stops, dotspos = self.con_controller.get_galvo_scan_parameters()
        self.p.trigger.update_galvo_scan_parameters(gv_start=gv_starts[0], gv_stop=gv_stops[0], laser_start=dotspos[0],
                                                    laser_interval=dotspos[1])
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        atr, dtr, pos = self.p.trigger.generate_trigger_sequence_gs(lasers, camera)
        # atr, dtr = self.p.trigger.generate_galvo_scanning(lasers, camera)
        self.m.daq.trigger_scan(atr, dtr)

    def record_gs(self):
        self.set_lasers()
        self.main_cam.prepare_live()
        self.write_trigger_gs()
        self.main_cam.start_live()
        time.sleep(0.05)
        self.m.daq.run_scan()
        time.sleep(0.05)
        self.main_cam.stop_live()
        print('Acquisition Done')
        self.imshow_main()
        self.lasers_off()

    def write_triggers(self):
        lasers = self.con_controller.get_lasers()
        camera, sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
        self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
        gv_starts, gv_stops, dotspos = self.con_controller.get_galvo_scan_parameters()
        self.p.trigger.update_galvo_scan_parameters(gv_start=gv_starts[0], gv_stop=gv_stops[0], laser_start=dotspos[0],
                                                    laser_interval=dotspos[1])
        axis_lengths, step_sizes, analog_start = self.con_controller.get_piezo_scan_parameters()
        axis_start_pos
        self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, axis_start_pos, analog_start)

    def save_data(self, file_name):
        tf.imwrite(file_name + '.tif', self.main_cam.get_last_image())
        print('Data saved')

    def push_actuator(self):
        n, a = self.ao_controller.get_actuator()
        values = [0.] * self.m.dm.nbAct
        values[n] = a
        self.m.dm.set_dm(self.p.shwfsr._cmd_add(values, self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))

    def set_zernike(self):
        indz, amp = self.ao_controller.get_zernike_mode()
        self.m.dm.set_dm(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(indz, amp),
                                                self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))
        # self.m.dm.set_dm(self.p.shwfsr._cmd_add([i * amp for i in self.m.dm.z2c[indz]],
        #                                         self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))

    def set_dm(self):
        i = int(self.ao_controller.get_cmd_index())
        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[i])
        self.p.shwfsr.current_cmd = i

    def update_dm(self):
        self.p.shwfsr._dm_cmd.append(self.p.shwfsr._temp_cmd[-1])
        self.ao_controller.update_cmd_index()
        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])

    def load_dm(self, filename):
        self.p.shwfsr._dm_cmd.append(self.p.shwfsr._read_cmd(filename))
        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])
        print('New DM cmd loaded')

    def save_dm(self):
        t = time.strftime("%Y%m%d_%H%M%S_")
        self.p.shwfsr._write_cmd(self.data_folder, t, flatfile=False)
        print('DM cmd saved')

    def run_influence_function(self):
        self.start_task_thread(task=self.influence_function, callback=None, iteration=1)

    def influence_function(self):
        fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M") + '_influence_function')
        try:
            os.makedirs(fd, exist_ok=True)
            print(f'Directory {fd} has been created successfully.')
        except Exception as er:
            print(f'Error creating directory {fd}: {er}')
        n, amp = self.ao_controller.get_actuator()
        self.set_lasers()
        self.set_img_wfs()
        self.set_wfs_camera_roi()
        self.wfs_cam.prepare_live()
        dgtr = self.generate_digital_trigger_sw()
        self.m.daq.trig_open(dgtr)
        self.wfs_cam.start_live()
        self.m.daq.trig_run()
        for i in range(self.m.dm.nbAct):
            shimg = []
            print(i)
            values = [0.] * self.m.dm.nbAct
            self.m.dm.set_dm(values)
            time.sleep(0.1)
            # self.m.daq.trig_run()
            # time.sleep(0.04)
            shimg.append(self.wfs_cam.get_last_image())
            # self.m.daq.trig_stop()
            values[i] = amp
            self.m.dm.set_dm(values)
            time.sleep(0.1)
            # self.m.daq.trig_run()
            # time.sleep(0.04)
            shimg.append(self.wfs_cam.get_last_image())
            # self.m.daq.trig_stop()
            values = [0.] * self.m.dm.nbAct
            self.m.dm.set_dm(values)
            time.sleep(0.1)
            # self.m.daq.trig_run()
            # time.sleep(0.04)
            shimg.append(self.wfs_cam.get_last_image())
            # self.m.daq.trig_stop()
            values[i] = - amp
            self.m.dm.set_dm(values)
            time.sleep(0.1)
            # self.m.daq.trig_run()
            # time.sleep(0.05)
            shimg.append(self.wfs_cam.get_last_image())
            # self.m.daq.trig_stop()
            tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        self.m.daq.trig_stop()
        self.wfs_cam.stop_live()
        self.lasers_off()
        md = self.ao_controller.get_img_wfs_method()
        self.p.shwfsr.generate_influence_matrix(fd, md, True)

    def set_img_wfs(self):
        parameters = self.ao_controller.get_parameters_img()
        self.p.shwfsr.update_parameters(parameters)
        print('SHWFS parameter updated')

    def start_img_wfs(self):
        try:
            self.set_lasers()
            self.set_img_wfs()
            self.set_wfs_camera_roi()
            self.wfs_cam.prepare_live()
            dgtr = self.generate_digital_trigger_sw()
            self.m.daq.trig_open(dgtr)
            self.wfs_cam.start_live()
            self.m.daq.trig_run()
        except Exception as e:
            self.logg.error(f"Error starting wfs: {e}")
        try:

            self.thread_wfs.start()
        except Exception as e:
            self.logg.error(f"Error starting wfs imshow: {e}")

    def stop_img_wfs(self):
        try:
            self.thread_wfs.quit()
            self.thread_wfs.wait()
        except Exception as e:
            self.logg.error(f"Error stopping wfs imshow: {e}")
        try:
            self.m.daq.trig_stop()
            self.wfs_cam.stop_live()
            self.lasers_off()
        except Exception as e:
            self.logg.error(f"Error stopping wfs: {e}")

    def imshow_img_wfs(self):
        self.p.shwfsr.offset = self.wfs_cam.get_last_image()
        self.view_controller.plot_sh(self.p.shwfsr.offset)

    def set_img_wfs_base(self):
        # self.p.shwfsr.base = self.wfs_cam.get_last_image()
        self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann')
        self.view_controller.plot_shb(self.p.shwfsr.base)
        print('wfs base set')

    def run_img_wfr(self):
        self.p.shwfsr.method = self.ao_controller.get_gradient_method_img()
        # self.p.shwfsr.offset = self.wfs_cam.get_last_image()
        self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
        self.p.shwfsr.offset = self.view_controller.get_image_data(r'ShackHartmann')
        self.start_task_thread(task=self.p.shwfsr.wavefront_reconstruction, callback=self.imshow_img_wfr, iteration=1)

    def imshow_img_wfr(self):
        if isinstance(self.p.shwfsr.wf, np.ndarray):
            if self.p.shwfsr.wf.size > 0:
                self.view_controller.plot_wf(self.p.shwfsr.wf)
                self.ao_controller.display_img_wf_properties(self.p.imgprocess.img_properties(self.p.shwfsr.wf))

    def compute_img_wf(self):
        self.p.shwfsr.run_wf_modal_recon()
        self.view_controller.plot_update(self.p.shwfsr._az)
        self.imshow_img_wfr()

    def save_img_wf(self, file_name):
        if isinstance(self.p.shwfsr.base, np.ndarray) and self.p.shwfsr.base.size > 0:
            tf.imwrite(file_name + '_shimg_base_raw.tif', self.p.shwfsr.base)
        if isinstance(self.p.shwfsr.offset, np.ndarray) and self.p.shwfsr.offset.size > 0:
            tf.imwrite(file_name + '_shimg_offset_raw.tif', self.p.shwfsr.offset)
        if isinstance(self.p.shwfsr.im, np.ndarray) and self.p.shwfsr.im.size > 0:
            tf.imwrite(file_name + '_shimg_processed.tif', self.p.shwfsr.im)
        if isinstance(self.p.shwfsr.wf, np.ndarray) and self.p.shwfsr.wf.size > 0:
            tf.imwrite(file_name + '_reconstructed_wf.tif', self.p.shwfsr.wf)
        print('WF Data saved')

    def run_close_loop_correction(self, n):
        self.prepare_close_loop_correction()
        self.start_task_thread(task=self.close_loop_correction, callback=self.stop_close_loop_correction, iteration=n)

    def prepare_close_loop_correction(self):
        self.set_img_wfs()
        self.set_lasers()
        self.set_wfs_camera_roi()
        self.wfs_cam.prepare_live()
        self.wfs_cam.start_live()
        dgtr = self.generate_digital_trigger_sw()
        self.m.daq.trig_open_ao(dgtr)

    def close_loop_correction(self):
        self.m.daq.trig_run_ao()
        self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
        self.p.shwfsr.offset = self.wfs_cam.get_last_image()
        self.p.shwfsr.get_correction(self.ao_controller.get_img_wfs_method())
        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.p.shwfsr.current_cmd = i

    def stop_close_loop_correction(self):
        self.m.daq.trig_run_ao()
        self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
        self.p.shwfsr.offset = self.wfs_cam.get_last_image()
        self.wfs_cam.stop_live()
        self.run_img_wfr()
        self.view_controller.plot_wf(self.p.shwfsr.wf)

    def start_ao_iteration(self):
        self.set_lasers()
        self.set_main_camera_roi()
        self.main_cam.prepare_live()
        dgtr = self.generate_digital_trigger_sw()
        self.m.daq.trig_open_ao(dgtr)
        self.main_cam.start_live()
        print("sensorless AO ready to start")

    def stop_ao_iteration(self):
        # self.m.daq.trig_stop()
        self.lasers_off()
        self.main_cam.stop_live()
        print("sensorless AO finished")

    def run_ao_optimize(self):
        print("run sensorless ao")
        self.start_task_thread(task=self.ao_optimize, callback=None, iteration=1)

    def ao_optimize(self):
        mode_start, mode_stop, amp_start, amp_step, amp_step_number = self.ao_controller.get_ao_iteration()
        lpr, mindex, metric = self.ao_controller.get_ao_parameters()
        name = time.strftime("%Y%m%d_%H%M%S_") + '_ao_iteration_' + metric
        new_folder = self.data_folder / name
        try:
            os.makedirs(new_folder, exist_ok=True)
            print(f'Directory {new_folder} has been created successfully.')
        except Exception as e:
            print(f'Error creating directory {new_folder}: {e}')
        results = [('Mode', 'Amp', 'Metric')]
        za = []
        mv = []
        zp = [0] * self.p.shwfsr._n_zernikes
        cmd = self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]
        self.start_ao_iteration()
        self.m.dm.set_dm(cmd)
        time.sleep(0.05)
        self.m.daq.trig_run_ao()
        time.sleep(0.1)
        fn = os.path.join(new_folder, 'original.tif')
        tf.imwrite(fn, self.main_cam.get_last_image())
        # self.m.daq.trig_stop()
        for mode in range(mode_start, mode_stop + 1):
            amprange = []
            dt = []
            for stnm in range(amp_step_number):
                amp = amp_start + stnm * amp_step
                amprange.append(amp)
                self.m.dm.set_dm(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(mode, amp), cmd))
                # self.m.dm.set_dm(self.p.shwfsr._cmd_add([i * amp for i in self.m.dm.z2c[mode]], cmd))
                time.sleep(0.05)
                self.m.daq.trig_run_ao()
                time.sleep(0.1)
                print(len(self.main_cam.data.data_list))
                fn = "zm%0.2d_amp%.4f" % (mode, amp)
                fn1 = os.path.join(new_folder, fn + '.tif')
                tf.imwrite(fn1, self.main_cam.get_last_image())
                if mindex == 0:
                    dt.append(self.p.imgprocess.snr(self.main_cam.get_last_image(), lpr))
                if mindex == 1:
                    dt.append(self.p.imgprocess.peakv(self.main_cam.get_last_image()))
                if mindex == 2:
                    dt.append(self.p.imgprocess.hpf(self.main_cam.get_last_image(), lpr))
                results.append((mode, amp, dt[stnm]))
                print('--', stnm, amp, dt[stnm])
                # self.m.daq.trig_stop()
            za.extend(amprange)
            mv.extend(dt)
            pmax = self.p.imgprocess.peak(amprange, dt)
            if pmax != 0.0:
                zp[mode] = pmax
                print('--setting mode %d at value of %.4f--' % (mode, pmax))
                cmd = self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(mode, pmax), cmd)
                self.m.dm.set_dm(cmd)
            else:
                print('----------------mode %d value equals %.4f----' % (mode, pmax))
        self.m.dm.set_dm(cmd)
        time.sleep(0.05)
        self.m.daq.trig_run_ao()
        time.sleep(0.1)
        self.main_cam.get_last_image()
        fn = os.path.join(new_folder, 'final.tif')
        tf.imwrite(fn, self.main_cam.get_last_image())
        self.p.shwfsr._dm_cmd.append(cmd)
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.p.shwfsr.current_cmd = i
        self.p.shwfsr._write_cmd(new_folder, '_')
        self.p.shwfsr._save_sensorless_results(os.path.join(new_folder, 'results.xlsx'), za, mv, zp)
        self.stop_ao_iteration()


class TaskWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)


class TaskWorker(QtCore.QRunnable):
    def __init__(self, task=None, callback=None, nl=0):
        super(TaskWorker, self).__init__()
        self.task = task if task is not None else self._do_nothing
        self.callback = callback
        self.nl = nl
        self.stop_requested = False
        self.signals = TaskWorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            if self.nl == 0:
                while not self.stop_requested:
                    self.task()
                    if self.callback is not None:
                        self.callback()
                    self.signals.finished.emit()
            else:
                for i in range(self.nl):
                    if self.stop_requested:
                        break
                    self.task()
                    if self.callback is not None:
                        self.callback()
                    self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
            return

    def stop(self):
        self.stop_requested = True

    @staticmethod
    def _do_nothing():
        pass


class LoopWorker(QtCore.QObject):
    signal_loop = QtCore.pyqtSignal()

    def __init__(self, loop=None, callback=None, dt=0, parent=None):
        super().__init__(parent)
        self.loop = loop if loop is not None else self._do_nothing
        self.callback = callback
        self.dt = dt
        self._stop = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._do)
        if dt > 0:
            self.timer.setInterval(dt)

    def start(self):
        self._stop = False
        if not self.timer.isActive():
            self.timer.start()

    def stop(self):
        self._stop = True
        if self.timer.isActive():
            self.timer.stop()
        if self.callback is not None:
            self.callback()

    @QtCore.pyqtSlot()
    def _do(self):
        if self._stop:
            return
        self.loop()
        self.signal_loop.emit()

    @staticmethod
    def _do_nothing():
        pass
