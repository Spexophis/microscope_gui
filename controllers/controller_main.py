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
        self.task_worker = None
        self.task_thread = QtCore.QThread()
        # MCL Piezo
        self.v.get_control_widget().Signal_piezo_move.connect(self.set_piezo)
        # MCL Mad Deck
        self.v.get_control_widget().Signal_deck_move_single_step.connect(self.move_deck_single_step)
        self.v.get_control_widget().Signal_deck_move_continuous.connect(self.move_deck_continuous)
        # Galvo Scanners
        self.v.get_control_widget().Signal_galvo_set.connect(self.set_galvo)
        # Cobolt Lasers
        self.v.get_control_widget().Signal_set_laser.connect(self.set_laser)
        # Main Image Control
        self.v.get_control_widget().Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.v.get_control_widget().Signal_switch_emccd_cooler.connect(self.switch_emdccd_cooler)
        self.v.get_control_widget().Signal_plot_trigger.connect(self.plot_trigger)
        self.v.get_control_widget().Signal_video.connect(self.video)
        self.v.get_control_widget().Signal_fft.connect(self.fft)
        self.v.get_control_widget().Signal_plot_profile.connect(self.plot_live)
        # Main Data Recording
        self.v.get_control_widget().Signal_data_acquire.connect(self.data_acquisition)
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

        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
        self.set_piezo_position_z()

        self.m.dm.set_dm(self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd])

        self.close_loop_thread = None

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

        self.pixel_size_main = self.main_cam.ps / 210
        # self.pixel_size_wfs = self.main_cam.ps / 210

        print("Finish setting up controllers")

    def run_task(self, task, iteration=1, callback=None, parent=None):
        if self.task_worker is not None:
            self.task_worker = None
        self.task_thread = QtCore.QThread()
        self.task_worker = TaskWorker(task=task, n=iteration, callback=callback, parent=parent)
        self.task_worker.moveToThread(self.task_thread)
        self.task_thread.started.connect(self.task_worker.run)
        self.task_worker.signals.finished.connect(self.task_finish)
        self.task_thread.start()
        self.v.get_dialog()

    def task_finish(self):
        self.task_thread.quit()
        self.task_thread.wait()
        self.v.dialog.accept()

    def move_deck_single_step(self, direction):
        if direction:
            self.move_deck_up()
        else:
            self.move_deck_down()

    def move_deck_up(self):
        try:
            _moving = self.m.md.isMoving()
            if _moving:
                print("MadDeck is moving")
            else:
                self.m.md.moveRelativeAxis(3, 0.001524, velocity=1.5)
                self.m.md.wait()
                p = self.m.md.getPositionStepsTakenAxis(3)
                self.con_controller.display_deck_position(p)
        except Exception as e:
            self.logg.error_log.error(f"MadDeck Error: {e}")

    def move_deck_down(self):
        try:
            _moving = self.m.md.isMoving()
            if _moving:
                print("MadDeck is moving")
            else:
                self.m.md.moveRelativeAxis(3, -0.001524, velocity=1.5)
                self.m.md.wait()
                p = self.m.md.getPositionStepsTakenAxis(3)
                self.con_controller.display_deck_position(p)
        except Exception as e:
            self.logg.error_log.error(f"MadDeck Error: {e}")

    def move_deck_continuous(self, moving):
        if moving:
            self.move_deck()
        else:
            self.stop_deck()

    def move_deck(self):
        try:
            _moving = self.m.md.isMoving()
            if _moving:
                self.m.md.stopMoving()
                p = self.m.md.getPositionStepsTakenAxis(3)
                self.con_controller.display_deck_position(p)
                distance, velocity = self.con_controller.get_deck_movement()
                self.m.md.moveRelativeAxis(3, distance, velocity=velocity)
            else:
                distance, velocity = self.con_controller.get_deck_movement()
                self.m.md.moveRelativeAxis(3, distance, velocity=velocity)
        except Exception as e:
            self.logg.error_log.error(f"MadDeck Error: {e}")

    def stop_deck(self):
        try:
            _moving = self.m.md.isMoving()
            if _moving:
                self.m.md.stopMoving()
                p = self.m.md.getPositionStepsTakenAxis(3)
                self.con_controller.display_deck_position(p)
                print("MadDeck is Stopped")
            else:
                print("MadDeck is Stopped")
        except Exception as e:
            self.logg.error_log.error(f"MadDeck Error: {e}")

    def set_piezo(self, axis):
        if axis == "x":
            self.set_piezo_position_x()
        elif axis == "y":
            self.set_piezo_position_y()
        elif axis == "z":
            self.set_piezo_position_z()

    def set_piezo_position_x(self):
        try:
            pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
            self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
            self.con_controller.display_piezo_position_x(self.m.pz.read_position(0))
        except Exception as e:
            self.logg.error_log.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_y(self):
        try:
            pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
            self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
            self.con_controller.display_piezo_position_y(self.m.pz.read_position(1))
        except Exception as e:
            self.logg.error_log.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_z(self):
        try:
            pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
            z = self.m.pz.move_position(2, pos_z)
            self.con_controller.display_piezo_position_z(z)
        except Exception as e:
            self.logg.error_log.error(f"MCL Piezo Error: {e}")

    def set_galvo(self):
        try:
            voltx, volty = self.con_controller.get_galvo_scan()
            self.m.daq.set_galvo_position(voltx, volty)
        except Exception as e:
            self.logg.error_log.error(f"Galvo Error: {e}")

    def set_laser(self, laser, switch):
        if switch:
            try:
                self.m.laser.set_constant_power(laser, self.con_controller.get_cobolt_laser_power(laser))
                self.m.laser.laser_on(laser)
            except Exception as e:
                self.logg.error_log.error(f"Cobolt Laser Error: {e}")
        else:
            try:
                self.m.laser.laser_off(laser)
            except Exception as e:
                self.logg.error_log.error(f"Cobolt Laser Error: {e}")

    def set_lasers(self):
        try:
            self.m.laser.set_modulation_mode(["405", "488_0", "488_1", "488_2"], [0, 0, 0, 0])
            self.m.laser.laser_on("all")
            self.m.laser.set_modulation_mode(["405", "488_0", "488_1", "488_2"], self.con_controller.get_cobolt_laser_power("all"))
        except Exception as e:
            self.logg.error_log.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.m.laser.laser_off("all")
        except Exception as e:
            self.logg.error_log.error(f"Cobolt Laser Error: {e}")

    def set_main_camera_roi(self):
        try:
            if "EMCCD" == self.main_camera:
                x, y, n, b = self.con_controller.get_emccd_roi()
                self.main_cam.bin_h, self.main_cam.bin_h = b, b
                self.main_cam.start_h, self.main_cam.end_h = x, x + n - 1
                self.main_cam.start_v, self.main_cam.end_v = y, y + n - 1
                self.main_cam.set_roi()
                self.main_cam.gain = self.con_controller.get_emccd_gain()
                self.main_cam.set_gain()
            elif "sCMOS" == self.main_camera:
                x, y, n, b = self.con_controller.get_scmos_roi()
                self.main_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
            else:
                print("Invalid Main Camera")
        except Exception as e:
            self.logg.error_log.error(f"Camera Error: {e}")

    def set_wfs_camera_roi(self):
        try:
            if "sCMOS" == self.wfs_camera:
                x, y, n, b = self.con_controller.get_scmos_roi()
                self.wfs_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
            elif "EMCCD" == self.wfs_camera:
                x, y, n, b = self.con_controller.get_emccd_roi()
                self.wfs_cam.set_roi(b, b, x, x + n - 1, y, y + n - 1)
                gain = self.con_controller.get_emccd_gain()
                self.main_cam.set_gain(gain)
            else:
                self.logg.error_log.info("Invalid WFS Camera")
        except Exception as e:
            self.logg.error_log.error(f"Camera Error: {e}")

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
        try:
            self.con_controller.display_camera_temperature(self.m.ccdcam.get_ccd_temperature())
        except Exception as e:
            self.logg.error_log.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler(self, sw):
        if sw:
            self.switch_emdccd_cooler_on()
        else:
            self.switch_emdccd_cooler_off()

    def switch_emdccd_cooler_on(self):
        try:
            self.m.ccdcam.cooler_on()
        except Exception as e:
            self.logg.error_log.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler_off(self):
        try:
            self.m.ccdcam.cooler_off()
        except Exception as e:
            self.logg.error_log.error(f"CCD Camera Error: {e}")

    def update_trigger_parameters(self):
        try:
            lasers = self.con_controller.get_lasers()
            camera = self.con_controller.get_camera()
            sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
            self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
            gv_starts, gv_stops, dotspos = self.con_controller.get_galvo_scan_parameters()
            self.p.trigger.update_galvo_scan_parameters(gv_start=gv_starts[0], gv_stop=gv_stops[0],
                                                        laser_start=dotspos[0],
                                                        laser_interval=dotspos[1])
            axis_lengths, step_sizes, analog_start = self.con_controller.get_piezo_scan_parameters()
            positions = self.con_controller.get_piezo_positions()
            self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, positions)
            self.p.trigger.update_camera_parameters(self.main_cam.t_clean, self.main_cam.t_readout,
                                                    self.main_cam.t_kinetic)
            return lasers, camera
        except Exception as e:
            self.logg.error_log.error(f"Trigger Error: {e}")

    def generate_live_triggers(self):
        lasers, camera = self.update_trigger_parameters()
        return self.p.trigger.generate_digital_triggers(lasers, camera)

    def prepare_video(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            self.m.daq.write_digital_sequences(self.generate_live_triggers(), mode="continuous")
        except Exception as e:
            self.logg.error_log.error(f"Error starting main camera video: {e}")

    def start_video(self):
        try:
            self.prepare_video()
            self.main_cam.start_live()
            self.m.daq.run_digital_trigger()
            self.thread_video.start()
        except Exception as e:
            self.logg.error_log.error(f"Error starting main camera video: {e}")

    def stop_video(self):
        try:
            self.thread_video.quit()
            self.thread_video.wait()
            self.m.daq.stop_triggers()
            self.main_cam.stop_live()
            self.lasers_off()
        except Exception as e:
            self.logg.error_log.error(f"Error stopping main camera video: {e}")

    def video(self, sw):
        if sw:
            self.start_video()
        else:
            self.stop_video()

    def imshow_main(self):
        try:
            self.view_controller.plot_main(self.main_cam.get_last_image())
        except Exception as e:
            self.logg.error_log.error(f"Error showing main camera video: {e}")

    def fft(self, sw):
        if sw:
            self.run_fft()
        else:
            self.stop_fft()

    def run_fft(self):
        try:
            self.thread_fft.start()
        except Exception as e:
            self.logg.error_log.error(f"Error starting fft: {e}")

    def stop_fft(self):
        try:
            self.thread_fft.quit()
            self.thread_fft.wait()
        except Exception as e:
            self.logg.error_log.error(f"Error stopping fft: {e}")

    def imshow_fft(self):
        try:
            self.view_controller.plot_fft(self.p.imgprocess.fourier_transform(self.main_cam.get_last_image()))
        except Exception as e:
            self.logg.error_log.error(f"Error showing fft: {e}")

    def plot_live(self, sw):
        if sw:
            self.start_plot_live()
        else:
            self.stop_plot_live()

    def start_plot_live(self):
        try:
            self.thread_plot.start()
        except Exception as e:
            self.logg.error_log.error(f"Error starting plot: {e}")

    def stop_plot_live(self):
        try:
            self.thread_plot.quit()
            self.thread_plot.wait()
        except Exception as e:
            self.logg.error_log.error(f"Error stopping plot: {e}")

    def profile_plot(self):
        try:
            ax = self.con_controller.get_profile_axis()
            self.view_controller.plot_update(self.p.imgprocess.get_profile(self.main_cam.get_last_image(), ax))
        except Exception as e:
            self.logg.error_log.error(f"Error plotting profile: {e}")

    def plot_trigger(self):
        try:
            dtr = self.generate_live_triggers()
            self.view_controller.plot_update(dtr[0])
            for i in range(dtr.shape[0] - 1):
                self.view_controller.plot(dtr[i + 1] + i + 1)
        except Exception as e:
            self.logg.error_log.error(f"Error plotting digital triggers: {e}")

    def data_acquisition(self):
        acq_mod = self.con_controller.get_acquisition_mode()
        if acq_mod == "Widefield 3D":
            self.run_widefield_zstack()
        if acq_mod == "Confocal 2D":
            self.run_confocal_scanning()
        if acq_mod == "GalvoScan 2D":
            self.run_galvo_scanning()
        if acq_mod == "BeadScan 2D":
            self.run_bead_scan()

    def prepare_widefield_zstack(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            self.m.daq.write_digital_sequences(self.generate_live_triggers(), mode="finite")
        except Exception as e:
            self.logg.error_log.error(f"Error starting widefield zstack: {e}")

    def widefield_zstack(self):
        self.prepare_widefield_zstack()
        try:
            self.main_cam.start_live()
            positions = self.con_controller.get_piezo_positions()
            axis_lengths, step_sizes, analog_start = self.con_controller.get_piezo_scan_parameters()
            num_steps = int(axis_lengths[2] / (2 * step_sizes[2]))
            start = positions[2] - num_steps * step_sizes[2]
            end = positions[2] + num_steps * step_sizes[2]
            zps = np.arange(start, end + step_sizes[2], step_sizes[2])
            print(zps)
            data = []
            for i, z in enumerate(zps):
                pz = self.m.pz.move_position(2, z)
                print(i, pz)
                self.m.daq.run_digital_trigger()
                time.sleep(0.05)
                data.append(self.main_cam.get_last_image())
                self.m.daq.stop_triggers(_close=False)
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_widefield_zstack.tif')
            tf.imwrite(fd, np.asarray(data), imagej=True,
                       resolution=(1 / self.pixel_size_main, 1 / self.pixel_size_main), metadata={'unit': 'um'})
        except Exception as e:
            self.logg.error_log.error(f"Error running widefield zstack: {e}")
        self.finish_widefield_zstack()

    def finish_widefield_zstack(self):
        try:
            self.set_piezo_position_z()
            self.main_cam.stop_live()
            self.lasers_off()
            self.m.daq.stop_triggers()
            self.logg.error_log.info("Widefield image stack acquired")
        except Exception as e:
            self.logg.error_log.error(f"Error stopping widefield zstack: {e}")

    def run_widefield_zstack(self):
        self.run_task(task=self.widefield_zstack)

    def prepare_galvo_scanning(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            lasers, camera = self.update_trigger_parameters()
            atr, dtr, pos = self.p.trigger.generate_galvo_scanning(lasers, camera)
            self.m.daq.write_triggers(piezo_sequence=None, galvo_sequence=atr, digital_sequences=dtr)
        except Exception as e:
            self.logg.error_log.error(f"Error starting galvo scanning: {e}")

    def galvo_scanning(self):
        self.prepare_galvo_scanning()
        try:
            self.main_cam.start_live()
            time.sleep(0.02)
            self.m.daq.run_triggers()
            time.sleep(0.1)
            data = self.main_cam.get_last_image()
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_galvo_scanning.tif')
            tf.imwrite(fd, data, imagej=True, resolution=(1 / self.pixel_size_main, 1 / self.pixel_size_main),
                       metadata={'unit': 'um'})
        except Exception as e:
            self.logg.error_log.error(f"Error running galvo scanning: {e}")
        self.finish_galvo_scanning()

    def finish_galvo_scanning(self):
        try:
            self.m.daq.stop_triggers()
            self.main_cam.stop_live()
            self.lasers_off()
            self.logg.error_log.info("Galvo scanning image acquired")
        except Exception as e:
            self.logg.error_log.error(f"Error stopping galvo scanning: {e}")

    def run_galvo_scanning(self):
        self.run_task(task=self.galvo_scanning)

    def prepare_confocal_scanning(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            lasers, camera = self.update_trigger_parameters()
            atr, dtr, pos = self.p.trigger.generate_confocal_triggers(lasers, camera)
            self.m.daq.write_triggers(piezo_sequence=None, galvo_sequence=atr, digital_sequences=dtr)
        except Exception as e:
            self.logg.error_log.error(f"Error starting confocal scanning: {e}")

    def confocal_scanning(self):
        self.prepare_confocal_scanning()
        try:
            self.main_cam.start_live()
            time.sleep(0.02)
            self.m.daq.run_triggers()
            time.sleep(0.1)
            data = self.main_cam.get_last_image()
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_confocal_scanning.tif')
            tf.imwrite(fd, data, imagej=True, resolution=(1 / self.pixel_size_main, 1 / self.pixel_size_main),
                       metadata={'unit': 'um'})
            # self.view_controller.plot_main(data)
        except Exception as e:
            self.logg.error_log.error(f"Error running confocal scanning: {e}")
        self.finish_confocal_scanning()

    def finish_confocal_scanning(self):
        try:
            self.m.daq.stop_triggers()
            self.main_cam.stop_live()
            self.lasers_off()
            self.logg.error_log.info("Confocal scanning image acquired")
        except Exception as e:
            self.logg.error_log.error(f"Error stopping confocal scanning: {e}")

    def run_confocal_scanning(self):
        self.run_task(task=self.confocal_scanning)

    def prepare_bead_scan(self):
        try:
            self.set_lasers()
            self.set_main_camera_roi()
        except Exception as e:
            self.logg.error_log.error(f"Error starting beads scanning: {e}")

    def bead_scan_2d(self):
        self.prepare_bead_scan()
        try:
            lasers, camera = self.update_trigger_parameters()
            atr, dtr, pos = self.p.trigger.generate_bead_scan_2d(4)
            self.main_cam.prepare_data_acquisition(pos)
            self.main_cam.start_data_acquisition()
            time.sleep(0.02)
            self.m.daq.write_triggers(piezo_sequence=None, galvo_sequence=atr, digital_sequences=dtr)
            self.m.daq.run_triggers()
            time.sleep(0.1)
            data = self.main_cam.get_data(pos)
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_bead_scanning.tif')
            tf.imwrite(fd, data)
            # self.view_controller.plot_main(data)
        except Exception as e:
            self.logg.error_log.error(f"Error running beads scanning: {e}")
        self.finish_bead_scan()

    def finish_bead_scan(self):
        try:
            self.m.daq.stop_triggers()
            self.lasers_off()
            self.logg.error_log.info("Beads scanning image acquired")
        except Exception as e:
            self.logg.error_log.error(f"Error stopping confocal scanning: {e}")

    def run_bead_scan(self):
        self.run_task(task=self.bead_scan_2d)

    def save_data(self, file_name):
        try:
            tf.imwrite(file_name + '.tif', self.main_cam.get_last_image(), imagej=True,
                       resolution=(1 / self.pixel_size_main, 1 / self.pixel_size_main), metadata={'unit': 'um'})
        except Exception as e:
            self.logg.error_log.error(f"Error saving data: {e}")

    def push_actuator(self):
        try:
            n, a = self.ao_controller.get_actuator()
            values = [0.] * self.m.dm.nbAct
            values[n] = a
            self.m.dm.set_dm(self.p.shwfsr._cmd_add(values, self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def set_zernike(self):
        try:
            indz, amp = self.ao_controller.get_zernike_mode()
            self.m.dm.set_dm(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(indz, amp),
                                                    self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))
            # self.m.dm.set_dm(self.p.shwfsr._cmd_add([i * amp for i in self.m.dm.z2c[indz]],
            #                                         self.p.shwfsr._dm_cmd[self.p.shwfsr.current_cmd]))
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def set_dm(self):
        try:
            i = int(self.ao_controller.get_cmd_index())
            self.m.dm.set_dm(self.p.shwfsr._dm_cmd[i])
            self.p.shwfsr.current_cmd = i
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def update_dm(self):
        try:
            self.p.shwfsr._dm_cmd.append(self.p.shwfsr._temp_cmd[-1])
            self.ao_controller.update_cmd_index()
            self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def load_dm(self, filename):
        try:
            self.p.shwfsr._dm_cmd.append(self.p.shwfsr._read_cmd(filename))
            self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])
            print('New DM cmd loaded')
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def save_dm(self):
        try:
            t = time.strftime("%Y%m%d_%H%M%S_")
            self.p.shwfsr._write_cmd(self.data_folder, t, flatfile=False)
            print('DM cmd saved')
        except Exception as e:
            self.logg.error_log.error(f"DM Error: {e}")

    def update_wfs_trigger_parameters(self):
        try:
            lasers = self.con_controller.get_lasers()
            camera = self.con_controller.get_camera()
            sequence_time, digital_starts, digital_ends = self.con_controller.get_digital_parameters()
            self.p.trigger.update_digital_parameters(sequence_time, digital_starts, digital_ends)
            # self.p.trigger.update_camera_parameters(self.wfs_cam.t_clean, self.wfs_cam.t_readout,
            #                                         self.wfs_cam.t_kinetic)
            return lasers, camera
        except Exception as e:
            self.logg.error_log.error(f"Trigger Error: {e}")

    def generate_wfs_trigger(self):
        lasers, camera = self.update_wfs_trigger_parameters()
        return self.p.trigger.generate_wfs_triggers(lasers, camera)

    def set_img_wfs(self):
        try:
            parameters = self.ao_controller.get_parameters_img()
            self.p.shwfsr.update_parameters(parameters)
            print('SHWFS parameter updated')
        except Exception as e:
            self.logg.error_log.error(f"SHWFS Error: {e}")

    def prepare_img_wfs(self):
        try:
            self.set_lasers()
            self.set_img_wfs()
            self.set_wfs_camera_roi()
            self.wfs_cam.prepare_live()
            self.m.daq.write_digital_sequences(self.generate_wfs_trigger(), mode="continuous")
        except Exception as e:
            self.logg.error_log.error(f"Error starting wfs: {e}")

    def start_img_wfs(self):
        try:
            self.prepare_img_wfs()
            self.wfs_cam.start_live()
            self.m.daq.run_digital_trigger()
            self.thread_wfs.start()
        except Exception as e:
            self.logg.error_log.error(f"Error starting wfs: {e}")

    def stop_img_wfs(self):
        try:
            self.thread_wfs.quit()
            self.thread_wfs.wait()
            self.m.daq.stop_triggers()
            self.wfs_cam.stop_live()
            self.lasers_off()
        except Exception as e:
            self.logg.error_log.error(f"Error stopping wfs: {e}")

    def imshow_img_wfs(self):
        try:
            self.p.shwfsr.offset = self.wfs_cam.get_last_image()
            self.view_controller.plot_sh(self.p.shwfsr.offset)
        except Exception as e:
            self.logg.error_log.error(f"Error showing shwfs: {e}")

    def set_img_wfs_base(self):
        try:
            # self.p.shwfsr.base = self.wfs_cam.get_last_image()
            self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann')
            self.view_controller.plot_shb(self.p.shwfsr.base)
            print('wfs base set')
        except Exception as e:
            self.logg.error_log.error(f"SHWFS Error: {e}")

    def run_img_wfr(self):
        self.run_task(task=self.img_wfr)

    def img_wfr(self):
        try:
            self.p.shwfsr.method = self.ao_controller.get_gradient_method_img()
            # self.p.shwfsr.offset = self.wfs_cam.get_last_image()
            self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
            self.p.shwfsr.offset = self.view_controller.get_image_data(r'ShackHartmann')
            self.p.shwfsr.wavefront_reconstruction()
        except Exception as e:
            self.logg.error_log.error(f"SHWFS Reconstruction Error: {e}")

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

    def run_influence_function(self):
        self.run_task(self.influence_function)

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
        self.m.daq.write_digital_sequences(self.generate_wfs_trigger(), mode="finite")
        self.wfs_cam.start_live()
        for i in range(self.m.dm.nbAct):
            shimg = []
            print(i)
            values = [0.] * self.m.dm.nbAct
            self.m.dm.set_dm(values)
            time.sleep(0.04)
            self.m.daq.run_digital_trigger()
            shimg.append(self.wfs_cam.get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values[i] = amp
            self.m.dm.set_dm(values)
            time.sleep(0.04)
            self.m.daq.run_digital_trigger()
            shimg.append(self.wfs_cam.get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values = [0.] * self.m.dm.nbAct
            self.m.dm.set_dm(values)
            time.sleep(0.04)
            self.m.daq.run_digital_trigger()
            shimg.append(self.wfs_cam.get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values[i] = - amp
            self.m.dm.set_dm(values)
            time.sleep(0.04)
            self.m.daq.run_digital_trigger()
            shimg.append(self.wfs_cam.get_last_image())
            self.m.daq.stop_triggers(_close=False)
            tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        self.wfs_cam.stop_live()
        self.m.daq.stop_triggers()
        self.lasers_off()
        md = self.ao_controller.get_img_wfs_method()
        self.p.shwfsr.generate_influence_matrix(fd, md, True)

    def run_close_loop_correction(self, n):
        self.run_task(task=self.close_loop_correction)

    def prepare_close_loop_correction(self):
        try:
            self.set_img_wfs()
            self.set_lasers()
            self.set_wfs_camera_roi()
            self.wfs_cam.prepare_live()
            self.m.daq.write_digital_sequences(self.generate_wfs_trigger(), mode="finite")
            self.wfs_cam.start_live()
        except Exception as e:
            self.logg.error_log.error(f"CloseLoop Correction Error: {e}")

    def close_loop_correction(self):
        self.prepare_close_loop_correction()
        try:
            self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
            self.m.daq.run_digital_trigger()
            self.p.shwfsr.offset = self.wfs_cam.get_last_image()
            self.m.daq.stop_triggers(_close=False)
            self.p.shwfsr.get_correction(self.ao_controller.get_img_wfs_method())
            self.m.dm.set_dm(self.p.shwfsr._dm_cmd[-1])
            self.ao_controller.update_cmd_index()
            i = int(self.ao_controller.get_cmd_index())
            self.p.shwfsr.current_cmd = i
        except Exception as e:
            self.logg.error_log.error(f"CloseLoop Correction Error: {e}")
        self.stop_close_loop_correction()

    def stop_close_loop_correction(self):
        try:
            self.p.shwfsr.base = self.view_controller.get_image_data(r'ShackHartmann(Base)')
            self.m.daq.run_digital_trigger()
            self.p.shwfsr.offset = self.wfs_cam.get_last_image()
            self.wfs_cam.stop_live()
            self.m.daq.stop_triggers()
            self.lasers_off()
            self.run_img_wfr()
            self.view_controller.plot_wf(self.p.shwfsr.wf)
        except Exception as e:
            self.logg.error_log.error(f"CloseLoop Correction Error: {e}")

    def run_ao_optimize(self):
        self.run_task(task=self.ao_optimize)

    def ao_optimize(self):
        try:
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
            self.set_lasers()
            self.set_main_camera_roi()
            self.main_cam.prepare_live()
            self.m.daq.write_digital_sequences(self.generate_live_triggers(), mode="finite")
            self.main_cam.start_live()
            print("Sensorless AO start")
            self.m.dm.set_dm(cmd)
            time.sleep(0.05)
            self.m.daq.run_digital_trigger()
            time.sleep(0.05)
            print(len(self.main_cam.data.data_list))
            fn = os.path.join(new_folder, 'original.tif')
            tf.imwrite(fn, self.main_cam.get_last_image())
            for mode in range(mode_start, mode_stop + 1):
                amprange = []
                dt = []
                for stnm in range(amp_step_number):
                    amp = amp_start + stnm * amp_step
                    amprange.append(amp)
                    self.m.dm.set_dm(self.p.shwfsr._cmd_add(self.p.shwfsr.get_zernike_cmd(mode, amp), cmd))
                    # self.m.dm.set_dm(self.p.shwfsr._cmd_add([i * amp for i in self.m.dm.z2c[mode]], cmd))
                    time.sleep(0.05)
                    self.m.daq.run_digital_trigger()
                    time.sleep(0.05)
                    print(len(self.main_cam.data.data_list))
                    self.m.daq.stop_triggers(_close=False)
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
            self.m.daq.run_digital_trigger()
            time.sleep(0.05)
            self.main_cam.get_last_image()
            fn = os.path.join(new_folder, 'final.tif')
            tf.imwrite(fn, self.main_cam.get_last_image())
            self.p.shwfsr._dm_cmd.append(cmd)
            self.ao_controller.update_cmd_index()
            i = int(self.ao_controller.get_cmd_index())
            self.p.shwfsr.current_cmd = i
            self.p.shwfsr._write_cmd(new_folder, '_')
            self.p.shwfsr._save_sensorless_results(os.path.join(new_folder, 'results.xlsx'), za, mv, zp)
            self.lasers_off()
            self.m.daq.stop_triggers()
            self.main_cam.stop_live()
            print("sensorless AO finished")
        except Exception as e:
            self.logg.error_log.error(f"Sensorless AO Error: {e}")


class TaskWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)


class TaskWorker(QtCore.QObject):
    def __init__(self, task=None, n=1, callback=None, parent=None):
        super().__init__(parent)
        self.task = task if task is not None else self._do_nothing
        self.callback = callback
        self.n = n
        self.signals = TaskWorkerSignals()

    def run(self):
        try:
            for i in range(self.n):
                self._do()
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
            return

    @QtCore.pyqtSlot()
    def _do(self):
        self.task()
        if self.callback is not None:
            self.callback()

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
