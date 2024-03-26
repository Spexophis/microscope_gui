import os
import time
import traceback

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt5 import QtCore

from miao.controllers import controller_ao, controller_con, controller_view
from miao.tools import tool_improc as ipr


class MainController(QtCore.QObject):

    def __init__(self, view, module, process, config, logg, path, parent=None):
        super().__init__(parent)

        self.v = view
        self.m = module
        self.p = process
        self.config = config
        self.logg = logg.error_log
        self.data_folder = path
        self.view_controller = controller_view.ViewController(self.v.view_view)
        self.con_controller = controller_con.ConController(self.v.con_view)
        self.ao_controller = controller_ao.AOController(self.v.ao_view)
        self._setup_threads()
        self._set_signal_connections()
        self._initial_setup()
        self.lasers = []
        self.cameras = {"imaging": 0, "wfs": 1}
        # dedicated thread pool for tasks
        self.task_worker = None
        self.task_thread = QtCore.QThread()

    def _setup_threads(self):
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

    def _set_signal_connections(self):
        self.v.view_view.Signal_image_metrics.connect(self.compute_image_metrics)
        # MCL Piezo
        self.v.con_view.Signal_piezo_move.connect(self.set_piezo_positions)
        # MCL Mad Deck
        self.v.con_view.Signal_deck_read_position.connect(self.deck_read_position)
        self.v.con_view.Signal_deck_zero_position.connect(self.deck_zero_position)
        self.v.con_view.Signal_deck_move_single_step.connect(self.move_deck_single_step)
        self.v.con_view.Signal_deck_move_continuous.connect(self.move_deck_continuous)
        # Galvo Scanners
        self.v.con_view.Signal_galvo_set.connect(self.set_galvo)
        self.v.con_view.Signal_galvo_scan_update.connect(self.update_galvo_scanner)
        # Cobolt Lasers
        self.v.con_view.Signal_set_laser.connect(self.set_laser)
        # Main Image Control
        self.v.con_view.Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.v.con_view.Signal_switch_emccd_cooler.connect(self.switch_emdccd_cooler)
        self.v.con_view.Signal_plot_trigger.connect(self.plot_trigger)
        self.v.con_view.Signal_video.connect(self.video)
        self.v.con_view.Signal_fft.connect(self.fft)
        self.v.con_view.Signal_plot_profile.connect(self.plot_live)
        # Main Data Recording
        self.v.con_view.Signal_data_acquire.connect(self.data_acquisition)
        self.v.con_view.Signal_save_file.connect(self.save_data)
        # DM
        self.v.ao_view.Signal_dm_selection.connect(self.select_dm)
        self.v.ao_view.Signal_push_actuator.connect(self.push_actuator)
        self.v.ao_view.Signal_set_zernike.connect(self.set_zernike)
        self.v.ao_view.Signal_set_dm.connect(self.set_dm)
        self.v.ao_view.Signal_load_dm.connect(self.load_dm)
        self.v.ao_view.Signal_update_cmd.connect(self.update_dm)
        self.v.ao_view.Signal_save_dm.connect(self.save_dm)
        self.v.ao_view.Signal_influence_function.connect(self.run_influence_function)
        # WFS
        self.v.ao_view.Signal_img_shwfs_base.connect(self.set_reference_wf)
        self.v.ao_view.Signal_img_wfs.connect(self.img_wfs)
        self.v.ao_view.Signal_img_shwfr_run.connect(self.run_img_wfr)
        self.v.ao_view.Signal_img_shwfs_compute_wf.connect(self.compute_img_wf)
        self.v.ao_view.Signal_img_shwfs_save_wf.connect(self.save_img_wf)
        self.v.ao_view.Signal_img_shwfs_acquisition.connect(self.run_shwfs_acquisition)
        # AO
        self.v.ao_view.Signal_img_shwfs_correct_wf.connect(self.run_close_loop_correction)
        self.v.ao_view.Signal_sensorlessAO_run.connect(self.run_sensorless_iteration)

    def _initial_setup(self):
        try:
            p = self.m.md.get_position_steps_taken(3)
            self.con_controller.display_deck_position(p)

            self.reset_piezo_positions()

            self.update_galvo_scanner()

            self.magnifications = [157.5, 1, 1]
            self.pixel_sizes = []
            self.pixel_sizes = [self.m.cam_set[i].ps / mag for i, mag in enumerate(self.magnifications)]

            for key in self.m.dm.keys():
                self.v.ao_view.QComboBox_dms.addItem(key)
            self.dfm = self.m.dm[self.v.ao_view.QComboBox_dms.currentText()]
            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    def run_task(self, task, iteration=1, parent=None, callback=None):
        if self.task_worker is not None:
            self.task_worker = None
        self.task_thread = QtCore.QThread()
        self.task_worker = TaskWorker(task=task, n=iteration, parent=parent)
        self.task_worker.moveToThread(self.task_thread)
        self.task_thread.started.connect(self.task_worker.run)
        self.task_worker.signals.finished.connect(self.task_finish)
        if callback is not None:
            self.task_worker.signals.finished.connect(callback)
        self.task_thread.start()
        self.v.get_dialog()

    def task_finish(self):
        self.task_thread.quit()
        self.task_thread.wait()
        self.v.dialog.accept()

    @QtCore.pyqtSlot()
    def deck_read_position(self):
        self.con_controller.display_deck_position(self.m.md.position)

    @QtCore.pyqtSlot()
    def deck_zero_position(self):
        self.m.md.position = 0
        self.con_controller.display_deck_position(self.m.md.position)

    @QtCore.pyqtSlot(bool)
    def move_deck_single_step(self, direction: bool):
        if direction:
            self.move_deck_up()
        else:
            self.move_deck_down()

    def move_deck_up(self):
        try:
            _moving = self.m.md.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.m.md.move_relative(3, 0.000762, velocity=0.8)
                self.con_controller.display_deck_position(self.m.md.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    def move_deck_down(self):
        try:
            _moving = self.m.md.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.m.md.move_relative(3, -0.000762, velocity=0.8)
                self.con_controller.display_deck_position(self.m.md.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    @QtCore.pyqtSlot(bool, int, float)
    def move_deck_continuous(self, moving: bool, direction: int, velocity: float):
        if moving:
            self.m.md.move_deck(direction, velocity)
        else:
            self.m.md.stop_deck()

    def reset_piezo_positions(self):
        pos_x, pos_y, pos_z = self.con_controller.get_piezo_positions()
        self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
        self.set_piezo_position_z(pos_z)

    @QtCore.pyqtSlot(str, float, float, float)
    def set_piezo_positions(self, axis: str, value_x: float, value_y: float, value_z: float):
        if axis == "x":
            self.set_piezo_position_x(value_x, value_y)
        if axis == "y":
            self.set_piezo_position_y(value_x, value_y)
        if axis == "z":
            self.set_piezo_position_z(value_z)

    def set_piezo_position_x(self, pos_x, pos_y):
        try:
            self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
            self.con_controller.display_piezo_position_x(self.m.pz.read_position(0))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_y(self, pos_x, pos_y):
        try:
            self.m.daq.set_piezo_position(pos_x / 10., pos_y / 10.)
            self.con_controller.display_piezo_position_y(self.m.pz.read_position(1))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_z(self, pos_z):
        try:
            z = self.m.pz.move_position(2, pos_z)
            self.con_controller.display_piezo_position_z(z)
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    @QtCore.pyqtSlot(float, float)
    def set_galvo(self, voltx: float, volty: float):
        try:
            self.m.daq.set_galvo_position(voltx, volty)
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

    @QtCore.pyqtSlot(list, bool, float)
    def set_laser(self, laser: list, sw: bool, pw: float):
        if sw:
            try:
                self.m.laser.set_constant_power(laser, [pw])
                self.m.laser.laser_on(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")
        else:
            try:
                self.m.laser.laser_off(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")

    def set_lasers(self):
        try:
            self.m.laser.set_modulation_mode(["405", "488_0", "488_1", "488_2"], [0, 0, 0, 0])
            self.m.laser.laser_on("all")
            self.m.laser.set_modulation_mode(["405", "488_0", "488_1", "488_2"],
                                             self.con_controller.get_cobolt_laser_power("all"))
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.m.laser.laser_off("all")
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    @QtCore.pyqtSlot()
    def check_emdccd_temperature(self):
        try:
            self.con_controller.display_camera_temperature(self.m.ccdcam.get_ccd_temperature())
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    @QtCore.pyqtSlot(bool)
    def switch_emdccd_cooler(self, sw: bool):
        if sw:
            self.switch_emdccd_cooler_on()
        else:
            self.switch_emdccd_cooler_off()

    def switch_emdccd_cooler_on(self):
        try:
            self.m.ccdcam.cooler_on()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler_off(self):
        try:
            self.m.ccdcam.cooler_off()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def set_camera_roi(self, key="imaging"):
        try:
            if self.cameras[key] == 0:
                x, y, n, b = self.con_controller.get_emccd_roi()
                self.m.cam_set[0].bin_h, self.m.cam_set[0].bin_h = b, b
                self.m.cam_set[0].start_h, self.m.cam_set[0].end_h = x, x + n - 1
                self.m.cam_set[0].start_v, self.m.cam_set[0].end_v = y, y + n - 1
                self.m.cam_set[0].set_roi()
                self.m.cam_set[0].gain = self.con_controller.get_emccd_gain()
                self.m.cam_set[0].set_gain()
            if self.cameras[key] == 1:
                x, y, n, b = self.con_controller.get_scmos_roi()
                self.m.cam_set[1].set_roi(b, b, x, x + n - 1, y, y + n - 1)
            if self.cameras[key] == 2:
                pass
        except Exception as e:
            self.logg.error(f"Camera Error: {e}")

    @QtCore.pyqtSlot()
    def update_galvo_scanner(self):
        galvo_frequency, galvo_positions, galvo_ranges, dot_pos = self.con_controller.get_galvo_scan_parameters()
        self.p.trigger.update_galvo_scan_parameters(frequency=galvo_frequency, origins=galvo_positions,
                                                    ranges=galvo_ranges, foci=dot_pos)
        self.con_controller.display_dot_step(self.p.trigger.dot_step_v)

    def update_trigger_parameters(self, cam_key):
        try:
            digital_starts, digital_ends = self.con_controller.get_digital_parameters()
            self.p.trigger.update_digital_parameters(digital_starts, digital_ends)
            galvo_frequency, galvo_positions, galvo_ranges, dot_pos = self.con_controller.get_galvo_scan_parameters()
            self.p.trigger.update_galvo_scan_parameters(frequency=galvo_frequency, origins=galvo_positions,
                                                        ranges=galvo_ranges, foci=dot_pos)
            self.p.trigger.dot_step_v = self.con_controller.get_galvo_step()
            axis_lengths, step_sizes = self.con_controller.get_piezo_scan_parameters()
            positions = self.con_controller.get_piezo_positions()
            self.p.trigger.update_piezo_scan_parameters(axis_lengths, step_sizes, positions)
            self.p.trigger.update_camera_parameters(self.m.cam_set[self.cameras[cam_key]].t_clean,
                                                    self.m.cam_set[self.cameras[cam_key]].t_readout,
                                                    self.m.cam_set[self.cameras[cam_key]].t_kinetic)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def generate_live_triggers(self, cam_key):
        self.update_trigger_parameters(cam_key)
        return self.p.trigger.generate_digital_triggers(self.lasers, self.cameras[cam_key])

    def prepare_video(self, vd_mod):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.m.cam_set[self.cameras["imaging"]].prepare_live()
        if vd_mod == "Wide Field":
            self.m.daq.write_digital_sequences(self.generate_live_triggers("imaging"), finite=False)
        elif vd_mod == "Line Scan":
            self.update_trigger_parameters("imaging")
            gtr, ptr, dtr, pos = self.p.trigger.generate_linescan_resolft_2d()
            self.m.daq.write_triggers(piezo_sequences=None, galvo_sequences=gtr, digital_sequences=dtr, finite=False)
        elif vd_mod == "Dot Scan":
            self.update_trigger_parameters("imaging")
            gtr, ptr, dtr, pos = self.p.trigger.generate_dotscan_resolft_2d()
            self.m.daq.write_triggers(piezo_sequences=None, galvo_sequences=gtr, digital_sequences=dtr, finite=False)
        else:
            self.m.cam_set[self.cameras["imaging"]].stop_live()
            self.lasers_off()
            raise ValueError("Invalid video mode")

    def start_video(self, vm):
        try:
            self.prepare_video(vm)
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            return
        try:
            self.m.cam_set[self.cameras["imaging"]].start_live()
            self.m.daq.run_digital_trigger()
            self.thread_video.start()
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")

    def stop_video(self):
        try:
            self.thread_video.quit()
            self.thread_video.wait()
            self.m.daq.stop_triggers()
            self.m.cam_set[self.cameras["imaging"]].stop_live()
            self.lasers_off()
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    @QtCore.pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            self.start_video(md)
        else:
            self.stop_video()

    @QtCore.pyqtSlot()
    def imshow_main(self):
        try:
            self.view_controller.plot_main(self.m.cam_set[self.cameras["imaging"]].get_last_image(),
                                           layer=self.cameras["imaging"])
        except Exception as e:
            self.logg.error(f"Error showing imaging video: {e}")

    @QtCore.pyqtSlot()
    def compute_image_metrics(self):
        try:
            img = self.m.cam_set[self.cameras["imaging"]].get_last_image()
            img = img - img.min()
            img = img / img.max()
            m1 = ipr.calculate_focus_measure(img)
            m2 = ipr.calculate_focus_measure_with_laplacian(img)
            m3 = ipr.calculate_focus_measure_with_sobel(img)
            self.view_controller.display_metrics(m1, m2, m3)
        except Exception as e:
            self.logg.error(f"Error compute image metrics: {e}")

    @QtCore.pyqtSlot(bool)
    def fft(self, sw: bool):
        if sw:
            self.run_fft()
        else:
            self.stop_fft()

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

    @QtCore.pyqtSlot()
    def imshow_fft(self):
        try:
            self.view_controller.plot_fft(
                ipr.fourier_transform(self.view_controller.get_image_data(layer=self.cameras["imaging"])))
        except Exception as e:
            self.logg.error(f"Error showing fft: {e}")

    @QtCore.pyqtSlot(bool)
    def plot_live(self, sw: bool):
        if sw:
            self.start_plot_live()
        else:
            self.stop_plot_live()

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

    @QtCore.pyqtSlot()
    def profile_plot(self):
        try:
            ax = self.con_controller.get_profile_axis()
            self.view_controller.plot_update(
                ipr.get_profile(self.view_controller.get_image_data(layer=self.cameras["imaging"]), ax))
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @QtCore.pyqtSlot()
    def plot_trigger(self):
        try:
            dtr = self.generate_live_triggers("imaging")
            self.view_controller.plot_update(dtr[0])
            for i in range(dtr.shape[0] - 1):
                self.view_controller.plot(dtr[i + 1] + i + 1)
        except Exception as e:
            self.logg.error(f"Error plotting digital triggers: {e}")

    @QtCore.pyqtSlot(str, int)
    def data_acquisition(self, acq_mod: str, acq_num: int):
        if acq_mod == "Wide Field 2D":
            self.run_widefield_zstack(acq_num)
        if acq_mod == "Wide Field 3D":
            self.run_widefield_zstack(acq_num)
        if acq_mod == "Line Scan 2D":
            self.run_line_scanning(acq_num)
        if acq_mod == "Dot Scan 2D":
            self.run_dot_scanning(acq_num)
        if acq_mod == "Bead Scan 2D":
            self.run_bead_scan(acq_num)

    def prepare_widefield_zstack(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.m.cam_set[self.cameras["imaging"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_live_triggers("imaging"), finite=True)

    def widefield_zstack(self):
        try:
            self.prepare_widefield_zstack()
        except Exception as e:
            self.logg.error(f"Error starting widefield zstack: {e}")
            return
        try:
            positions = self.con_controller.get_piezo_positions()
            axis_lengths, step_sizes = self.con_controller.get_piezo_scan_parameters()
            if step_sizes[2] == 0:
                self.logg.error(f"Error running widefield zstack: z step size cannot be zero")
                return
            else:
                num_steps = int(axis_lengths[2] / (2 * step_sizes[2]))
                start = positions[2] - num_steps * step_sizes[2]
                end = positions[2] + num_steps * step_sizes[2]
                zps = np.arange(start, end + step_sizes[2], step_sizes[2])
                data = []
                pzs = []
                self.m.cam_set[self.cameras["imaging"]].start_live()
                for i, z in enumerate(zps):
                    pz = self.m.pz.move_position(2, z)
                    pzs.append([i, pz])
                    self.m.daq.run_digital_trigger()
                    time.sleep(0.05)
                    data.append(self.m.cam_set[self.cameras["imaging"]].get_last_image())
                    self.m.daq.stop_triggers(_close=False)
                self.logg.info(pzs)
                fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_widefield_zstack.tif')
                tf.imwrite(fd, np.asarray(data), imagej=True, resolution=(
                    1 / self.pixel_sizes[self.cameras["imaging"]], 1 / self.pixel_sizes[self.cameras["imaging"]]),
                           metadata={'unit': 'um',
                                     'indices': list(self.m.cam_set[self.cameras["imaging"]].data.ind_list)})
        except Exception as e:
            self.logg.error(f"Error running widefield zstack: {e}")
            return
        self.finish_widefield_zstack()

    def finish_widefield_zstack(self):
        try:
            self.reset_piezo_positions()
            self.m.cam_set[self.cameras["imaging"]].stop_live()
            self.lasers_off()
            self.m.daq.stop_triggers()
            self.logg.info("Widefield image stack acquired")
        except Exception as e:
            self.logg.error(f"Error stopping widefield zstack: {e}")

    def run_widefield_zstack(self, n: int):
        self.run_task(task=self.widefield_zstack, iteration=n)

    def prepare_line_scanning(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.update_trigger_parameters("imaging")
        gtr, ptr, dtr, pos = self.p.trigger.generate_linescan_resolft_2d()
        self.m.cam_set[self.cameras["imaging"]].acq_num = pos
        self.m.cam_set[self.cameras["imaging"]].prepare_data_acquisition()
        self.m.daq.write_triggers(piezo_sequences=ptr, galvo_sequences=gtr, digital_sequences=dtr)

    def line_scanning(self):
        try:
            self.prepare_line_scanning()
        except Exception as e:
            self.logg.error(f"Error preparing confocal scanning: {e}")
            return
        try:
            self.m.cam_set[self.cameras["imaging"]].start_data_acquisition()
            time.sleep(0.02)
            self.m.daq.run_triggers()
            time.sleep(1)
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_confocal_scanning.tif')
            tf.imwrite(fd, self.m.cam_set[self.cameras["imaging"]].get_data(), imagej=True, resolution=(
                1 / self.pixel_sizes[self.cameras["imaging"]], 1 / self.pixel_sizes[self.cameras["imaging"]]),
                       metadata={'unit': 'um', 'indices': list(self.m.cam_set[self.cameras["imaging"]].data.ind_list)})
        except Exception as e:
            self.logg.error(f"Error running confocal scanning: {e}")
            return
        self.finish_line_scanning()

    def finish_line_scanning(self):
        try:
            self.m.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.m.daq.stop_triggers()
            self.lasers_off()
            self.logg.info("Confocal scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping confocal scanning: {e}")

    def run_line_scanning(self, n: int):
        self.run_task(task=self.line_scanning, iteration=n)

    def prepare_dot_scanning(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.update_trigger_parameters("imaging")
        gtr, ptr, dtr, pos = self.p.trigger.generate_dotscan_resolft_2d()
        self.m.cam_set[self.cameras["imaging"]].acq_num = pos
        self.m.cam_set[self.cameras["imaging"]].prepare_data_acquisition()
        self.m.daq.write_triggers(piezo_sequences=ptr, galvo_sequences=gtr, digital_sequences=dtr)

    def dot_scanning(self):
        try:
            self.prepare_dot_scanning()
        except Exception as e:
            self.logg.error(f"Error preparing galvo scanning: {e}")
            return
        try:
            self.m.cam_set[self.cameras["imaging"]].start_data_acquisition()
            time.sleep(0.02)
            self.m.daq.run_triggers()
            time.sleep(1.)
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_galvo_scanning.tif')
            tf.imwrite(fd, self.m.cam_set[self.cameras["imaging"]].get_data(), imagej=True, resolution=(
                1 / self.pixel_sizes[self.cameras["imaging"]], 1 / self.pixel_sizes[self.cameras["imaging"]]),
                       metadata={'unit': 'um', 'indices': list(self.m.cam_set[self.cameras["imaging"]].data.ind_list)})
        except Exception as e:
            self.logg.error(f"Error running galvo scanning: {e}")
            return
        self.finish_dot_scanning()

    def finish_dot_scanning(self):
        try:
            self.m.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.m.daq.stop_triggers()
            self.lasers_off()
            self.logg.info("Galvo scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping galvo scanning: {e}")

    def run_dot_scanning(self, n: int):
        self.run_task(task=self.dot_scanning, iteration=n)

    def prepare_bead_scan(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.m.cam_set[self.cameras["imaging"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_live_triggers("imaging"), finite=True)

    def bead_scan_2d(self):
        try:
            self.prepare_bead_scan()
        except Exception as e:
            self.logg.error(f"Error preparing beads scanning: {e}")
            return
        try:
            positions = self.con_controller.get_piezo_positions()
            axis_lengths, step_sizes = self.con_controller.get_piezo_scan_parameters()
            starts = [pos - length / 2 for pos, length in zip(positions, axis_lengths)]
            ends = [pos + length / 2 for pos, length in zip(positions, axis_lengths)]
            pos = [[starts[dim] + step * step_sizes[dim] for step in
                    range(int((ends[dim] - starts[dim]) / step_sizes[dim]) + 1)] for dim in range(len(positions))]
            data = []
            scan = []
            self.m.cam_set[self.cameras["imaging"]].start_live()
            for z_ in pos[2]:
                pz = self.m.pz.move_position(2, z_)
                time.sleep(0.02)
                for y_ in pos[1]:
                    for x_ in pos[0]:
                        scan.append([x_, y_, z_])
                        self.m.daq.set_piezo_position(x_ / 10., y_ / 10.)
                        time.sleep(0.02)
                        self.m.daq.run_digital_trigger()
                        time.sleep(0.04)
                        data.append(self.m.cam_set[self.cameras["imaging"]].get_last_image())
                        self.m.daq.stop_triggers(_close=False)
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M%S") + '_bead_scanning.tif')
            tf.imwrite(fd, np.asarray(data), imagej=True, resolution=(
                1 / self.pixel_sizes[self.cameras["imaging"]], 1 / self.pixel_sizes[self.cameras["imaging"]]),
                       metadata={'unit': 'um',
                                 'indices': list(self.m.cam_set[self.cameras["imaging"]].data.ind_list),
                                 'scans': scan})
        except Exception as e:
            self.logg.error(f"Error running beads scanning: {e}")
            return
        self.finish_bead_scan()

    def finish_bead_scan(self):
        try:
            self.reset_piezo_positions()
            self.m.cam_set[self.cameras["imaging"]].stop_live()
            self.lasers_off()
            self.m.daq.stop_triggers()
            self.logg.info("Beads scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping confocal scanning: {e}")

    def run_bead_scan(self, n: int):
        self.run_task(task=self.bead_scan_2d, iteration=n)

    @QtCore.pyqtSlot(str)
    def save_data(self, file_name: str):
        try:
            tf.imwrite(file_name + '.tif', self.m.cam_set[self.cameras["imaging"]].get_last_image(), imagej=True,
                       resolution=(
                           1 / self.pixel_sizes[self.cameras["imaging"]],
                           1 / self.pixel_sizes[self.cameras["imaging"]]),
                       metadata={'unit': 'um', 'indices': list(self.m.cam_set[self.cameras["imaging"]].data.ind_list)})
        except Exception as e:
            self.logg.error(f"Error saving data: {e}")

    @QtCore.pyqtSlot(str)
    def select_dm(self, dm_n):
        self.dfm = self.m.dm[dm_n]
        self.v.ao_view.QComboBox_cmd.clear()
        self.v.ao_view.QComboBox_cmd.addItems([str(i) for i in range(len(self.dfm.dm_cmd))])
        self.v.ao_view.QComboBox_cmd.setCurrentIndex(self.dfm.current_cmd)

    @QtCore.pyqtSlot(int, float)
    def push_actuator(self, n: int, a: float):
        try:
            values = [0.] * self.dfm.n_actuator
            values[n] = a
            self.dfm.set_dm(self.dfm.cmd_add(values, self.dfm.dm_cmd[self.dfm.current_cmd]))
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    def set_zernike(self, factory=False):
        try:
            indz, amp = self.ao_controller.get_zernike_mode()
            if factory:
                self.dfm.set_dm(
                    self.dfm.cmd_add([i * amp for i in self.dfm.z2c[indz]], self.dfm.dm_cmd[self.dfm.current_cmd]))
            else:
                self.dfm.set_dm(
                    self.dfm.cmd_add(self.dfm.get_zernike_cmd(indz, amp), self.dfm.dm_cmd[self.dfm.current_cmd]))
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @QtCore.pyqtSlot()
    def set_dm(self):
        try:
            i = int(self.ao_controller.get_cmd_index())
            self.dfm.set_dm(self.dfm.dm_cmd[i])
            self.dfm.current_cmd = i
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @QtCore.pyqtSlot()
    def update_dm(self):
        try:
            self.dfm.dm_cmd.append(self.dfm.temp_cmd[-1])
            self.ao_controller.update_cmd_index()
            self.dfm.set_dm(self.dfm.dm_cmd[-1])
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @QtCore.pyqtSlot(str)
    def load_dm(self, filename: str):
        try:
            self.dfm.dm_cmd.append(self.dfm.read_cmd(filename))
            self.dfm.set_dm(self.dfm.dm_cmd[-1])
            print('New DM cmd loaded')
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @QtCore.pyqtSlot()
    def save_dm(self):
        try:
            t = time.strftime("%Y%m%d_%H%M%S_")
            self.dfm.write_cmd(self.data_folder, t, flatfile=False)
            self.logg.info('DM cmd saved')
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    def update_wfs_trigger_parameters(self, cam_key):
        try:
            digital_starts, digital_ends = self.con_controller.get_digital_parameters()
            self.p.trigger.update_digital_parameters(digital_starts, digital_ends)
            self.p.trigger.update_camera_parameters(self.m.cam_set[self.cameras[cam_key]].t_clean,
                                                    self.m.cam_set[self.cameras[cam_key]].t_readout,
                                                    self.m.cam_set[self.cameras[cam_key]].t_kinetic)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def generate_wfs_trigger(self, cam_key):
        self.update_wfs_trigger_parameters(cam_key)
        return self.p.trigger.generate_digital_triggers(self.lasers, self.cameras[cam_key])

    def set_img_wfs(self):
        try:
            parameters = self.ao_controller.get_parameters_img()
            self.p.shwfsr.update_parameters(parameters)
            self.logg.info('SHWFS parameter updated')
        except Exception as e:
            self.logg.error(f"SHWFS Error: {e}")

    def _prepare_img_wfs(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["wfs"] = self.ao_controller.get_wfs_camera()
        self.set_camera_roi("wfs")
        self.set_img_wfs()
        self.m.cam_set[self.cameras["wfs"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_wfs_trigger("wfs"), finite=False)

    def _start_img_wfs(self):
        try:
            self._prepare_img_wfs()
        except Exception as e:
            self.logg.error(f"Error starting wfs: {e}")
        try:
            self.m.cam_set[self.cameras["wfs"]].start_live()
            self.m.daq.run_digital_trigger()
            self.thread_wfs.start()
        except Exception as e:
            self.logg.error(f"Error starting wfs: {e}")

    def _stop_img_wfs(self):
        try:
            self.thread_wfs.quit()
            self.thread_wfs.wait()
            self.m.cam_set[self.cameras["wfs"]].stop_live()
            self.m.daq.stop_triggers()
            self.lasers_off()
        except Exception as e:
            self.logg.error(f"Error stopping wfs: {e}")

    def img_wfs(self, sw):
        if sw:
            self._start_img_wfs()
        else:
            self._stop_img_wfs()

    @QtCore.pyqtSlot()
    def imshow_img_wfs(self):
        try:
            self.p.shwfsr.meas = self.m.cam_set[self.cameras["wfs"]].get_last_image()
            self.view_controller.plot_sh(self.p.shwfsr.meas, layer=self.cameras["wfs"])
        except Exception as e:
            self.logg.error(f"Error showing shwfs: {e}")

    @QtCore.pyqtSlot()
    def set_reference_wf(self):
        try:
            self.p.shwfsr.ref = self.m.cam_set[self.cameras["wfs"]].get_last_image()
            self.view_controller.plot_shb(self.p.shwfsr.ref)
            self.logg.info('shwfs base set')
        except Exception as e:
            self.logg.error(f"Error setting shwfs base: {e}")

    @QtCore.pyqtSlot()
    def run_img_wfr(self):
        self.run_task(task=self.img_wfr, callback=self.imshow_img_wfr)

    def img_wfr(self):
        try:
            self.p.shwfsr.method = self.ao_controller.get_gradient_method_img()
            # self.p.shwfsr.ref = self.view_controller.get_image_data(4)
            # self.p.shwfsr.meas = self.view_controller.get_image_data(self.cameras["wfs"])
            self.p.shwfsr.wavefront_reconstruction()
        except Exception as e:
            self.logg.error(f"SHWFS Reconstruction Error: {e}")

    def imshow_img_wfr(self):
        try:
            self.view_controller.plot_wf(self.p.shwfsr.wf)
            self.ao_controller.display_img_wf_properties(ipr.img_properties(self.p.shwfsr.wf))
        except Exception as e:
            self.logg.error(f"SHWFS Wavefront Show Error: {e}")

    @QtCore.pyqtSlot()
    def compute_img_wf(self):
        # self.dfm.run_wf_modal_recon()
        # self.view_controller.plot_update(self.dfm.az)
        self.imshow_img_wfr()

    @QtCore.pyqtSlot(str)
    def save_img_wf(self, file_name: str):
        try:
            tf.imwrite(file_name + '_shimg_base_raw.tif', self.p.shwfsr.ref)
        except Exception as e:
            self.logg.error(f"Error saving shwfs _base: {e}")
        try:
            tf.imwrite(file_name + '_shimg_offset_raw.tif', self.p.shwfsr.meas)
        except Exception as e:
            self.logg.error(f"Error saving shwfs offset: {e}")
        try:
            tf.imwrite(file_name + '_shimg_processed.tif', self.p.shwfsr.im)
        except Exception as e:
            self.logg.error(f"Error saving shwfs imgstack: {e}")
        try:
            tf.imwrite(file_name + '_reconstructed_wf.tif', self.p.shwfsr.wf)
        except Exception as e:
            self.logg.error(f"Error saving shwfs wavefront: {e}")

    @QtCore.pyqtSlot()
    def run_influence_function(self):
        try:
            self._prepare_influence_function()
        except Exception as e:
            self.logg.error(f"Error prepare influence function: {e}")
            return
        self.run_task(self.influence_function, callback=self._finish_influence_function)

    def _prepare_influence_function(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["wfs"] = self.ao_controller.get_wfs_camera()
        self.set_camera_roi("wfs")
        self.m.cam_set[self.cameras["wfs"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_wfs_trigger("wfs"), finite=True)

    def influence_function(self):
        try:
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M") + '_influence_function')
            os.makedirs(fd, exist_ok=True)
            self.logg.info(f'Directory {fd} has been created successfully.')
        except Exception as er:
            self.logg.error(f'Error creating directory: {er}')
            return
        n, amp = self.ao_controller.get_actuator()
        self.m.cam_set[self.cameras["wfs"]].start_live()
        for i in range(self.dfm.n_actuator):
            shimg = []
            self.v.dialog_text.setText(f"actuator {i}")
            values = [0.] * self.dfm.n_actuator
            self.dfm.set_dm(values)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            shimg.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values[i] = amp
            self.dfm.set_dm(values)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            shimg.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values = [0.] * self.dfm.n_actuator
            self.dfm.set_dm(values)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            shimg.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            values[i] = - amp
            self.dfm.set_dm(values)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            shimg.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        try:
            md = self.ao_controller.get_img_wfs_method()
            self.v.dialog_text.setText(f"computing influence function")
            self.p.shwfsr.generate_influence_matrix(data_folder=fd, dm_info=(self.dfm.n_actuator, self.dfm.amp),
                                                    method=md, sv=True)
        except Exception as e:
            self.logg.error(f"Error computing influence function: {e}")
            return

    def single_actuator(self, act_ind, p_amp):
        self.logg.info(f"actuator # {act_ind}")
        self.dfm.set_dm(self.dfm.dm_cmd[-1])
        values = [0.] * self.dfm.n_actuator
        values[act_ind] = p_amp
        self.dfm.set_dm(values)
        time.sleep(0.02)
        self.m.daq.run_digital_trigger()
        time.sleep(0.02)
        self.m.daq.stop_triggers(_close=False)
        return self.m.cam_set[self.cameras["wfs"]].get_last_image()

    def _finish_influence_function(self):
        try:
            self.lasers_off()
            self.m.cam_set[self.cameras["wfs"]].stop_live()
            self.m.daq.stop_triggers()
        except Exception as e:
            self.logg.error(f"Error finishing influence function: {e}")

    @QtCore.pyqtSlot(int)
    def run_close_loop_correction(self, nlp: int):
        try:
            self._prepare_close_loop_correction()
        except Exception as e:
            self.logg.error(f"Prepare CloseLoop Correction Error: {e}")
            return
        try:
            self.run_task(task=self.close_loop_correction, iteration=nlp, callback=self._finish_close_loop_correction)
        except Exception as e:
            self.logg.error(f"CloseLoop Correction Error: {e}")

    def _prepare_close_loop_correction(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["wfs"] = self.ao_controller.get_wfs_camera()
        self.set_camera_roi("wfs")
        self.m.cam_set[self.cameras["wfs"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_wfs_trigger("wfs"), finite=True)

    def close_loop_correction(self):
        self.m.cam_set[self.cameras["wfs"]].start_live()
        self.m.daq.run_digital_trigger()
        time.sleep(0.04)
        self.p.shwfsr.meas = self.m.cam_set[self.cameras["wfs"]].get_last_image()
        self.m.daq.stop_triggers(_close=False)
        if self.ao_controller.get_img_wfs_method() == "phase":
            self.dfm.get_correction(self.p.shwfsr.wavefront_reconstruction(rt=True), method="phase")
        else:
            self.dfm.get_correction(self.p.shwfsr.get_gradient_xy(),
                                    method=self.ao_controller.get_img_wfs_method())
        self.dfm.set_dm(self.dfm.dm_cmd[-1])
        self.ao_controller.update_cmd_index()
        i = int(self.ao_controller.get_cmd_index())
        self.dfm.current_cmd = i

    def _finish_close_loop_correction(self):
        try:
            # self.p.shwfsr.ref = self.view_controller.get_image_data(4)
            self.m.daq.run_digital_trigger()
            self.p.shwfsr.meas = self.m.cam_set[self.cameras["wfs"]].get_last_image()
            self.m.cam_set[self.cameras["wfs"]].stop_live()
            self.m.daq.stop_triggers()
            self.lasers_off()
            self.img_wfr()
            self.view_controller.plot_wf(self.p.shwfsr.wf)
        except Exception as e:
            self.logg.error(f"CloseLoop Correction Error: {e}")

    @QtCore.pyqtSlot()
    def run_sensorless_iteration(self):
        try:
            self._prepare_sensorless_iteration()
        except Exception as e:
            self.logg.error(f"Prepare sensorless iteration Error: {e}")
            return
        self.run_task(task=self.sensorless_iteration, callback=self._finish_sensorless_iteration)

    def _prepare_sensorless_iteration(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["imaging"] = self.con_controller.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.m.cam_set[self.cameras["imaging"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_live_triggers("imaging"), finite=True)

    def sensorless_iteration(self):
        try:
            lpr, hpr, mindex, metric = self.ao_controller.get_ao_parameters()
            name = time.strftime("%Y%m%d_%H%M%S_") + '_ao_iteration_' + metric
            new_folder = os.path.join(self.data_folder, name)
            os.makedirs(new_folder, exist_ok=True)
            self.logg.info(f'Directory {new_folder} has been created successfully.')
        except Exception as e:
            self.logg.error(f'Error creating directory for sensorless iteration: {e}')
            return
        try:
            mode_start, mode_stop, amp_start, amp_step, amp_step_number = self.ao_controller.get_ao_iteration()
            results = [('Mode', 'Amp', 'Metric')]
            za = []
            mv = []
            zp = [0] * self.dfm.n_zernike
            cmd = self.dfm.dm_cmd[self.dfm.current_cmd]
            self.m.cam_set[self.cameras["imaging"]].start_live()
            self.logg.info("Sensorless AO iteration starts")
            self.dfm.set_dm(cmd)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            self.m.daq.stop_triggers(_close=False)
            fn = os.path.join(new_folder, 'original.tif')
            tf.imwrite(fn, self.m.cam_set[self.cameras["imaging"]].get_last_image())
            for mode in range(mode_start, mode_stop + 1):
                self.v.dialog_text.setText(f"Zernike mode #{mode}")
                amprange = []
                dt = []
                for stnm in range(amp_step_number):
                    amp = amp_start + stnm * amp_step
                    amprange.append(amp)
                    self.dfm.set_dm(self.dfm.cmd_add(self.dfm.get_zernike_cmd(mode, amp), cmd))
                    # self.dfm.set_dm(self.dfm.cmd_add([i * amp for i in self.dfm.z2c[mode]], cmd))
                    time.sleep(0.02)
                    self.m.daq.run_digital_trigger()
                    time.sleep(0.04)
                    self.m.daq.stop_triggers(_close=False)
                    fn = "zm%0.2d_amp%.4f" % (mode, amp)
                    fn1 = os.path.join(new_folder, fn + '.tif')
                    tf.imwrite(fn1, self.m.cam_set[self.cameras["imaging"]].get_last_image())
                    if mindex == 0:
                        dt.append(ipr.snr(self.m.cam_set[self.cameras["imaging"]].get_last_image(), lpr, hpr))
                    if mindex == 1:
                        dt.append(self.m.cam_set[self.cameras["imaging"]].get_last_image().max())
                    if mindex == 2:
                        dt.append(ipr.hpf(self.m.cam_set[self.cameras["imaging"]].get_last_image(), hpr))
                    results.append((mode, amp, dt[stnm]))
                za.extend(amprange)
                mv.extend(dt)
                self.logg.info(f"zernike mode #{mode}, ({amprange}), ({dt})")
                try:
                    pmax = ipr.peak_find(amprange, dt)
                    zp[mode] = pmax
                    self.logg.info("setting mode %d at value of %.4f" % (mode, pmax))
                    cmd = self.dfm.cmd_add(self.dfm.get_zernike_cmd(mode, pmax), cmd)
                    self.dfm.set_dm(cmd)
                except ValueError as e:
                    self.logg.error(f"mode {mode} error {e}")
            self.dfm.set_dm(cmd)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            self.m.daq.stop_triggers(_close=False)
            fn = os.path.join(new_folder, 'final.tif')
            tf.imwrite(fn, self.m.cam_set[self.cameras["imaging"]].get_last_image())
            self.dfm.dm_cmd.append(cmd)
            self.ao_controller.update_cmd_index()
            i = int(self.ao_controller.get_cmd_index())
            self.dfm.current_cmd = i
            self.dfm.write_cmd(new_folder, '_')
            self.dfm.save_sensorless_results(os.path.join(new_folder, 'results.xlsx'), za, mv, zp)
        except Exception as e:
            self.logg.error(f"Sensorless AO Error: {e}")

    def _finish_sensorless_iteration(self):
        try:
            self.lasers_off()
            self.m.daq.stop_triggers()
            self.m.cam_set[self.cameras["imaging"]].stop_live()
            self.logg.info("sensorless AO finished")
        except Exception as e:
            self.logg.error(f"Finish Sensorless AO Error: {e}")

    @QtCore.pyqtSlot()
    def run_shwfs_acquisition(self):
        try:
            self._prepare_shwfs_acquisition()
        except Exception as e:
            self.logg.error(f"Error prepare shwfs acquisition: {e}")
            return
        self.run_task(self.shwfs_acquisition, callback=self._finish_shwfs_acquisition)

    def _prepare_shwfs_acquisition(self):
        self.lasers = self.con_controller.get_lasers()
        self.set_lasers()
        self.cameras["wfs"] = self.ao_controller.get_wfs_camera()
        self.set_camera_roi("wfs")
        self.m.cam_set[self.cameras["wfs"]].prepare_live()
        self.m.daq.write_digital_sequences(self.generate_wfs_trigger("wfs"), finite=True)

    def shwfs_acquisition(self):
        try:
            fd = os.path.join(self.data_folder, time.strftime("%Y%m%d%H%M") + '_shwfs_acquisition')
            os.makedirs(fd, exist_ok=True)
            self.logg.info(f'Directory {fd} has been created successfully.')
        except Exception as er:
            self.logg.error(f'Error creating directory: {er}')
            return
        modes = np.arange(4, 22)
        self.m.cam_set[self.cameras["wfs"]].start_live()
        for i in range(64):
            self.v.dialog_text.setText(f"Acquisition #{i}")
            data = []
            cmd = self.dfm.dm_cmd[self.dfm.current_cmd]
            self.dfm.set_dm(cmd)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            data.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            amps = np.random.rand(modes.shape[0]) / 5
            for m, mode in enumerate(modes):
                amp = amps[m]
                cmd = self.dfm.cmd_add(self.dfm.get_zernike_cmd(mode, amp), cmd)
            self.dfm.set_dm(cmd)
            time.sleep(0.02)
            self.m.daq.run_digital_trigger()
            time.sleep(0.04)
            data.append(self.m.cam_set[self.cameras["wfs"]].get_last_image())
            self.m.daq.stop_triggers(_close=False)
            try:
                self.p.shwfsr.method = self.ao_controller.get_gradient_method_img()
                self.p.shwfsr.ref = data[0]
                self.p.shwfsr.meas = data[1]
                self.p.shwfsr.wavefront_reconstruction()
            except Exception as e:
                self.logg.error(f"SHWFS Reconstruction Error: {e}")
                return
            t = time.strftime("%Y%m%d_%H%M%S_")
            fn = os.path.join(fd, t + "shwfs_proc_images.tif")
            tf.imwrite(fn, self.p.shwfsr.im)
            fn = os.path.join(fd, t + "shwfs_recon_wf.tif")
            tf.imwrite(fn, self.p.shwfsr.wf)
            fn = os.path.join(fd, t + "shwfs_wf_zcoffs.xlsx")
            df = pd.DataFrame(amps, index=modes, columns=['Amplitudes'])
            with pd.ExcelWriter(fn, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Zernike Amplitudes')

    def _finish_shwfs_acquisition(self):
        try:
            self.lasers_off()
            self.m.cam_set[self.cameras["wfs"]].stop_live()
            self.m.daq.stop_triggers()
        except Exception as e:
            self.logg.error(f"Error finishing influence function: {e}")


class TaskWorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)


class TaskWorker(QtCore.QObject):
    def __init__(self, task=None, n=1, parent=None):
        super().__init__(parent)
        self.task = task if task is not None else self._do_nothing
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
