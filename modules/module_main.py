import pprint

from modules import module_andorixon
from modules import module_deformablemirror
from modules import module_hamamatsu
from modules import module_laser
from modules import module_mcldeck
from modules import module_mclpiezo
from modules import module_nidaq


class MainModule:

    def __init__(self, config, logg, path):
        self.config = config
        self.logg = logg
        self.data_folder = path
        try:
            self.ccdcam = module_andorixon.EMCCDCamera(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.scmoscam = module_hamamatsu.HamamatsuCameraMR(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.dm = module_deformablemirror.DeformableMirror(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.laser = module_laser.CoboltLaser(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.daq = module_nidaq.NIDAQ(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.md = module_mcldeck.MCLMicroDrive(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.pz = module_mclpiezo.MCLNanoDrive(self.logg.error_log)
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        pprint.pprint("Finish initiating devices")

    def close(self):
        try:
            self.ccdcam.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.scmoscam.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.laser.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.dm.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.daq.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.md.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
        try:
            self.pz.close()
        except Exception as e:
            self.logg.error_log.error(f"{e}")
