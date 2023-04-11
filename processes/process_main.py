from processes import process_image
from processes import process_shwfs
from processes import process_trigger
from processes import beads_scan_reconstruction


class MainProcess:

    def __init__(self):
        self.imgprocess = process_image.ImageProcessing()
        self.shwfsr = process_shwfs.SHWavefrontSensing()
        self.trigger = process_trigger.Trigger_Sequence()
        self.bsrecon = beads_scan_reconstruction.BeadScanReconstruction()
