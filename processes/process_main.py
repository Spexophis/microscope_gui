from processes import beads_scan_reconstruction
from processes import process_image
from processes import process_shwfs
from processes import process_trigger


class MainProcess:

    def __init__(self):
        self.imgprocess = process_image.ImageProcessing()
        self.shwfsr = process_shwfs.WavefrontSensing()
        self.trigger = process_trigger.TriggerSequence()
        self.bsrecon = beads_scan_reconstruction.BeadScanReconstruction()
