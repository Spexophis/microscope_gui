from processes import beads_scan_reconstruction, process_shwfs, process_image, process_trigger


class MainProcess:

    def __init__(self):
        self.imgprocess = process_image.ImageProcessing()
        self.shwfsr = process_shwfs.WavefrontSensing()
        self.trigger = process_trigger.TriggerSequence()
        self.bsrecon = beads_scan_reconstruction.BeadScanReconstruction()
