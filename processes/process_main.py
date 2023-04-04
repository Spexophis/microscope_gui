import process_image
import process_aotool
import process_ShackHartmannWavefrontReconstruction
import process_trigger
import beads_scan_reconstruction


class MainProcess:

    def __init__(self):
        self.imgprocess = process_image.ImageProcessing()
        self.aotool = process_aotool.aotool()
        self.shwfsr = process_ShackHartmannWavefrontReconstruction.Wavefront_Reconstruction()
        self.trigger = process_trigger.Trigger_Sequence()
        self.bsrecon = beads_scan_reconstruction.BeadScanReconstruction()
