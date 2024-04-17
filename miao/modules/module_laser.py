import pycobolt


class CoboltLaser:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        laser_dict = {"405": 'COM4',
                      "488_0": 'COM5',
                      "488_1": 'COM6',
                      "488_2": 'COM7'}
        self.lasers, self._h = self._initiate_lasers(laser_dict)

    def __del__(self):
        pass

    def _initiate_lasers(self, laser_dict):
        lasers = {}
        for laser, com_port in laser_dict.items():
            try:
                lasers[laser] = pycobolt.Cobolt06MLD(port=com_port)
                self.logg.info("{} Laser Connected".format(laser))
            except Exception as e:
                self.logg.error(f"405 nm Laser Error: {e}")
        _h = {key: True for key in lasers.keys()}
        return lasers, _h

    def close(self):
        self.laser_off("all")
        for key, _l in self._h.items():
            if _l:
                del self.lasers[key]

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def laser_off(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l0')
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].send_cmd('l0')

    def laser_on(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l1')
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].send_cmd('l1')

    def set_constant_power(self, laser, power):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].constant_power(power[ind])

    def set_constant_current(self, laser, current):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].constant_current(current[ind])

    def set_modulation_mode(self, laser, pw):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].modulation_mode(pw[ind])
                self.lasers[ln].digital_modulation(enable=1)
