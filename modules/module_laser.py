import pycobolt


class CoboltLaser:

    def __init__(self, logg=None):
        super().__init__()
        if logg is None:
            import logging
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
            self.logg = logging
        else:
            self.logg = logg
        self.lasers = {}
        try:
            self.lasers["405"] = pycobolt.Cobolt06MLD(port='COM4')
            self.logg.info('405 nm Laser Connected')
        except Exception as e:
            self.logg.error(f"405 nm Laser Error: {e}")
        try:
            self.lasers["488_0"] = pycobolt.Cobolt06MLD(port='COM5')
            self.logg.info('488 nm #0 Laser Connected')
        except Exception as e:
            self.logg.error(f"488 nm #0 Laser Error: {e}")
        try:
            self.lasers["488_1"] = pycobolt.Cobolt06MLD(port='COM6')
            self.logg.info('488 nm Laser #1 Connected')
        except Exception as e:
            self.logg.error(f"488 nm Laser #1 Error: {e}")
        try:
            self.lasers["488_2"] = pycobolt.Cobolt06MLD(port='COM7')
            self.logg.info('488 nm Laser #2 Connected')
        except Exception as e:
            self.logg.error(f"488 nm Laser #2 Error: {e}")
        self._h = {key: True for key in self.lasers.keys()}

    def __del__(self):
        pass

    def close(self):
        self.laser_off("all")
        for key, _l in self._h.items():
            if _l:
                del self.lasers[key]

    def laser_off(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l0')
        else:
            self.lasers[laser].send_cmd('l0')

    def laser_on(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l1')
        else:
            self.lasers[laser].send_cmd('l1')

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
