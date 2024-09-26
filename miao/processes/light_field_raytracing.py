class LightFieldRays:

    def __init__(self):
        self.f_obj = 3  # mm
        self.f_tub = 200  # mm
        self.f_ml = 5.2  # mm
        self.d_ml = 6.75  # mm

    def compute_ray(self, d0, alpha0):
        f = self.f_obj + self.f_tub
        m = self.f_tub / self.f_obj
        alpha = (((d0 + self.f_obj) * (1 + (2 / m)) + alpha0 * self.f_tub) / self.f_ml) - ((f * alpha0 + ((d0 + self.f_obj) / m)) / self.f_tub) + alpha0
        d = self.d_ml * alpha - (d0 + self.f_obj) * (1 + (2 / m)) + alpha0 * self.f_tub
        return d, alpha
