import matplotlib._png as png


class DataContainer:
    def __init__(self, img_path, traffic_light, EM = None):
        self.img = img_path
        self.EM = EM
        self.auxialary = []
        self.tfls = []
        self.candidates = []
        self.traffic_light = traffic_light
