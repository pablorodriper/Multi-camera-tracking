class Detection(object):
    def __init__(self, frame, label, xtl, ytl, width, height, confidence=1, track_id=None, histogram=None):
        self.frame = frame
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.width = width
        self.height = height
        self.confidence = confidence
        self.bbox = [xtl, ytl, xtl+width, ytl+height]
        self.track_id = track_id
        self.histogram = histogram

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n frame={0}, label={1}, confidence={6}, TopLeftXY=({2},{3}), width={4}, height={5}'.format(self.frame, self.label, self.xtl, self.ytl, self.width, self.height, self.confidence)
