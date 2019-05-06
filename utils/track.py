class Track(object):
    def __init__(self, id, detections, time_since_update=0, hits=0, hit_streak=0, current_update=False, global_id=None):
        self.id = id
        self.detections = detections
        self.time_since_update = time_since_update
        self.hits = hits
        self.hit_streak = hit_streak
        self.current_update = current_update
        self.global_id = global_id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n Id={0}, Detections={1}, Time since update={2}, Hits={3}, Hits streak={4}'.format(self.id, self.detections, self.time_since_update, self.hits, self.hit_streak)
