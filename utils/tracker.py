from .yolox.tracker.byte_tracker import BYTETracker
from .yolox.tracking_utils.timer import Timer
from .yolox.utils.visualize import plot_tracking



class Param:
    track_thresh = 0.  # 0.5
    track_buffer = 2
    match_thresh = 0.8
    min_box_area = 10
    aspect_ratio_thresh = 1.6
    mot20 = False
    num_outside_frame = 1


class Tracker:
    def __init__(self, size) -> None:
        self.size = size
        self.params = Param()
        self.tracker = BYTETracker(self.params, frame_rate=30)
        self.timer = Timer()
        self.frame_id = 0
        self.prices = [100, 30, 40, 60, 80]
        self.last_price = 0
        self.total_price = 0

    def __call__(self, dets, class_ids, img):
        online_targets = self.tracker.update(dets, class_ids, img.shape[:2], self.size)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > self.params.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)

            ###########################################################
            cond1 = not t.is_price_caculated
            cond2 = t.num_outside_frame >= self.params.num_outside_frame
            cond3 = t.is_start_inside
            if cond1 and cond2 and cond3:
                self.last_price = self.prices[t.class_id]
                self.total_price += self.last_price
                t.is_start_inside = False
                t.is_price_caculated = True
                print(f'current price: {self.total_price}')
            #elif not t.is_start_inside and not t.is_outside and not t.is_price_caculated:
            elif not t.is_start_inside and not t.is_outside:
                self.last_price = -self.prices[t.class_id]
                self.total_price = max(0, self.total_price + self.last_price)
                t.is_start_inside = True
                t.is_price_caculated = False
                t.num_outside_frame = 0
                print(f'current price: {self.total_price}')
            ###########################################################

        self.timer.toc()
        plotted = plot_tracking(img, 
                                online_tlwhs, 
                                online_ids, 
                                self.total_price,
                                self.last_price,
                                frame_id=self.frame_id + 1, 
                                fps=1. / self.timer.average_time)

        return plotted
    
    def reset(self):
        self.total_price = 0
        self.last_price = 0
        self.tracker.reset()