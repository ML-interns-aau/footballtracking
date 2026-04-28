import cv2
import numpy as np

class CameraMotionEstimator:
    def __init__(self, frame: np.ndarray):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        self.cumulative_motion = np.zeros(2)

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.p0 is None or len(self.p0) < 10:
            self.p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            
        if self.p0 is None:
            self.prev_gray = curr_gray
            return 0.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.p0, None, **self.lk_params)
        
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            if len(good_new) > 0 and len(good_old) > 0:
                motion = good_old - good_new
                avg_motion = np.mean(motion, axis=0)
                
                self.p0 = good_new.reshape(-1, 1, 2)
            else:
                avg_motion = np.zeros(2)
                self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
        else:
            avg_motion = np.zeros(2)
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            
        self.prev_gray = curr_gray
        self.cumulative_motion += avg_motion
        
        return float(avg_motion[0]), float(avg_motion[1])
