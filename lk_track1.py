#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
import time
import math
from common import anorm2, draw_str
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 5
        self.detect_interval = 1
        self.tracks = []
        self.cam = cv2.VideoCapture('1.mp4')
        self.frame_idx = 128

    def run(self):
        mats = []
        ckos = []
        while True:
            #time.sleep(0.1)
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 10
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                        #pass
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                        #pass
                    new_tracks.append(tr)
                    #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                
                tracks_copy = []
                for track in new_tracks:
                    lengt = len(track)-1
                    summ = (np.array(track[0])-np.array(track[lengt]))/len(track)
                    summ[0] = math.fabs(summ[0])
                    summ[1] = math.fabs(summ[1])
                    if (summ[0] > 0.5 or summ[1] > 0.5):
                        tracks_copy.append(track)
                
                last_points = []
                if len(tracks_copy) > 1:
                    for track in tracks_copy:
                        lengt = len(track)-1
                        last_points.append(track[lengt])
                    #print(last_points)
                    mx = 0
                    my = 0
                    for l_point in last_points:
                        mx += l_point[0]
                        my += l_point[1]
                    mx = mx / len(last_points)
                    my = my / len(last_points)
                    mat = [round(mx,2), round(my,2)]
                    cv2.circle(vis, (np.int32(mat[0]), np.int32(mat[1])), 5, 0, -1)
                    #print(mat)
                    cko = round(np.mean(mat),2)
                    #print(cko)
                    camera_w = 320
                    camera_a = 60
                    cat_w = 0.3
                    cat_a = cko * camera_a / camera_w
                    cat_a_r = cat_a * math.pi / 180
                    #my = cat_w / math.tan(cat_a_r)
                    my = cat_w / math.atan(cat_a_r)
                    mat = [round(mx,2), round(my,2)]
                    #print(mat)
                    mats.append(mat)
                    ckos.append(cko)
                
                self.tracks = new_tracks
                #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 255))
                cv2.polylines(vis, [np.int32(tr) for tr in tracks_copy], False, (0, 255, 0))
                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                draw_str(vis, (20, 20), 'track count: %d' % len(tracks_copy))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    #cv2.circle(mask, (x, y), 5, 0, -1)
                    pass
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        #print(mats)
        #print(ckos)
        iterat = 1
        mats_good = []
        cur = mats[0]
        mats_good.append(cur)
        for mat in mats:
            if iterat != len(mats):
                check = math.fabs(cur[1] - mats[iterat][1])
                if check < 0.4:
                    mats_good.append(mats[iterat])
                    cur = mats[iterat]
            iterat += 1
        
        f = open('plot.txt', 'w')
        for mat in mats_good:
            f.write(str(mat[0]))
            f.write(' ')
            f.write(str(mat[1]))
            f.write('\n')
        f.close()

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
