# adapted from: https://github.com/Sentdex/pygta5/blob/master/grabscreen.py
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import matplotlib.pyplot as plt
from ctypes import windll
from config import *

def capture_win_alt(window_name: str, hwnd):
    # Adapted from https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui

    windll.user32.SetProcessDPIAware()
    #hwnd = win32gui.FindWindow(None, window_name)


    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top
    bar_height = 35
    offset_height_top = 135 
    offset_height_bottom = 135 


    offset_sides = 100 # ignore this many pixels on sides, 
    width =  w - 2*offset_sides
    height = h - offset_height_top-offset_height_bottom

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bitmap)

    # If Special K is running, this number is 3. If not, 1
    result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
    img = img[bar_height+offset_height_top:height + bar_height+offset_height_top, offset_sides:width + offset_sides]

    if not result:  # result should be 1
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        img = np.zeros((width, height, 3), dtype=np.uint8)
        print(f"Unable to acquire screenshot! Result: {result}")

    if True:
            # because we use a shrunk image for input into the NN
            # we kind of want to make it larger so we can see what's happening
            # of course it's lossy compared to the original game
            target_width = 800
            scale = target_width / img.shape[1] # how much to magnify
            dim = (target_width,int(img.shape[0] * scale))
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('resized',resized) 


    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


    # Release resources
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img


def fps_capture_test():

    if False:
        # can use this to manually find hwin, id of selected window
        # actually can look this up from name directly
        while True:
            hwin = win32gui.GetForegroundWindow()
            print(hwin)
            time.sleep(0.2)

    time_start = time.time()
    n_grabs=20000
    hwin = win32gui.FindWindow(None,'Counter-Strike 2')
    for i in range(n_grabs):
        img_small = capture_win_alt("Counter-Strike 2", hwin)
        if True:
            # because we use a shrunk image for input into the NN
            # we kind of want to make it larger so we can see what's happening
            # of course it's lossy compared to the original game
            target_width = 800
            scale = target_width / img_small.shape[1] # how much to magnify
            dim = (target_width,int(img_small.shape[0] * scale))
            resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('resized',resized) 


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

    time_end = time.time()
    avg_time = (time_end-time_start)/n_grabs
    fps = 1/avg_time
    print('avg_time',np.round(avg_time,5))
    print('fps',np.round(fps,2))
    return


if __name__ == "__main__":
    fps_capture_test()