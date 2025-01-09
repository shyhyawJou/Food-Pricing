import cv2
import numpy as np
import os
from PxUtil import MJPEGStreamerManager, convert2RGB
from loguru import logger
import re
from time import time
import threading
import subprocess
#from detect_tflite_my import Sushi_Plate_Detector
from detect_dla import Sushi_Plate_Detector
from utils import Tracker, byte_tracker



class StreamManager:
    def __init__(self):
        self.g_streamMgr = MJPEGStreamerManager()
        self.g_streamPort = 9527
        self.stream_name = "Sushi_demo_camera_2_stream"
        self.stream_width = 640
        self.stream_height = 480
        self.listener_thread = None
        self.stop_event = threading.Event()
        self.capture = None
        #self.model = Sushi_Plate_Detector('best_float32.tflite')
        self.model = Sushi_Plate_Detector('best_float32.dla')
        self.wh = [448, 448]
        self.tracker = Tracker(self.wh[::-1])
        self.do_tracking = True
        self.show_roi = False
        self.show_fps = False
        #self.roi = [300, 0, 200, 480]
        self.roi = [180, 0, 200, 480]  # x0, y0, w, h
        byte_tracker.BOUNDARY_X = self.roi[0] + self.roi[2]

    def start_camera(self):
        # 啟動相機
        try:
            script_path = os.path.expanduser('find_camera.sh')
            result = subprocess.run(
                ["sh", script_path],
                capture_output=True,    
                text=True, 
                check=True
            )
            camera_id = int(re.search('\d+', result.stdout)[0])
            print(f"相機 ID 是：{camera_id}")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")
        
        self.capture = cv2.VideoCapture(camera_id)  # 使用第一台攝影機
        if not self.capture.isOpened():
            logger.error("Failed to open camera.")
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.stream_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stream_height)
        logger.info("Camera started successfully.")
        return True

    def stop_camera(self):
        if self.capture:
            self.capture.release()
            logger.info("Camera stopped.")
        self.capture = None

    def start_stream(self):
        # 建立 MJPEG Streamer
        self.g_streamMgr.create_streamer(
            self.g_streamPort,
            stream_name=self.stream_name,
            stream_size=(self.stream_width, self.stream_height),
        )
        logger.info(
            f"MJPEG Stream Status: Running stream '{self.stream_name}' "
            f"{(self.stream_width, self.stream_height)} on port {self.g_streamPort}"
        )

    def stop_stream(self):
        # 停止 MJPEG Streamer
        self.g_streamMgr.stop_streamer(self.g_streamPort, self.stream_name)
        logger.info(
            f"MJPEG Stream '{self.stream_name}' on port {self.g_streamPort} stopped."
        )

    def stream_frames(self):
        # 持續從相機獲取影像並傳遞到 streamer
        self.listener_thread = threading.Thread(target=self.command_listener)
        self.listener_thread.start()
        
        while True:
            t00 = time()
            ret, img = self.capture.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break
            
            #print(img.shape)
            t0 = time()
            img, boxes, cls_ids = self.model.detect(img)
            if self.do_tracking:
                img = self.tracker(boxes, cls_ids, img)
            t1 = time()
            #print(t1 - t0)

            if self.show_roi:
                x0, y0, w, h = self.roi
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 5)
            
            # 保持影像格式與相機輸出一致（通常為 BGR）
            frame = img  # 直接使用相機的原始影像數據

            # 設置影像幀到 streamer
            self.g_streamMgr.set_frame_for_streamer(
                self.g_streamPort, self.stream_name, frame
            )
            t11 = time()

            if self.show_fps:
                h, w = frame.shape[:2]
                cv2.putText(frame, f'FPS: {1. / (t11 - t00):.1f}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                            thickness=2)

    def stop_all_thread(self):
        # 發送停止信號
        self.stop_event.set()
        # 等待執行緒結束
        #self.listener_thread.join()
        print("Listener thread has been stopped.")

    def command_listener(self):
        while not self.stop_event.set():
            cmd = input("Enter command (roi/no roi/count/no count/fps/no fps/reset): ").strip()
            if cmd[:3] == 'roi':
                self.show_roi = True
                if len(cmd) > 3:
                    cmd = list(map(int, cmd.split()[1:]))
                    self.roi = cmd
                    print(f'set ROI (x0, y0, w, h): {self.roi}')
            elif cmd == 'no roi':
                self.show_roi = False
            elif cmd == 'count':
                self.do_tracking = True
            elif cmd == 'no count':
                self.do_tracking = False
            elif cmd == 'fps':
                self.show_fps = True
            elif cmd == 'no fps':
                self.show_fps = False
            elif cmd == 'reset':
                self.tracker.reset()
            else:
                print(f"Unknown command: {cmd}")

    def cleanup(self):
        # 清理所有資源
        self.stop_camera()
        self.stop_stream()
        self.g_streamMgr.stop_all_streamers()
        cv2.destroyAllWindows()


# 主函數
def main():
    manager = StreamManager()
    try:
        if not manager.start_camera():
            return
        manager.start_stream()
        manager.stream_frames()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        exit()
        #manager.listener_thread.join()
    finally:
        manager.stop_all_thread()
        manager.cleanup()



if __name__ == "__main__":
    main()
