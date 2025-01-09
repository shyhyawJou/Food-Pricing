import csv
import cv2
import requests
from datetime import datetime
from loguru import logger
from pathlib import Path
# import pymssql
import numpy as np
from mjpeg_streamer import MjpegServer, Stream
from threading import Thread
import time
import threading
import multiprocessing


# class DataRecorder:
#     def __init__(self, filename) -> None:
#         self.filename = filename
#         self._createFile()
#
#     # --------------------------------------------------------------------------
#     def _createFile(self):
#         try:
#             with open(self.filename, 'x', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(
#                     ['timestamp', 'duration', 'project_name', 'task_name', 'prediction', 'ground_truth', 'score'])
#         except FileExistsError as e:
#             pass
#
#     # --------------------------------------------------------------------------
#     def recorder_data(self, duration, proj, task, pred, truth, score):
#         with open(self.filename, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             writer.writerow([timestamp, duration, proj, task, pred, truth, score])


# ?====[MSSQL]==================================================================
# class MsSqlClient:
#     def __init__(self, host, username, password, database='BDDB'):
#         self._host = host
#         self._username = username
#         self._password = password
#         self._database = database
#         self._conn = None
#
#     # ==========================================================================
#     def connect(self):
#         try:
#             self._conn = pymssql.connect(
#                 host=self._host,
#                 user=self._username,
#                 password=self._password,
#                 database=self._database
#             )
#         except pymssql.DatabaseError as e:
#             logger.error(f"Error connecting to database: {e}")
#
#     # --------------------------------------------------------------------------
#     def disconnect(self):
#         if self._conn:
#             self._conn.close()
#
#     # --------------------------------------------------------------------------
#     def _execute(self, sql, params=None, fetch=False):
#         try:
#             with self._conn.cursor(as_dict=True) as cursor:
#                 cursor.execute(sql, params or [])
#                 if fetch:
#                     return cursor.fetchall()
#                 else:
#                     self._conn.commit()
#         except pymssql.DatabaseError as e:
#             logger.error(f"Error executing query: {e}")
#
#     # --------------------------------------------------------------------------
#     def count(self, table_name):
#         sql = f"SELECT COUNT(*) AS quantity FROM {table_name}"
#         result = self._execute(sql, fetch=True)
#         return result[0]['quantity'] if result else 0
#
#     # --------------------------------------------------------------------------
#     def create_table(self, name):
#         sql = f"""
#             CREATE TABLE {name} (
#                 line NVARCHAR(32) NOT NULL,
#                 station NVARCHAR(32) NOT NULL,
#                 model NVARCHAR(32) NOT NULL,
#                 test_time DATETIME DEFAULT (SYSDATETIME()) NOT NULL,
#                 test_result NVARCHAR(MAX) NULL,
#                 final_result NCHAR(4) NOT NULL,
#                 cycle_time REAL NOT NULL,
#                 other NVARCHAR(MAX) NULL
#             ) ON [PRIMARY]
#         """
#         self._execute(sql)
#
#     # --------------------------------------------------------------------------
#     def delete_table(self, name):
#         sql = f"DROP TABLE {name}"
#         self._execute(sql)
#
#     # --------------------------------------------------------------------------
#     def get_table_columns(self, table_name):
#         # * 獲取表的欄位名稱，用於檢查資料是否符合
#         sql = f"""
#             SELECT COLUMN_NAME
#             FROM INFORMATION_SCHEMA.COLUMNS
#             WHERE TABLE_NAME = '{table_name}'
#         """
#         result = self._execute(sql, fetch=True)
#         return set([row['COLUMN_NAME'] for row in result])
#
#     # --------------------------------------------------------------------------
#     def insert(self, record, table_name):
#         # * 獲取表的欄位名稱
#         table_columns = self.get_table_columns(table_name)
#
#         # * 過濾出 record 中與表欄位匹配的 key 和 value
#         valid_record = {key: value for key, value in record.items() if key in table_columns}
#
#         # * 如果沒有有效的欄位，則不執行插入
#         if not valid_record:
#             logger.error("No valid fields in the record to insert.")
#             return
#
#         fields = ', '.join(valid_record.keys())
#         placeholders = ', '.join(['%s'] * len(valid_record))
#         values = tuple(valid_record.values())
#
#         sql = f"INSERT INTO {table_name} ({fields}) VALUES ({placeholders})"
#         try:
#             self._execute(sql, params=values)
#             logger.info("Record inserted successfully.")
#         except Exception as e:
#             logger.error(f"Error inserting record: {e}")
#
#     # --------------------------------------------------------------------------
#     def query(self, table_name):
#         sql = f"SELECT * FROM {table_name}"
#         result = self._execute(sql, fetch=True)
#
#         # * 格式化 `test_time` 為字串
#         for row in result:
#             row['test_time'] = str(row['test_time'])
#         return result


# ?====[MJpeg Streamer]=========================================================
class MJPEGStreamerProcess(multiprocessing.Process):
    def __init__(self, host='0.0.0.0', port=5000, stream_name='stream', stream_size=(640, 480), stream_quality=50,
                 fps=30):
        super(MJPEGStreamerProcess, self).__init__()
        self.host = host
        self.port = port
        self.stream_name = stream_name
        self.stream_size = stream_size
        self.stream_quality = stream_quality
        self.fps = fps
        self.server = None
        self.stream = None
        self.frame_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

    # --------------------------------------------------------------------------
    def run(self):
        # *啟動MJPEG串流服務
        logger.info(f"Starting streaming server for stream '{self.stream_name}' on port {self.port}...")
        self.stream = Stream(self.stream_name, size=self.stream_size, quality=self.stream_quality, fps=self.fps)
        self.server = MjpegServer(self.host, self.port)
        self.server.add_stream(self.stream)
        self.server.start()

        # * 伺服器進程持續運行，直到接收到停止信號
        while not self.stop_event.is_set():
            try:
                # * 從 Queue 中獲取影像幀並更新到 Stream
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.stream.set_frame(frame)
            except Exception as e:
                logger.error(f"Error in processing frame: {e}")
            time.sleep(0.02)
        # * 收到停止信號後停止伺服器
        logger.info(f"Stopping streaming server for stream '{self.stream_name}' on port {self.port}...")
        self.server.stop()
        logger.info(f"Server for stream '{self.stream_name}' on port {self.port} stopped successfully.")

    # --------------------------------------------------------------------------
    def stop(self):
        # *停止MJPEG串流服務
        logger.info(f"Stopping request received for stream '{self.stream_name}' on port {self.port}...")
        self.stop_event.set()

    # --------------------------------------------------------------------------
    def send_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)


# *=============================================================================
class MJPEGStreamerManager:
    def __init__(self):
        self.streamers = {}  ## key: (port, stream_name), value: MJPEGStreamer

    # --------------------------------------------------------------------------
    def create_streamer(self, port, stream_name='stream', stream_size=(640, 480), stream_quality=50, fps=30):
        # *創建並啟動一個新的MJPEG串流伺服器,基於port和stream_name
        key = (port, stream_name)
        if key in self.streamers:
            logger.error(f"Server for stream '{stream_name}' on port {port} is already running.")
            return

        streamer = MJPEGStreamerProcess(port=port, stream_name=stream_name, stream_size=stream_size,
                                        stream_quality=stream_quality, fps=fps)
        streamer.start()
        self.streamers[key] = streamer
        logger.info(f"Server for stream '{stream_name}' started on port {port}.")

    # --------------------------------------------------------------------------
    def stop_streamer(self, port, stream_name):
        # *停止指定埠和流名稱的MJPEG串流伺服器
        key = (port, stream_name)
        if key in self.streamers:
            self.streamers[key].stop()
            self.streamers[key].join()
            del self.streamers[key]
            logger.info(f"Server for stream '{stream_name}' on port {port} stopped.")
        else:
            logger.error(f"No server for stream '{stream_name}' running on port {port}.")

    # --------------------------------------------------------------------------
    def set_frame_for_streamer(self, port, stream_name, frame):
        # *為指定的流設置影像幀
        key = (port, stream_name)
        if key in self.streamers:
            self.streamers[key].send_frame(frame)
        else:
            logger.error(f"No server for stream '{stream_name}' running on port {port}.")

    # --------------------------------------------------------------------------
    def stop_all_streamers(self):
        # *停止所有運行中的串流伺服器
        for (port, stream_name), streamer in self.streamers.items():
            streamer.stop()
            streamer.join()
        self.streamers.clear()
        logger.info("All servers stopped.")


# ?====[Misc]===================================================================
# def drawText(img, text,
#              font=cv2.FONT_HERSHEY_DUPLEX,
#              pos=(0, 0),
#              font_scale=0.7,
#              font_thickness=1,
#              line_type=cv2.LINE_AA,
#              text_color=(0, 255, 0),
#              text_color_bg=(63, 68, 67)):
#     x, y = pos
#     text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
#     text_w, text_h = text_size
#     cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
#     cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)),
#                 font, font_scale, text_color, font_thickness, line_type)
#     return text_size


# ------------------------------------------------------------------------------
def drawRect(img, rect, color=(217, 214, 214), thickness=1):
    # * 將十六進位數字轉換成 RGB tuple
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')  ## 移除開頭的 '#'
        rgb_tuple = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb_tuple

    if isinstance(color, str):
        color = hex_to_rgb(color)

    # * 將正規化的座標轉換為實際像素座標
    height, width, _ = img.shape
    pt1 = (int(rect[0] * width), int(rect[1] * height))
    pt2 = (int((rect[0] + rect[2]) * width), int((rect[1] + rect[3]) * height))

    cv2.rectangle(img, pt1, pt2, color, thickness)


# ------------------------------------------------------------------------------
def convert2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------------
def writeVideo(frame_queue, fps, width, height, save_path):
    # * 使用 'mp4v' 編碼器來輸出 MP4 格式的影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    # logger.debug(f"Writing {save_path} from queue... fps {fps:.2f}")

    while not frame_queue.empty():
        frame = frame_queue.get()
        # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.write(frame)
    out.release()


# ------------------------------------------------------------------------------
def sendRequest(img):
    try:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(img, (640, 480))
        response = requests.post(
            'http://10.46.13.24:8080/upload_frame',
            # files={'frame': jpeg.tobytes()},
            data=frame.tobytes(),  # Sending raw numpy data as bytes
            headers={'Content-Type': 'application/octet-stream'},  # Inform the server it's binary data
            timeout=3  # Set a timeout for network performance
        )
        if response.status_code != 200:
            logger.error(f"Failed to send frame: {response.json()}")
    except Exception as e:
        logger.error(f"Error sending frame: {e}")


# ------------------------------------------------------------------------------
def readImage(filepath: str):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------------
def writeImage(img, filepath: str):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))