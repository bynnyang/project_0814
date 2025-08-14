import os
import time
from typing import Dict, Any
import mcap
from google.protobuf.message import Message
from google.protobuf.descriptor import Descriptor

# 假设 protobuf 文件已经编译为 Python 模块
from data_reader_pb2 import (
    EnhanceLoc,
    HDMapInfo,
    FyLine,
    FyObstacles,
    FyObject,
    FyFreespace,
    FyAebFlag,
    FyOdometry,
    VEH10ms,
    VEH50ms,
    IMU10msJ5,
    dgbAebOut,
    PncDebugOut,
)

class SilDataReader:
    def __init__(self):
        self.reader = None
        self.data_dicts = {
            "enh_loc_dict": {},
            "hdmap_dict": {},
            "lane_line_dict": {},
            "obstacles_dict": {},
            "objects_light_dict": {},
            "objects_bbox_dict": {},
            "objects_sign_dict": {},
            "freespace_dict": {},
            "aeb_flag_dict": {},
            "odometry_dict": {},
            "mcu_data_10ms_dict": {},
            "mcu_data_50ms_dict": {},
            "mcu_data_imu10msj5_dict": {},
            "dgb_aeb_strtg_dict": {},
            "np_debug_20ms_dict": {},
            "fct_debug_100ms_dict": {},
        }

    def load(self, t_start: float, t_end: float, file_name: str) -> int:
        if not os.path.exists(file_name):
            print(f"File {file_name} does not exist.")
            return 1

        self.reader = mcap.Reader(file_name)
        start_time_ns = self.reader.get_summary().message_start_time
        t_start_ns = start_time_ns + int(t_start * 1e9)
        t_end_ns = start_time_ns + int(t_end * 1e9)

        for record in self.reader.records():
            if record.channel.schema.encoding != "protobuf":
                print(f"Expected message encoding 'protobuf', got {record.channel.schema.encoding}")
                continue

            if record.message.log_time < t_start_ns:
                continue
            elif record.message.log_time > t_end_ns:
                break

            message_class = self.get_message_class(record.channel.schema.name)
            if not message_class:
                continue

            message = message_class()
            message.ParseFromString(record.message.data)

            elapse_time_ms = (record.message.log_time - start_time_ns) // 1e6
            self.store_message(record.channel.schema.name, elapse_time_ms, message)

        self.print_data_counts()
        return 0

    def get_message_class(self, schema_name: str) -> Message:
        message_classes = {
            "nio.ad.messages.EnhanceLoc": EnhanceLoc,
            "nio.ad.messages.HDMapInfo": HDMapInfo,
            "fy.ad.perception.FyLine": FyLine,
            "fy.ad.perception.FyObstacles": FyObstacles,
            "fy.ad.perception.FyObject": FyObject,
            "fy.ad.perception.FyFreespace": FyFreespace,
            "fy.ad.perception.FyAebFlag": FyAebFlag,
            "fy.ad.perception.FyOdometry": FyOdometry,
            "nio.ad.messages.VEH10ms": VEH10ms,
            "nio.ad.messages.VEH50ms": VEH50ms,
            "nio.ad.messages.IMU10msJ5": IMU10msJ5,
            "nio.ad.messages.debug.dgbAebOut": dgbAebOut,
            "nio.ad.messages.debug.PncDebugOut": PncDebugOut,
        }
        return message_classes.get(schema_name)

    def store_message(self, schema_name: str, elapse_time_ms: int, message: Message):
        if schema_name == "nio.ad.messages.EnhanceLoc":
            self.data_dicts["enh_loc_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.HDMapInfo":
            self.data_dicts["hdmap_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyLine":
            self.data_dicts["lane_line_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyObstacles":
            self.data_dicts["obstacles_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyObject":
            if record.channel.topic == "perception/object_light-6v":
                self.data_dicts["objects_light_dict"][elapse_time_ms] = message
            elif record.channel.topic == "perception/object_parsing_bbox-6v":
                self.data_dicts["objects_bbox_dict"][elapse_time_ms] = message
            elif record.channel.topic == "perception/object_sign-6v":
                self.data_dicts["objects_sign_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyFreespace":
            self.data_dicts["freespace_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyAebFlag":
            self.data_dicts["aeb_flag_dict"][elapse_time_ms] = message
        elif schema_name == "fy.ad.perception.FyOdometry":
            self.data_dicts["odometry_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.VEH10ms":
            self.data_dicts["mcu_data_10ms_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.VEH50ms":
            self.data_dicts["mcu_data_50ms_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.IMU10msJ5":
            self.data_dicts["mcu_data_imu10msj5_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.debug.dgbAebOut":
            self.data_dicts["dgb_aeb_strtg_dict"][elapse_time_ms] = message
        elif schema_name == "nio.ad.messages.debug.PncDebugOut":
            if record.channel.topic == "function/fct/np_debug_out":
                self.data_dicts["np_debug_20ms_dict"][elapse_time_ms] = message
            elif record.channel.topic == "function/fct/fct_debug_out":
                self.data_dicts["fct_debug_100ms_dict"][elapse_time_ms] = message

    def print_data_counts(self):
        for key, value in self.data_dicts.items():
            print(f"{key} count: {len(value)}")

    def read_data_by_time(self, time: float, data_dict: Dict[int, Any]) -> Any:
        ms_time = int(time * 1e3)
        min_ms_diff = float("inf")
        pick_key = None
        for key in data_dict.keys():
            ms_diff = ms_time - key
            if ms_diff < 0:
                continue
            if ms_diff < min_ms_diff:
                min_ms_diff = ms_diff
                pick_key = key
        return data_dict.get(pick_key)

    def get_car_info_by_time(self, time: float) -> Any:
        data = self.read_data_by_time(time, self.data_dicts["dgb_aeb_strtg_dict"])
        return data.carinfo if data else None

    def close(self):
        if self.reader:
            self.reader.close()
            self.reader = None

# 示例用法
if __name__ == "__main__":
    reader = SilDataReader()
    try:
        reader.load(t_start=0.0, t_end=10.0, file_name="data/20250518T103500.mcap")
        car_info = reader.get_car_info_by_time(time=5.0)
        print(car_info)
    finally:
        reader.close()