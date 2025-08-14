import os
from mcap_protobuf.reader import read_protobuf_messages
import json


class Car_info:
    def __init__(self, pose_x, pose_y, pose_heading, car_gear, velocity):
        self.pose_x = pose_x
        self.pose_y = pose_y
        self.pose_heading = pose_heading
        self.car_gear = car_gear
        self.vel = velocity

    def __eq__(self, other):
        if not isinstance(other, Car_info):
            return False
        return (self.pose_x == other.pose_x and
                self.pose_y == other.pose_y and
                self.pose_heading == other.pose_heading and
                self.car_gear == other.car_gear)

    def __hash__(self):
        return hash((self.pose_x, self.pose_y, self.pose_heading, self.car_gear))

class SilDataReader:
    def __init__(self):
        self.data_dicts = {
            "parking_debug_out_dict": {},
            "par_state_machine_dict": {},
            "parking_zongmu_50ms": {},
            "par_perception": {},
        }

    def load(self, t_start: float, t_end: float, file_name: str) -> int:
        if not os.path.exists(file_name):
            print(f"File {file_name} does not exist.")
            return 1

        start_time_ns = None
        for message in read_protobuf_messages(file_name):
            if start_time_ns is None:
                start_time_ns = message.log_time_ns
            timestamp_ns = message.log_time_ns

            if timestamp_ns < start_time_ns + int(t_start * 1e9):
                continue
            if timestamp_ns > start_time_ns + int(t_end * 1e9):
                break

            self.store_message(message.topic, (timestamp_ns - start_time_ns) // 1e6, message.proto_msg)

        self.print_data_counts()
        return 0

    def store_message(self, message_topic: str, elapse_time_ms: int, message):
        if message_topic == "function/parking/debug_out":
            self.data_dicts["parking_debug_out_dict"][elapse_time_ms] = message
        elif message_topic == "function/parking/par_perception":
            self.data_dicts["par_perception"][elapse_time_ms] = message
        elif message_topic == "function/parking/parking_zongmu_50ms":
            self.data_dicts["parking_zongmu_50ms"][elapse_time_ms] = message
        elif message_topic == "function/parking/par_state_machine":
            self.data_dicts["par_state_machine_dict"][elapse_time_ms] = message


    def print_data_counts(self):
        for key, value in self.data_dicts.items():
            print(f"{key} count: {len(value)}")

    def read_data_by_time(self, time: float, data_dict: dict) -> any:
        ms_time = int(time * 1e3)
        min_ms_diff = float("inf")
        pick_key = None
        for key in data_dict.keys():
            ms_diff = abs(ms_time - key)
            if ms_diff < min_ms_diff:
                min_ms_diff = ms_diff
                pick_key = key
        return data_dict.get(pick_key)
    
    def read_data_by_time_return_pairs(self, time: float, data_dict: dict) -> any:
        ms_time = int(time * 1e3)
        min_ms_diff = float("inf")
        pick_key = None
        for key in data_dict.keys():
            ms_diff = abs(ms_time - key)
            if ms_diff < min_ms_diff:
                min_ms_diff = ms_diff
                pick_key = key
        if pick_key is not None:
            car_info_frame = Car_info(data_dict[pick_key].ctrl_debug.ctrl_adapter_in.stCurrentPosf32X_Coor, data_dict[pick_key].ctrl_debug.ctrl_adapter_in.stCurrentPosf32Y_Coor,
                                      data_dict[pick_key].ctrl_debug.ctrl_adapter_in.stCurrentPosf32Theta, data_dict[
                                          pick_key].ctrl_debug.ctrl_adapter_in.ChassisToCtrlInfoeVehicleGear, data_dict[pick_key].ctrl_debug.ctrl_adapter_in.ChassisToCtrlInfof32VehicleSpeed
)
            return pick_key, car_info_frame
        else:
            return None, None

    def get_car_info_by_time(self, time: float) -> any:
        key, value = self.read_data_by_time_return_pairs(time, self.data_dicts["parking_debug_out_dict"])
        return key, value
    
    def get_cluster_info_by_time(self, time: float) -> any:
        value = self.read_data_by_time(time, self.data_dicts["parking_zongmu_50ms"])
        return value
    
    def get_state_machine_info_by_time(self, time: float) -> any:
        data = self.read_data_by_time(time, self.data_dicts["par_state_machine_dict"])
        return data
    

class ProcessMcapData:
    def __init__(self, file_name, output_folder_path, t_start=0.0, t_end=59.5, time_step=0.5):
        self.file_name = file_name
        self.output_folder_path = output_folder_path
        self.t_start = t_start
        self.t_end = t_end
        self.time_step = time_step
        self.reader = SilDataReader()
        self.car_info_data = {}
        self.car_info_data_unique = {}
        self.measurements = None
        self.clusters = None
        self.pillars = None

    def save_measurements(self, measurements, cnt, measurement_tag="measurements"):
        measurements_path = os.path.join(self.output_folder_path, measurement_tag)
        os.makedirs(measurements_path, exist_ok=True)
        measurements_filename = os.path.join(measurements_path, "{:04d}.json".format(cnt))
        if measurements == None:
            return
        with open(measurements_filename, 'w') as json_file:
            json.dump(measurements, json_file, indent=4)


    def parser_measurements_msg(self, msg):
        pose_ret = {
            'x':msg.pose_x,
            'y':msg.pose_y,
            'yaw':msg.pose_heading,
            'velocity':msg.vel,
            'gear':msg.car_gear,
        }

        return pose_ret
    
    def parser_clusters_msg(self, time):
        my_dict_template = {
            "id": None,
            "p0": {
                "x": None,
                "y": None
            },
            "p1": {
                "x": None,
                "y": None
            }
        }
        clusters_list =[]
        clusters_msg = self.reader.get_cluster_info_by_time(time)
        for clusters in clusters_msg.ups_cluster_info.clusters:
            for index, line in enumerate(clusters.lines):
                if(line.height != 2):
                    my_dict = my_dict_template.copy()
                    my_dict["id"] = clusters.id
                    my_dict["p0"]["x"] = line.p0.x
                    my_dict["p0"]["y"] = line.p0.y
                    my_dict["p1"]["x"] = line.p1.x
                    my_dict["p1"]["y"] = line.p1.y
                    clusters_list.append(my_dict)
        return clusters_list

    def process_data(self):
        self.reader.load(self.t_start, self.t_end, self.file_name)
        current_time=  self.t_start
        while current_time < self.t_end:
            park_state  = self.reader.get_state_machine_info_by_time(current_time)
            if park_state.feature_status == 4:
                car_key, car_value = self.reader.get_car_info_by_time(current_time)
                self.car_info_data[car_key] = car_value
            current_time += self.time_step
        seen_values = set()  # 用于记录已经出现过的值
        for key, value in self.car_info_data.items():
            if value not in seen_values:
                seen_values.add(value)
                self.car_info_data_unique[key] = value
        num_cnt = 0
        for key, value in self.car_info_data_unique.items():
            self.measurements = self.parser_measurements_msg(value)
            self.save_measurements(self.measurements, num_cnt)
            self.clusters = self.parser_clusters_msg(key)
            self.save_measurements(self.clusters, num_cnt,measurement_tag="clusters")
            num_cnt += 1
        self.save_measurements(self.measurements, 1, measurement_tag="parking_goal")

        
        


        
    
def main():
    t_start = 0.0
    t_end = 59.5
    time_step = 0.5
    file_name="data/20250518T103500.mcap"
    output_folder_path = "./e2e_dataset"
    data_process = ProcessMcapData(file_name, output_folder_path, t_start, t_end, time_step)
    data_process.process_data()


# 示例用法
if __name__ == "__main__":
    main()