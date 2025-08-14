import sys
import csv

from mcap_protobuf.reader import read_protobuf_messages

def main():
    csv_file_path = "apa_data.csv"

    # 定义CSV文件列名
    csv_columns = [
        "timestamp",
        "DRpose_x",
        "DRpose_y",
        "DRpose_z",
        "DRpose_yaw",
        "DRpose_pitch",
        "DRpose_roll",
        "DRpose_vx",
        "DRpose_vy",
        "DRpose_vz",
        "DRpose_ax",
        "DRpose_ay",
        "DRpose_az",
        "DRpose_yawrate"
    ]

    # 创建并写入CSV文件标题行
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for msg in read_protobuf_messages("data/20250518T103500.mcap"):
            # print(f"{msg.topic}: {msg.proto_msg}")
            if msg.topic == "function/vehicle_in/mcu_data_10ms":
                print(f"mcu ts {msg.proto_msg.publish_ptp_ts}")
            if msg.topic == "function/parking/par_planning":
                print(f"planning ts {msg.proto_msg.publish_ptp_ts}")

        

if __name__ == "__main__":
    main()