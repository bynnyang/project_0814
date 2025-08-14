import sys
import csv

from mcap_protobuf.reader import read_protobuf_messages


def main():

    for msg in read_protobuf_messages("data/20250518T103500.mcap"):
           # print(f"{msg.topic}: {msg.proto_msg}")
        if msg.topic == "function/vehicle_in/mcu_data_10ms":
            print(f"mcu ts {msg.proto_msg.publish_ptp_ts}")
        if msg.topic == "function/parking/par_planning":
            print(f"planning ts {msg.proto_msg.publish_ptp_ts}")

        

if __name__ == "__main__":
    main()