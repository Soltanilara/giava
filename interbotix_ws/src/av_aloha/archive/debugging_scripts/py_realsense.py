import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} device(s)\n")

for i, dev in enumerate(devices):

    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)

    print(f"Device {i}")
    print(f"Name:   {name}")
    print(f"Serial: {serial}")
    print()


# LEFT WRITS
# Device 2
# Name:   Intel RealSense D405
# Serial: 230322272239


# RIGHT WRIST
# Device 2
# Name:   Intel RealSense D405
# Serial: 230322270105


# TOP SCENE
# Device 0
# Name:   Intel RealSense D405
# Serial: 230322270396

# BOTTOM SCENE
# Device 0
# Name:   Intel RealSense D405
# Serial: 230322271312


# import pyrealsense2 as rs

# ctx = rs.context()

# for dev in ctx.query_devices():

#     name = dev.get_info(rs.camera_info.name)
#     serial = dev.get_info(rs.camera_info.serial_number)

#     print("\n" + "=" * 60)
#     print(f"{name} | {serial}")
#     print("=" * 60)

#     for sensor in dev.query_sensors():

#         print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")

#         profiles = sensor.get_stream_profiles()

#         for p in profiles:

#             vsp = p.as_video_stream_profile()

#             print(
#                 f"  {p.stream_type()} | "
#                 f"{vsp.width()}x{vsp.height()} | "
#                 f"{p.fps()} FPS | "
#                 f"{p.format()}"
#             )