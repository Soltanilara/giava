import depthai as dai

with dai.Device() as device:
    # With current setup, should print out Connected cameras: [<CameraBoardSocket.CAM_B: 1>, <CameraBoardSocket.CAM_C: 2>]
    print("Connected cameras: ", device.getConnectedCameras())