import depthai as dai

pipeline = dai.Pipeline()

# Create device and output information
with dai.Device() as device:
    device.startPipeline(pipeline)

    print('MxId:', device.getDeviceInfo().getMxId())
    print('USB speed:', device.getUsbSpeed())
    print('Connected cameras:', device.getConnectedCameras())

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("input_name")

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("output_name")

    input_q = device.getInputQueue("input_name", maxSize=4, blocking=False)
    output_q = device.getOutputQueue("output_name", maxSize=4, blocking=False)

    while True:
        output_q.get()

        cfg = dai.ImageManipConfig()
        input_q.send(cfg)