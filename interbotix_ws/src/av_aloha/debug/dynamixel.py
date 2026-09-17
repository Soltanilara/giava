import time
from dynamixel_sdk import *

PORT = "/dev/ttyDXL_puppet_right"   # middle arm port
BAUD = 1000000

portHandler = PortHandler(PORT)
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(BAUD)

for dxl_id in range(20):
    model, comm_result, error = packetHandler.ping(portHandler, dxl_id)

    if comm_result == COMM_SUCCESS:
        print(f"Found ID {dxl_id}, model {model}")

portHandler.closePort()

ADDR_PRESENT_POSITION = 132

ID=9      # replace with your new motor ID

port=PortHandler(PORT)
packet=PacketHandler(2.0)

port.openPort()
port.setBaudRate(BAUD)

for i in range(5):

    model, result, error = packet.ping(port, ID)

    if result == COMM_SUCCESS:
        print("SUCCESS", model)

    else:
        print("FAIL", packet.getTxRxResult(result))

    time.sleep(.2)

    ADDR_PRESENT_POSITION = 132

for i in range(10):

    pos, result, error = packet.read4ByteTxRx(
        port,
        ID,
        ADDR_PRESENT_POSITION
    )

    print(result, pos)