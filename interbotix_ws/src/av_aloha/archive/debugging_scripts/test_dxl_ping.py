# from dynamixel_sdk import *

# PORT = "/dev/ttyUSB1"

# BAUDS = [
#     57600,
#     # 115200,
#     # 1000000,
#     # 2000000,
#     # 3000000,
#     # 4000000,
# ]

# portHandler = PortHandler(PORT)
# packetHandler = PacketHandler(2.0)

# if not portHandler.openPort():
#     print("Couldn't open port")
#     quit()

# for baud in BAUDS:
#     portHandler.setBaudRate(baud)
#     print(f"\nTrying baud {baud}")

#     model, result, error = packetHandler.ping(portHandler, 1)

#     print("result =", result)
#     print("error  =", error)
#     print("model  =", model)

# portHandler.closePort()






# CHANGE ID OF DYNAMIXEL MOTOR
# from dynamixel_sdk import *

# PORT = "/dev/ttyUSB1"
# BAUD = 57600

# OLD_ID = 1
# NEW_ID = 9

# ADDR_TORQUE_ENABLE = 64
# ADDR_ID = 7

# portHandler = PortHandler(PORT)
# packetHandler = PacketHandler(2.0)

# assert portHandler.openPort()
# assert portHandler.setBaudRate(BAUD)

# # Disable torque before writing EEPROM
# dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
#     portHandler,
#     OLD_ID,
#     ADDR_TORQUE_ENABLE,
#     0,
# )

# print("Disable torque:", dxl_comm_result, dxl_error)

# dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
#     portHandler,
#     OLD_ID,
#     ADDR_ID,
#     NEW_ID,
# )

# print("Change ID:", dxl_comm_result, dxl_error)

# portHandler.closePort()


# # PING BOTH ID'S 1 and 9 to verify that the ID has been changed

from dynamixel_sdk import *

PORT = "/dev/ttyUSB1"
BAUD = 1000000

portHandler = PortHandler(PORT)
packetHandler = PacketHandler(2.0)

assert portHandler.openPort()
assert portHandler.setBaudRate(BAUD)

for dxl_id in [9]:
    model, result, error = packetHandler.ping(portHandler, dxl_id)
    print(f"ID {dxl_id}")
    print("  result =", result)
    print("  error  =", error)
    print("  model  =", model)
    print()

portHandler.closePort()





# # CHANGE BAUD RATE
# from dynamixel_sdk import *

# PORT = "/dev/ttyUSB1"
# BAUD = 57600

# DXL_ID = 9

# ADDR_TORQUE_ENABLE = 64
# ADDR_BAUD_RATE = 8

# # 3 = 1 Mbps
# NEW_BAUD = 3

# portHandler = PortHandler(PORT)
# packetHandler = PacketHandler(2.0)

# assert portHandler.openPort()
# assert portHandler.setBaudRate(BAUD)

# # Disable torque
# comm_result, error = packetHandler.write1ByteTxRx(
#     portHandler,
#     DXL_ID,
#     ADDR_TORQUE_ENABLE,
#     0,
# )

# print("Disable torque:", comm_result, error)

# # Change baud
# comm_result, error = packetHandler.write1ByteTxRx(
#     portHandler,
#     DXL_ID,
#     ADDR_BAUD_RATE,
#     NEW_BAUD,
# )

# print("Change baud:", comm_result, error)

# portHandler.closePort()