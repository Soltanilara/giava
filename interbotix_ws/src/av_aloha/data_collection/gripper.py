"""
Code to configure and control Interbotix grippers, including:
 - Register configuration
 - Current-limit setup
 - Trigger-based gripper command
"""
import os

try:
    import rospy
except ImportError:
    rospy = None

try:
    from interbotix_xs_msgs.msg import JointSingleCommand
    from interbotix_xs_msgs.srv import (
        RegisterValues,
        RegisterValuesRequest,
    )
except ImportError:
    JointSingleCommand = None
    RegisterValues = None
    RegisterValuesRequest = None

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.5
GRIPPER_CURRENT_LIMIT = 100

# Reboot gripper motor.
def reboot_gripper(bot, sleep_time=1.0):
    if rospy is None:
        raise ImportError("rospy is required to reboot the gripper.")
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    rospy.sleep(sleep_time)

# Set a Dynamixel register for a single motor.
def set_register(robot_name, motor_name, reg_name, value):
    if rospy is None or RegisterValues is None or RegisterValuesRequest is None:
        raise ImportError("rospy and interbotix_xs_msgs are required to set gripper registers.")
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value
    return srv(req)

# Configure gripper current limit and operating mode.
def configure_gripper(bot, robot_name):
    if rospy is None:
        raise ImportError("rospy is required to configure the gripper.")
    bot.dxl.robot_torque_enable("single", "gripper", False)
    rospy.sleep(0.2)
    
    set_register(robot_name, "gripper", "Current_Limit", GRIPPER_CURRENT_LIMIT)
    rospy.sleep(0.2)

    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    bot.dxl.robot_torque_enable("single", "gripper", True)
    rospy.sleep(0.5)

# Open or close the gripper based on trigger state.
def update_gripper(bot, trigger_pressed, close_position=GRIPPER_CLOSED, open_position=GRIPPER_OPEN):
    position = close_position if trigger_pressed else open_position
    command_gripper(bot, position)
    return position


## Optional analog gripper: trigger deflection (0..1) -> aperture, instead of
## the default binary threshold.  Deadzone ignores a resting finger, everything
## above the latch is full close (a firm grip needs no precision), EMA kills
## trigger chatter.  Never used in a recorded session -- every dataset to date
## is GIAVA_GRIPPER_MODE=binary.
GRIPPER_MODE = os.environ.get("GIAVA_GRIPPER_MODE", "binary").strip().lower()
GRIPPER_ANALOG_DEADZONE = float(os.environ.get("GIAVA_GRIPPER_DEADZONE", "0.08"))
GRIPPER_ANALOG_LATCH = float(os.environ.get("GIAVA_GRIPPER_LATCH", "0.85"))
GRIPPER_ANALOG_ALPHA = float(os.environ.get("GIAVA_GRIPPER_ALPHA", "0.4"))

## Smoothed command per gripper, keyed by id(bot).
_analog_cmd = {}


def analog_trigger_to_position(trigger_value,
                               close_position=GRIPPER_CLOSED,
                               open_position=GRIPPER_OPEN):
    """Map raw trigger deflection (0..1) to a gripper position command."""
    t = min(max(float(trigger_value), 0.0), 1.0)
    if t <= GRIPPER_ANALOG_DEADZONE:
        return open_position
    if t >= GRIPPER_ANALOG_LATCH:
        return close_position
    frac = (t - GRIPPER_ANALOG_DEADZONE) / (GRIPPER_ANALOG_LATCH - GRIPPER_ANALOG_DEADZONE)
    return open_position + frac * (close_position - open_position)


def update_gripper_from_trigger(bot, trigger_value,
                                close_position=GRIPPER_CLOSED,
                                open_position=GRIPPER_OPEN):
    """One gripper tick from a raw analog trigger value (0..1).

    Dispatches on GIAVA_GRIPPER_MODE so existing sessions keep the binary
    behaviour bit-for-bit unless analog is asked for explicitly.
    """
    if GRIPPER_MODE != "analog":
        return update_gripper(bot, trigger_value > 0,
                              close_position=close_position,
                              open_position=open_position)

    target = analog_trigger_to_position(trigger_value, close_position, open_position)
    key = id(bot)
    prev = _analog_cmd.get(key, open_position)
    position = GRIPPER_ANALOG_ALPHA * target + (1.0 - GRIPPER_ANALOG_ALPHA) * prev
    ## Snap when within a whisker of either end so "fully open"/"fully closed"
    ## are actually reached instead of asymptotically approached.
    if abs(position - open_position) < 0.01:
        position = open_position
    elif abs(position - close_position) < 0.01:
        position = close_position
    _analog_cmd[key] = position
    command_gripper(bot, position)
    return position

# Directly send a gripper command.
def command_gripper(bot, position):
    if JointSingleCommand is None:
        raise ImportError("interbotix_xs_msgs is required to command the gripper.")
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = position
    bot.gripper.core.pub_single.publish(cmd)

# Open gripper
def open_gripper(bot):
    command_gripper(bot, GRIPPER_OPEN)

# Close gripper
def close_gripper(bot):
    command_gripper(bot, GRIPPER_CLOSED)

# Test gripper by sending open and close commands with a delay.
def test_gripper(bot, wait_time=1.0):
    if rospy is None:
        raise ImportError("rospy is required to test the gripper.")
    open_gripper(bot)
    rospy.sleep(wait_time)

    close_gripper(bot)
    rospy.sleep(wait_time)

    open_gripper(bot)
    rospy.sleep(wait_time)

# Read a register value from the gripper motor.
def get_register(robot_name, motor_name, reg_name):
    if rospy is None or RegisterValues is None or RegisterValuesRequest is None:
        raise ImportError("rospy and interbotix_xs_msgs are required to read gripper registers.")
    service_name = f"/{robot_name}/get_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name

    resp = srv(req)
    return list(resp.values)

# Print current gripper register values for debugging.
def print_gripper_registers(robot_name):
    try:
        current_limit = get_register(robot_name, "gripper", "Current_Limit")
        operating_mode = get_register(robot_name, "gripper", "Operating_Mode")
        torque_enable = get_register(robot_name, "gripper", "Torque_Enable")

        print(f"[{robot_name}] Current_Limit: {current_limit}")
        print(f"[{robot_name}] Operating_Mode: {operating_mode}")
        print(f"[{robot_name}] Torque_Enable: {torque_enable}")

    except Exception as e:
        print(f"[{robot_name}] Failed reading registers: {e}")
