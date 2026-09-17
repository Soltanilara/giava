#!/usr/bin/env python3
import argparse
import math
import rospy
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest

DEFAULT_MIN_CMD = -110.0
DEFAULT_MAX_CMD = 110.0
DEFAULT_CURRENT_LIMIT = 200
DEFAULT_WAIT_TIME = 1.0


def get_register(robot_name, motor_name, reg_name):
    service_name = f"/{robot_name}/get_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name

    resp = srv(req)
    return list(resp.values)


def set_register(robot_name, motor_name, reg_name, value):
    service_name = f"/{robot_name}/set_motor_registers"
    rospy.wait_for_service(service_name)
    srv = rospy.ServiceProxy(service_name, RegisterValues)

    req = RegisterValuesRequest()
    req.cmd_type = "single"
    req.name = motor_name
    req.reg = reg_name
    req.value = value

    return srv(req)



def print_gripper_registers(robot_name):
    current_limit = get_register(robot_name, "gripper", "Current_Limit")
    operating_mode = get_register(robot_name, "gripper", "Operating_Mode")
    torque_enable = get_register(robot_name, "gripper", "Torque_Enable")
    print(f"[{robot_name}] Current_Limit={current_limit} Operating_Mode={operating_mode} Torque_Enable={torque_enable}")

def setup_gripper(bot, robot_name, current_limit):
    print(f"\n=== Setting up {robot_name} ===")
    try:
        bot.dxl.robot_torque_enable("single", "gripper", False)
        rospy.sleep(0.2)

        set_register(robot_name, "gripper", "Current_Limit", int(current_limit))
        rospy.sleep(0.2)

        bot.dxl.robot_set_operating_modes("single", "gripper", "linear_position")
        rospy.sleep(0.2)

        bot.dxl.robot_torque_enable("single", "gripper", True)
        rospy.sleep(0.5)

        print_gripper_registers(robot_name)
    except Exception as e:
        print(f"[{robot_name}] setup warning: {e}")

# def setup_gripper(bot, robot_name, current_limit):
#     print(f"\n=== Setting up {robot_name} ===")
#     try:
#         bot.dxl.robot_torque_enable("single", "gripper", True)
#         rospy.sleep(0.2)
#         set_register(robot_name, "gripper", "Current_Limit", int(current_limit))
#         rospy.sleep(0.2)
#         bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
#         rospy.sleep(0.2)
#         bot.dxl.robot_torque_enable("single", "gripper", True)
#         rospy.sleep(0.5)
#         print_gripper_registers(robot_name)
#     except Exception as e:
#         print(f"[{robot_name}] setup warning: {e}")


def read_feedback(bot, robot_name):
    try:
        js = bot.dxl.joint_states
        pos = js.position[6]
        effort = js.effort[6] if len(js.effort) > 6 else float("nan")
        print(f"[{robot_name}] feedback pos={pos:.4f} effort={effort:.4f}")
    except Exception as e:
        print(f"[{robot_name}] feedback read failed: {e}")


def validate_value(value, min_cmd, max_cmd):
    if not math.isfinite(value):
        return False, "value is not finite"
    if value < min_cmd or value > max_cmd:
        return False, f"value {value} outside allowed range [{min_cmd}, {max_cmd}]"
    return True, None


def command_gripper(bot, robot_name, value, wait_time):
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = float(value)
    print(f"[{robot_name}] sending cmd={cmd.cmd}")
    bot.gripper.core.pub_single.publish(cmd)
    rospy.sleep(wait_time)
    read_feedback(bot, robot_name)


def print_help(min_cmd, max_cmd):
    print("\nCommands:")
    print("  <number>           send that gripper command to all selected robots")
    print("  <robot> <number>   send to one robot, e.g. 'puppet_right 70'")
    print("  open               send min range value")
    print("  close              send max range value")
    print("  status             print gripper feedback and registers")
    print("  help               show this help")
    print("  q / quit / exit    exit")
    print(f"\nAllowed command range: [{min_cmd}, {max_cmd}]")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for testing Interbotix gripper commands.")
    parser.add_argument("--robots", type=str, default="puppet_right", help="Comma-separated robot names")
    parser.add_argument("--min-cmd", type=float, default=DEFAULT_MIN_CMD, help="Minimum allowed command")
    parser.add_argument("--max-cmd", type=float, default=DEFAULT_MAX_CMD, help="Maximum allowed command")
    parser.add_argument("--wait-time", type=float, default=DEFAULT_WAIT_TIME, help="Seconds to wait after each command")
    parser.add_argument("--current-limit", type=int, default=DEFAULT_CURRENT_LIMIT, help="Current limit register value")
    args = parser.parse_args()

    robot_names = [r.strip() for r in args.robots.split(",") if r.strip()]
    if not robot_names:
        raise ValueError("No robot names provided")
    if args.min_cmd > args.max_cmd:
        raise ValueError("min-cmd must be <= max-cmd")

    # rospy.init_node("gripper_interactive_cli", anonymous=True)

    bots = {}
    for i, robot_name in enumerate(robot_names):
        bots[robot_name] = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name=robot_name,
            init_node=(i == 0),
        )

    for robot_name, bot in bots.items():
        setup_gripper(bot, robot_name, args.current_limit)

    print_help(args.min_cmd, args.max_cmd)

    while not rospy.is_shutdown():
        try:
            raw = input("\ngripper> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        if lowered == "help":
            print_help(args.min_cmd, args.max_cmd)
            continue
        if lowered == "status":
            for robot_name, bot in bots.items():
                read_feedback(bot, robot_name)
                try:
                    print_gripper_registers(robot_name)
                except Exception as e:
                    print(f"[{robot_name}] register read failed: {e}")
            continue
        if lowered == "open":
            value = args.min_cmd
            for robot_name, bot in bots.items():
                command_gripper(bot, robot_name, value, args.wait_time)
            continue
        if lowered == "close":
            value = args.max_cmd
            for robot_name, bot in bots.items():
                command_gripper(bot, robot_name, value, args.wait_time)
            continue

        parts = raw.split()

        if len(parts) == 1:
            try:
                value = float(parts[0])
            except ValueError:
                print("Invalid input. Enter a number, 'status', 'open', 'close', or 'help'.")
                continue

            ok, msg = validate_value(value, args.min_cmd, args.max_cmd)
            if not ok:
                print(f"Rejected: {msg}")
                continue

            for robot_name, bot in bots.items():
                command_gripper(bot, robot_name, value, args.wait_time)
            continue

        if len(parts) == 2:
            robot_name, value_str = parts
            if robot_name not in bots:
                print(f"Unknown robot '{robot_name}'. Known robots: {', '.join(bots.keys())}")
                continue
            try:
                value = float(value_str)
            except ValueError:
                print(f"Invalid numeric value '{value_str}'")
                continue

            ok, msg = validate_value(value, args.min_cmd, args.max_cmd)
            if not ok:
                print(f"Rejected: {msg}")
                continue

            command_gripper(bots[robot_name], robot_name, value, args.wait_time)
            continue

        print("Invalid input format. Use '<number>' or '<robot> <number>'.")


if __name__ == "__main__":
    main()