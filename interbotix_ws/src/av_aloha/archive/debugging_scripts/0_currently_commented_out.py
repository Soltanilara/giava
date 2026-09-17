"""             
# --------------------------------------------
# LOW PASS FILTER
# --------------------------------------------

filtered_q = (
    ALPHA * left_arm_q
    + (1 - ALPHA) * filtered_q
)

# --------------------------------------------
# COMMAND DEADBAND
# --------------------------------------------

send_command = False

if prev_cmd_q is None:
    send_command = True

else:

    diff = np.linalg.norm(
        filtered_q - prev_cmd_q
    )

    if diff > COMMAND_DEADBAND:
        send_command = True """

""" 
# ------------------------------------------------
# WORKSPACE CLAMP
# ------------------------------------------------

# forward/back

T_robot_target[0,3] = np.clip(
    T_robot_target[0,3],
    0.15,
    0.40,
)

# left/right

T_robot_target[1,3] = np.clip(
    T_robot_target[1,3],
    -0.25,
    0.25,
)

# up/down

T_robot_target[2,3] = np.clip(
    T_robot_target[2,3],
    0.15,
    0.40,
) """