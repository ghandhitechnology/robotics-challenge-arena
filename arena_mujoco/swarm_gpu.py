"""GPU motor loops for the same free-body modules used in native validation."""
import warp as wp


@wp.kernel
def motor_control(
    qvel: wp.array2d(dtype=float), commands: wp.array3d(dtype=float),
    wheel_dofs: wp.array2d(dtype=int), actuator_ids: wp.array2d(dtype=int),
    lift_target: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float), dt: float,
):
    e, r = wp.tid()
    for side in range(2):
        velocity = qvel[e, wheel_dofs[r, side]]
        desired = -20.0 * wp.clamp(commands[e, r, side], -1.0, 1.0)
        request = 0.00015 * (desired - velocity)
        limit = float(0.002)
        if request * velocity > 0.0:
            limit = limit * wp.max(0.0, 1.0 - wp.abs(velocity) / 26.1799388)
        ctrl[e, actuator_ids[r, side]] = wp.clamp(request, -limit, limit)
    wanted = 0.004 * (wp.clamp(commands[e, r, 2], -1.0, 1.0) + 1.0)
    previous = lift_target[e, r]
    current = previous + wp.clamp(wanted - previous, -0.01 * dt, 0.01 * dt)
    lift_target[e, r] = current
    ctrl[e, actuator_ids[r, 2]] = current
