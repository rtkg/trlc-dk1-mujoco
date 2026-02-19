"""Simple MuJoCo simulation of the TRLC DK-1 Follower URDF."""

import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

URDF_DIR = Path("/home/rtkg/Coding/trlc-dk1-follower-urdf")
URDF_PATH = URDF_DIR / "TRLC-DK1-Follower.urdf"

# Read the URDF and inject a <mujoco> extension block.
# MuJoCo's URDF parser strips directory prefixes from mesh filenames and uses
# meshdir to locate them, so we point meshdir at the collision mesh directory.
urdf_text = URDF_PATH.read_text()
extension = f"""
  <mujoco>
    <compiler meshdir="{URDF_DIR}/meshes/collision" balanceinertia="true"/>
    <option gravity="0 0 -9.81"/>
    <worldbody>
      <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="2 2 0.1" pos="0 0 -0.01" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
    </worldbody>
  </mujoco>
"""
urdf_text = urdf_text.replace("</robot>", extension + "</robot>")


def pd_controller(q, dq, q_des, kp, kd):
    """Critically-damped PD controller.

    Args:
        q: Current joint positions.
        dq: Current joint velocities.
        q_des: Desired joint positions.
        kp: Proportional gains (array, one per DOF).
        kd: Derivative gains (array, one per DOF).

    Returns:
        Control torques.
    """
    return kp * (q_des - q) - kd * dq


def compute_gains(model, data, wn=10.0):
    """Compute per-joint PD gains scaled by the mass matrix diagonal.

    Uses kp = M_ii * wn^2 and kd = 2 * M_ii * wn (critical damping).
    """
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, M, data.qM)
    m_diag = np.diag(M)
    kp = m_diag * wn**2
    kd = 2.0 * m_diag * wn
    return kp, kd


def _handle_sigint(sig, frame):
    print("\nInterrupted, shutting down...")
    os._exit(0)


def main():
    signal.signal(signal.SIGINT, _handle_sigint)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    model = mujoco.MjModel.from_xml_string(urdf_text)
    # Disable all collisions (URDF extension doesn't support <contact> excludes)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)

    kp, kd = compute_gains(model, data, wn=10.0)
    q_des = np.array(
        [0.0, 1.57, 2.85, 0.29, 0.0, 0.0, 0.0, 0.0]
    )  # pointing straight up
    q_des = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # default 'folded' config

    viewer = mujoco.viewer.launch_passive(model, data)
    for step in range(steps):
        if not viewer.is_running():
            break
        step_start = time.monotonic()

        # Gravity + Coriolis compensation + PD
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = (
            pd_controller(data.qpos, data.qvel, q_des, kp, kd) + data.qfrc_bias
        )
        mujoco.mj_step(model, data)

        # Print bias forces every ~1s
        if step % 500 == 0:
            print(
                f"t={data.time:5.1f}s  "
                f"qfrc_bias: {np.array2string(data.qfrc_bias, precision=4, suppress_small=True)}"
            )

        viewer.sync()
        elapsed = time.monotonic() - step_start
        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)

    os._exit(0)


if __name__ == "__main__":
    main()
