"""
Minimal feasibility test of the "sketch -> simple spec -> MJCF" idea.

A character is described as a list of capsule segments in the side-view (x, z) plane.
Each non-root segment attaches to its parent with a hinge joint at its `start` point.
The generator emits a Walker2d-style MJCF (rootx / rootz / rooty + hinge motors), so the
result can be loaded directly with gym.make("Walker2d-v5", xml_file=...).
"""
import json
import math
import xml.etree.ElementTree as ET

import mujoco

WALKER_MASS_REF = 20.0  # rough total mass of the stock walker; used to scale motor gear


def _fmt(*vals):
    return " ".join(f"{v:.4f}" for v in vals)


def spec_to_mjcf(spec: dict) -> tuple[str, dict]:
    segs = {s["name"]: s for s in spec["segments"]}
    root = next(s for s in spec["segments"] if s.get("parent") is None)

    # Put the whole character on the floor: shift z so the lowest capsule surface touches z = 0.
    lowest = min(min(s["start"][1], s["end"][1]) - s["radius"] for s in spec["segments"])
    dz = -lowest + 0.005
    for s in spec["segments"]:
        s["start"] = [s["start"][0], s["start"][1] + dz]
        s["end"] = [s["end"][0], s["end"][1] + dz]

    root_x = 0.5 * (root["start"][0] + root["end"][0])
    root_z = 0.5 * (root["start"][1] + root["end"][1])

    mj = ET.Element("mujoco", model=spec.get("name", "custom2d"))
    ET.SubElement(mj, "compiler", angle="degree", inertiafromgeom="true")
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", armature="0.01", damping="0.1", limited="true")
    ET.SubElement(default, "geom", conaffinity="0", condim="3", contype="1", density="1000",
                  friction="0.9 0.1 0.1", rgba="0.8 0.6 0.4 1")
    ET.SubElement(mj, "option", integrator="RK4", timestep="0.002")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=".4 .5 .6", rgb2="0 0 0",
                  width="100", height="100")
    ET.SubElement(asset, "texture", builtin="checker", height="100", name="texplane",
                  rgb1="0 0 0", rgb2="0.8 0.8 0.8", type="2d", width="100")
    ET.SubElement(asset, "material", name="MatPlane", reflectance="0.5", shininess="1",
                  specular="1", texrepeat="60 60", texture="texplane")

    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", cutoff="100", diffuse="1 1 1", dir="0 0 -1.3", directional="true",
                  exponent="1", pos="0 0 1.3", specular=".1 .1 .1")
    ET.SubElement(wb, "geom", conaffinity="1", condim="3", name="floor", pos="0 0 0",
                  rgba="0.8 0.9 0.8 1", size="40 40 40", type="plane", material="MatPlane")

    # world-frame origin of each body = its joint point (root: capsule centre)
    origins = {root["name"]: (root_x, root_z)}
    bodies = {}

    torso = ET.SubElement(wb, "body", name=root["name"], pos=_fmt(root_x, 0, root_z))
    ET.SubElement(torso, "camera", name="track", mode="trackcom", pos="0 -3 -0.25", xyaxes="1 0 0 0 0 1")
    ET.SubElement(torso, "joint", armature="0", axis="1 0 0", damping="0", limited="false",
                  name="rootx", pos="0 0 0", stiffness="0", type="slide")
    ET.SubElement(torso, "joint", armature="0", axis="0 0 1", damping="0", limited="false",
                  name="rootz", pos="0 0 0", ref=f"{root_z:.4f}", stiffness="0", type="slide")
    ET.SubElement(torso, "joint", armature="0", axis="0 1 0", damping="0", limited="false",
                  name="rooty", pos="0 0 0", stiffness="0", type="hinge")
    ET.SubElement(torso, "geom", name=f"{root['name']}_geom", type="capsule",
                  size=f"{root['radius']:.4f}",
                  fromto=_fmt(root["start"][0] - root_x, 0, root["start"][1] - root_z,
                              root["end"][0] - root_x, 0, root["end"][1] - root_z))
    bodies[root["name"]] = torso

    motor_joints = []

    def add_children(parent_name):
        for s in spec["segments"]:
            if s.get("parent") != parent_name:
                continue
            px, pz = origins[parent_name]
            ox, oz = s["start"]
            body = ET.SubElement(bodies[parent_name], "body", name=s["name"], pos=_fmt(ox - px, 0, oz - pz))
            origins[s["name"]] = (ox, oz)
            lo, hi = s.get("joint_range", [-90, 90])
            ET.SubElement(body, "joint", axis="0 -1 0", name=f"{s['name']}_joint", pos="0 0 0",
                          range=f"{lo} {hi}", type="hinge")
            ET.SubElement(body, "geom", name=f"{s['name']}_geom", type="capsule",
                          size=f"{s['radius']:.4f}",
                          friction=str(s.get("friction", 0.9)),
                          rgba=s.get("rgba", "0.8 0.6 0.4 1"),
                          fromto=_fmt(0, 0, 0, s["end"][0] - ox, 0, s["end"][1] - oz))
            bodies[s["name"]] = body
            if s.get("motor", True):
                motor_joints.append(f"{s['name']}_joint")
            add_children(s["name"])

    add_children(root["name"])

    # scale motor strength with total mass so big/small creatures are equally "strong"
    tmp_xml = ET.tostring(mj, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(tmp_xml)
    total_mass = float(m.body_mass.sum())
    gear = spec.get("gear", 100.0 * total_mass / WALKER_MASS_REF)

    act = ET.SubElement(mj, "actuator")
    for j in motor_joints:
        ET.SubElement(act, "motor", ctrllimited="true", ctrlrange="-1.0 1.0", gear=f"{gear:.1f}", joint=j)

    ET.indent(mj)
    xml = ET.tostring(mj, encoding="unicode")
    info = {
        "torso_z0": root_z,
        "total_mass": total_mass,
        "gear": gear,
        "n_motors": len(motor_joints),
        "healthy_z_range": (0.6 * root_z, 1.6 * root_z),
    }
    return xml, info


if __name__ == "__main__":
    import sys
    spec = json.load(open(sys.argv[1]))
    xml, info = spec_to_mjcf(spec)
    open(sys.argv[2], "w").write(xml)
    print(json.dumps(info, indent=2))
