"""Export the competition robot's native geometry for CAD reference and rendering."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arena_mujoco.competition_robot import add_competition_robot


def main():
    out = ROOT/'output/competition/robot'
    out.mkdir(parents=True,exist_ok=True)
    root = ET.Element('mujoco')
    ET.SubElement(root,'compiler',angle='radian')
    metadata = add_competition_robot(root,{})
    xml = ET.tostring(root,encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    address = model.joint(metadata['freejoint']).qposadr[0]
    data.qpos[address:address+3] = [0,0,.020]
    data.qpos[model.joint(metadata['lift_joint']).qposadr[0]] = .030
    for joint in metadata['jaw_joints']:
        data.qpos[model.joint(joint).qposadr[0]] = .019
    mujoco.mj_forward(model,data)
    names = {int(mujoco.mjtGeom.mjGEOM_BOX):'box',int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere',
             int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder'}
    geoms = []
    for gid in range(model.ngeom):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat,data.geom_xmat[gid])
        geoms.append({'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid),
                      'body':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid])),
                      'type':names[int(model.geom_type[gid])], 'size_m':model.geom_size[gid].tolist(),
                      'position_m':data.geom_xpos[gid].tolist(),'quaternion_wxyz':quat.tolist(),
                      'rgba':model.geom_rgba[gid].tolist()})
    payload = {'units':'meters','pose':{'lift_m':.030,'sample_aperture_m':.056},
               'robot':metadata,'geoms':geoms}
    (out/'geometry.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'robot.xml').write_text(xml+'\n')
    # The public spec describes the home pose, independent of the display pose.
    data.qpos[model.joint(metadata['lift_joint']).qposadr[0]] = 0
    mujoco.mj_forward(model,data)
    metadata['center_of_mass_body_m']=(data.subtree_com[model.body('competition_robot').id]
                                      -data.xpos[model.body('competition_robot').id]).tolist()
    metadata['sources']=['https://www.pololu.com/product/5188/specs',
                         'https://www.pololu.com/product/1452',
                         'https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/']
    (ROOT/'robot_design.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out/'geometry.json')


if __name__=='__main__':
    main()
