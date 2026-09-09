"""Blender: render and export the exact native robot collision assembly reference."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/competition/robot'
payload = json.loads((OUT/'geometry.json').read_text())
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system='METRIC'
scene.unit_settings.length_unit='MILLIMETERS'
scene.render.engine='CYCLES'
scene.cycles.samples=32
scene.render.threads_mode='FIXED'
scene.render.threads=4
scene.render.resolution_x=1280
scene.render.resolution_y=960
scene.render.resolution_percentage=100
scene.world.color=(.25,.25,.25)
scene.view_settings.view_transform='AgX'
robot_objects=[]
materials={}
for record in payload['geoms']:
    size=record['size_m']
    if record['type']=='box':
        bpy.ops.mesh.primitive_cube_add(size=2)
        obj=bpy.context.object
        obj.scale=size
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    elif record['type']=='cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=size[0],depth=2*size[1])
        obj=bpy.context.object
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=size[0])
        obj=bpy.context.object
    obj.name=record['name']
    obj.location=record['position_m']
    obj.rotation_mode='QUATERNION'
    obj.rotation_quaternion=record['quaternion_wxyz']
    obj['native_body']=record['body']
    obj['role']='Native collision geometry reference'
    key=tuple(round(v,3) for v in record['rgba'])
    if key not in materials:
        mat=bpy.data.materials.new(f'material_{len(materials)}')
        mat.diffuse_color=key
        mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value=key
        bsdf.inputs['Roughness'].default_value=.55
        materials[key]=mat
    obj.data.materials.append(materials[key])
    robot_objects.append(obj)
# Export only the robot. STL coordinates are millimeters; the blend uses meters.
bpy.ops.object.select_all(action='DESELECT')
for obj in robot_objects:obj.select_set(True)
bpy.ops.wm.stl_export(filepath=str(OUT/'assembly_reference_mm.stl'),export_selected_objects=True,global_scale=1000)
bpy.ops.object.select_all(action='DESELECT')
# The orange disk is a display reference for the lower contact band.
bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=.028,depth=.005,location=(0,.105,.0325))
disk=bpy.context.object;disk.name='Sample_display_reference'
mat=bpy.data.materials.new('Sample amber');mat.diffuse_color=(.95,.57,.07,1);mat.use_nodes=True
mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.95,.57,.07,1)
disk.data.materials.append(mat)
bpy.ops.mesh.primitive_plane_add(size=1,location=(0,0,-.0001))
ground=bpy.context.object;ground.name='Display_ground'
mat=bpy.data.materials.new('Ground');mat.diffuse_color=(.82,.84,.86,1);ground.data.materials.append(mat)
for name,location,power,size in [('Key',(.25,.2,.4),5,.30),('Fill',(-.2,.05,.25),2,.25)]:
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=location
    obj.rotation_euler=(Vector((0,.03,.05))-obj.location).to_track_quat('-Z','Y').to_euler()
data=bpy.data.cameras.new('Robot_overview');camera=bpy.data.objects.new('Robot_overview',data);scene.collection.objects.link(camera)
camera.location=(.26,.31,.23);camera.rotation_euler=(Vector((0,.035,.06))-camera.location).to_track_quat('-Z','Y').to_euler()
data.type='ORTHO';data.ortho_scale=.32;scene.camera=camera
scene.render.filepath=str(OUT/'robot_overview.png')
bpy.ops.render.render(write_still=True)
# A second saved camera lets a reviewer inspect the thin lower sample contacts.
data=bpy.data.cameras.new('Jaw_detail');detail=bpy.data.objects.new('Jaw_detail',data);scene.collection.objects.link(detail)
detail.location=(.12,.23,.10);detail.rotation_euler=(Vector((0,.102,.044))-detail.location).to_track_quat('-Z','Y').to_euler()
data.type='ORTHO';data.ortho_scale=.15;scene.camera=detail
scene.render.filepath=str(OUT/'jaw_detail.png')
bpy.ops.render.render(write_still=True)
scene.camera=camera
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'competition_robot.blend'))
print('Saved robot reference assembly and renders to',OUT)
