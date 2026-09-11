"""Build the five-robot concept in Blender, preserving unrelated scenes.

Run through Blender MCP with exec(compile(...)) or:
blender --background --python scripts/build_best_design_blender.py -- --render
All design inputs use SI in best_design.json. Drawing helpers use mm.
"""
from pathlib import Path
import json, math, struct, sys, hashlib
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'best_design'
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / 'best_design.json').read_text())
A = json.loads((ROOT / 'arena_spec.json').read_text())
PREFIX = 'BestDesign_'
M = {}


def material(name, rgb, metal=0, rough=.45):
    m = bpy.data.materials.get(PREFIX+name) or bpy.data.materials.new(PREFIX+name)
    m.diffuse_color=(*rgb[:3],1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*rgb[:3],1)
    p.inputs['Metallic'].default_value=metal; p.inputs['Roughness'].default_value=rough
    M[name]=m; return m


for args in [('deck',(.055,.083,.105),.25,.35),('rubber',(.018,.026,.032),0,.8),
             ('silver',(.55,.65,.72),.8,.25),('brass',(.64,.43,.16),.65,.32),
             ('pcb',(.045,.27,.18),.15,.4),('black',(.014,.022,.035),.1,.32),
             ('white',(.81,.87,.90),.05,.6),('muted',(.30,.40,.49),0,.6),
             ('floor',(.028,.045,.063),.1,.6),('paper',(.79,.79,.71),0,.75),
             ('kit',(.94,.90,.73),0,.65),('red',(.84,.16,.20),0,.4),
             ('green',(.13,.67,.42),0,.4),('yellow',(.96,.66,.10),0,.4),
             ('lab',(.18,.50,.95),.25,.35),('kit_role',(.73,.83,.88),.2,.4),
             ('line',(.22,.51,.65),.1,.5)]:
    material(*args)


def own(o,name,mat):
    o.name=PREFIX+name
    if mat: o.data.materials.append(M[mat])
    return o


def box(name,loc,size,mat='deck',bevel=.6):
    bpy.ops.mesh.primitive_cube_add(size=1,location=Vector(loc)*.001)
    o=own(bpy.context.object,name,mat); o.dimensions=Vector(size)*.001
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        mod=o.modifiers.new('Machined edges','BEVEL'); mod.width=bevel*.001; mod.segments=3
        mod=o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    return o


def cyl(name,loc,radius,depth,mat='silver',axis='Z',verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts,radius=radius*.001,depth=depth*.001,location=Vector(loc)*.001)
    o=own(bpy.context.object,name,mat)
    if axis=='X':o.rotation_euler[1]=math.pi/2
    if axis=='Y':o.rotation_euler[0]=math.pi/2
    bevel=o.modifiers.new('Edge radius','BEVEL');bevel.width=min(.3,depth/6)*.001;bevel.segments=2
    o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    return o


def line(name,points,mat='line',r=.5):
    cv=bpy.data.curves.new(PREFIX+name,'CURVE');cv.dimensions='3D';cv.bevel_depth=r*.001;cv.bevel_resolution=2
    sp=cv.splines.new('POLY');sp.points.add(len(points)-1)
    for p,co in zip(sp.points,points):p.co=(*[v*.001 for v in co],1)
    o=bpy.data.objects.new(PREFIX+name,cv);bpy.context.scene.collection.objects.link(o);cv.materials.append(M[mat]);return o


def label(body,loc,size=10,mat='white',align='LEFT',angle=0):
    cv=bpy.data.curves.new(PREFIX+'Label','FONT');cv.body=body;cv.size=size*.001;cv.align_x=align
    cv.space_character=1.12
    o=bpy.data.objects.new(PREFIX+'Label_'+body[:28],cv);bpy.context.scene.collection.objects.link(o)
    o.location=Vector(loc)*.001;o.rotation_euler[2]=angle;cv.materials.append(M[mat]);return o


def dim(a,b,body,offset=(0,0),size=9):
    p=[a[0]+offset[0],a[1]+offset[1],a[2]];q=[b[0]+offset[0],b[1]+offset[1],b[2]]
    line('dimension',[p,q],r=.35)
    for orig,co in [(a,p),(b,q)]:
        line('extension',[orig,co],r=.25)
        line('tick',[(co[0]-2,co[1]-2,co[2]),(co[0]+2,co[1]+2,co[2])],r=.35)
    label(body,[(p[0]+q[0])/2+3,(p[1]+q[1])/2+4,p[2]+.2],size,'white','CENTER')


def newscene(name):
    full=PREFIX+name
    old=bpy.data.scenes.get(full)
    if old:
        for o in list(old.objects):bpy.data.objects.remove(o,do_unlink=True)
        bpy.data.scenes.remove(old)
    s=bpy.data.scenes.new(full);bpy.context.window.scene=s
    s.unit_settings.system='METRIC';s.unit_settings.length_unit='MILLIMETERS'
    s.render.engine='CYCLES';s.cycles.samples=24
    s.cycles.use_denoising=True;s.render.resolution_x=2000;s.render.resolution_y=1500;s.render.resolution_percentage=100
    s.world=bpy.data.worlds.new(PREFIX+name+'_world');s.world.use_nodes=True
    s.world.node_tree.nodes['Background'].inputs[0].default_value=(.13,.17,.23,1)
    s.world.node_tree.nodes['Background'].inputs[1].default_value=.35
    s.view_settings.view_transform='AgX'
    s.render.image_settings.file_format='PNG'
    return s


def lighting(scale=1,center=(0,0,0)):
    for name,delta,power,size in [('Key',(400,-200,650),80,600),('Fill',(-350,200,400),45,450),('Rim',(80,400,550),65,400)]:
        loc=Vector(center)+Vector(delta)*scale
        bpy.ops.object.light_add(type='AREA',location=loc*.001)
        o=bpy.context.object;o.name=PREFIX+name;o.data.energy=power*scale*scale*.05;o.data.shape='DISK';o.data.size=size*scale*.001
        o.rotation_euler=(Vector(center)*.001-o.location).to_track_quat('-Z','Y').to_euler()


def camera(loc,target,scale):
    bpy.ops.object.camera_add(location=Vector(loc)*.001);o=bpy.context.object;o.name=PREFIX+'Camera'
    o.rotation_euler=(Vector(target)*.001-o.location).to_track_quat('-Z','Y').to_euler()
    o.data.type='ORTHO';o.data.ortho_scale=scale*.001;o.data.clip_start=.001;o.data.clip_end=100
    if loc[1]>target[1]+.01:o.rotation_euler.rotate_axis('Z',math.pi)
    bpy.context.scene.camera=o


def drill(deck,loc,r=1.6):
    cutter=cyl('hole_tool',loc,r,10,'silver')
    mod=deck.modifiers.new('M3 through hole','BOOLEAN');mod.operation='DIFFERENCE';mod.object=cutter
    bpy.context.view_layer.objects.active=deck;bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter,do_unlink=True)


def robot(role,origin=(0,0,0),aperture=26,lift=0,extension=0):
    """CAD packages actual component envelopes; parts are separate named meshes."""
    start=set(bpy.context.scene.objects);origin=Vector(origin)
    col='kit_role' if role=='kit' else role
    def B(n,p,s,m='deck',bevel=.6):return box(role+'_'+n,p,s,m,bevel)
    def C(n,p,r,d,m='silver',axis='Z'):return cyl(role+'_'+n,p,r,d,m,axis)
    deck=B('chassis_plate',(0,-20,27),(105,90,4),'deck',2)
    for x in (-29,29):
        cut=B('motor_clearance_tool',(x,0,27),(40,18,12),'black',0)
        mod=deck.modifiers.new('Motor clearance pocket','BOOLEAN');mod.object=cut;mod.operation='DIFFERENCE'
        bpy.context.view_layer.objects.active=deck;bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cut,do_unlink=True)
    for x in (-43,43):
        for y in (-55,15):
            drill(deck,(x,y,27));C('M3_bolt',(x,y,30),2.6,2,'silver')
    B('rear_bumper',(0,-63,32),(100,4,12),col,1)
    for sign in (-1,1):
        C('wheel',(sign*55,0,25),25,8,'rubber','X')
        C('wheel_hub',(sign*59.4,0,25),11,1,'silver','X')
        C('axle',(sign*47,0,25),1.5,15,'silver','X')
        for a in range(0,360,30):
            rad=math.radians(a)
            C('tire_tread',(sign*55,24.6*math.sin(rad),25+24.6*math.cos(rad)),0.4,8.0,'rubber','X')
        B('gear_motor',(sign*29,0,25),(32,12,10),'silver',.6)
        B('gearbox',(sign*43,0,25),(8,12,10),'brass',.4)
        B('encoder',(sign*11,0,25),(3,15,14),'pcb',.1)
        B('motor_bracket',(sign*40,0,32),(4,18,9),'deck',.5)
        B('wheel_guard',(sign*60,-3,53),(5,55,3),col,.8)
    C('rear_caster',(0,-53,8),8,8,'rubber','X')
    B('caster_mount',(0,-53,21),(18,17,12),'deck')
    B('2S_battery',(0,-40,37),(54,32,15),'black',1.7)
    B('battery_strap',(0,-40,45),(10,35,1.4),col,.2)
    for x in (-28,28):
        for y in (-52,-26):C('board_standoff',(x,y,49),1.6,7,'brass')
    B('Pi_Zero2_envelope',(0,-39,54),(65,30,1.6),'pcb',.3)
    B('processor',(0,-40,57),(12,12,4),'black',.3)
    for x in (-19,-5,10):B('I_O_connector',(x,-53,57),(10,6,6),'silver',.3)
    B('controller_and_IMU',(-31,-9,52),(30,25,2),'pcb',.3)
    B('MCU',(-31,-9,55),(8,8,3),'black',.3)
    B('regulator',(31,-13,47),(25,20,10),'pcb',.5)
    # 20 x 34 x 26 mm actuator envelopes from ROBOTIS drawing.
    B('lift_XL330',(31,17,62),(20,26,34),'black',1)
    C('lift_drum',(19,17,67),6,4,'brass','X')
    B('mast_foot',(0,20,34),(40,18,8),col,1)
    for x in (-14,14):
        C('lift_guide',(x,20,63),2.5,70,'silver')
        C('guide_bearing',(x,20,38+lift),5,16,'black')
    B('mast_header',(0,20,98),(40,16,6),col,1)
    C('lift_idler',(0,20,95),4,8,'silver','X')
    line(role+'_lift_cable',[(18,17,67),(0,20,95),(0,20,38+lift)],'black',.5)
    B('carriage',(0,20,38+lift),(38,14,14),'silver',1)
    B('fork_bridge',(0,39,37+lift),(65,32,4),col,.8)
    B('grip_XL330',(0,43,56+lift),(20,26,34),'black',1)
    C('grip_pinion',(0,43,36+lift),6,3,'brass')
    for sign in (-1,1):
        x=sign*(aperture/2+2.7)
        B('rack_slide',(sign*15,43+sign*4,34+lift),(40,3,3),'silver',.2)
        B('finger_spine',(x,61,22+lift),(4,39,22),'silver',.4)
        B('upper_pad',(sign*(aperture/2+.7),65,11+lift),(1.4,16,10),'rubber',.2)
        B('sample_support',(sign*(aperture/2+2.7),65,4.1+lift),(4,14,1.4),'silver',.15)
        B('sample_pad',(sign*(aperture/2+.6),65,4.1+lift),(1.2,14,1.4),'rubber',.1)
        B('pad_vertical_link',(x,58,9+lift),(4,5,9.8),'silver',.2)
    if role=='lab':
        moving_words=('fork_bridge','grip_XL330','grip_pinion','rack_slide','finger_spine','upper_pad','sample_support','sample_pad','pad_vertical_link')
        for o in set(bpy.context.scene.objects)-start:
            if any(word in o.name for word in moving_words):o.location.y+=extension*.001
        for x in (-20,20):
            B('extension_rail',(x,48,30+lift),(5,66,5),'silver',.4)
            B('extension_bearing',(x,35+extension,34+lift),(9,12,5),'black',.4)
        C('extension_leadscrew',(-28,47,33+lift),1.5,64,'brass','Y')
        B('extension_encoder_motor',(-28,0,33+lift),(12,32,10),'silver',.5)
        B('extension_nut',(-28,35+extension,33+lift),(9,7,7),'brass',.4)
    # Camera at the mast top. Its optics look toward the jaws and floor.
    B('camera_bracket',(0,15,100),(28,23,3),col,.6)
    B('camera_PCB',(0,17,104),(25,24,1.6),'pcb',.2)
    lens=C('camera_lens',(0,25,101),4.5,8,'black','Y');lens.rotation_euler[0]=math.pi/4
    glass=C('camera_glass',(0,28,98),3.5,.5,'lab','Y');glass.rotation_euler[0]=math.pi/4
    B('power_switch',(-40,-46,60),(8,10,7),'black',.5)
    C('status_LED',(39,-47,61),1.6,2,col)
    line(role+'_power_wire',[(-21,-38,44),(-25,-30,50),(-31,-9,54)],'red',.6)
    line(role+'_camera_flex',[(0,-39,57),(-8,-13,67),(-8,17,102)],'brass',1.3)
    label(role.upper(),(0,-31,61),8,'white','CENTER')
    created=set(bpy.context.scene.objects)-start
    for o in created:
        o.location+=origin*.001;o['robot_id']=role;o['design_concept']=True
    return deck,created


def kit(x,y,z=10):
    box('medical_kit',(x,y,z),(25,25,20),'kit',.7)
    box('cross_v',(x,y,z+10.15),(5,20,.3),'red',0)
    box('cross_h',(x,y,z+10.31),(20,5,.3),'red',0)


def labplate(center=(1068,490.5),size=(150,345)):
    o=box('laboratory',(center[0],center[1],1.5),(*size,3),'muted',.2)
    for y in (-100,0,100):
        hole=cyl('slot_cutter',(center[0],center[1]+y,1.5),30,10,None if False else 'silver')
        mod=o.modifiers.new('60mm slot','BOOLEAN');mod.object=hole;mod.operation='DIFFERENCE'
        bpy.context.view_layer.objects.active=o;bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(hole,do_unlink=True)
    return o


def arena():
    box('arena_support',(571.5,590.5,-5),(1143,1181,10),'paper',1)
    for x0,y0,x1,y1,col in [(0,358,180,861,'red'),(0,0,180,338,'yellow'),(0,881,180,1181,'yellow'),(863,701,1143,1181,'green'),(863,0,1143,280,'lab')]:
        box('zone_tint',((x0+x1)/2,(y0+y1)/2,.025),(x1-x0,y1-y0,.05),col,0)
    for x,y,w,h in [(190,590.5,20,1181),(90,348,180,20),(90,871,180,20),(853,941,20,480),(1003,691,280,20),(853,140,20,280),(1003,290,280,20),(340,609.5,20,500),(760,609.5,20,500),(550,609.5,420,20)]:
        box('20mm_tape',(x,y,.075),(w,h,.15),'black',0)
    for row in A['placements']['cylinder_rows']:
        for x in A['placements']['cylinder_x']:cyl('patient_'+row['color'],(x,row['y'],10),10,20,row['color'])
    for y in [1053,1098]:
        for x in A['placements']['kit_x']:kit(x,y)
    for x,y in A['placements']['samples']:cyl('sample',(x,y,2.5),28,5,'black')
    labplate()
    for txt,p in [('HOSPITAL',(90,594,1)),('PCC',(90,155,1)),('PCC',(90,1050,1)),('QUARANTINE',(1000,245,1))]:label(txt,p,17,'white','CENTER')


def export_stl(o,path):
    dg=bpy.context.evaluated_depsgraph_get();ev=o.evaluated_get(dg);mesh=ev.to_mesh();mesh.calc_loop_triangles()
    triangles=[]
    for t in mesh.loop_triangles:
        vs=[ev.matrix_world@mesh.vertices[i].co for i in t.vertices]
        normal=(vs[1]-vs[0]).cross(vs[2]-vs[0]).normalized()
        triangles.append((normal,vs))
    with path.open('wb') as f:
        f.write(b'Best design chassis fit mockup. Units mm. Not fabrication released.'.ljust(80,b' '));f.write(struct.pack('<I',len(triangles)))
        for n,vs in triangles:f.write(struct.pack('<12fH',*n,*[float(v)*1000 for xyz in vs for v in xyz],0))
    ev.to_mesh_clear()


def build():
    # Save any unsaved input before adding scenes, keeping all unrelated data.
    if bpy.data.is_dirty and not bpy.data.filepath:
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'incoming_scene_backup.blend'),copy=True)
    scenes=[]
    s=newscene('01_fleet_arena');scenes.append((s,'fleet_arena.png'))
    arena()
    for r in D['robots']:robot(r['id'],(*[v*1000 for v in r['start_xy_m']],0))
    label('FIVE ROBOTS / ONE SHARED MECHANISM',(15,1280,1),38)
    label('KOSAC senior preliminary  |  16 scored objects  |  120 seconds',(15,1230,1),20,'muted')
    label('1143 x 1181 mm  /  fixed senior field',(571,-70,1),18,'white','CENTER')
    camera((1500,-1350,2400),(570,630,0),1820);lighting(3,(570,600,0))
    s=newscene('02_start_packing');scenes.append((s,'start_packing.png'))
    box('start',(140,240,-3),(280,480,6),'floor',1)
    line('start_boundary',[(0,0,0),(280,0,0),(280,480,0),(0,480,0),(0,0,0)],'line',1)
    for r in D['robots']:
        x=r['start_xy_m'][0]*1000-863;y=r['start_xy_m'][1]*1000-701
        robot(r['id'],(x,y,0))
        line('envelope',[(x-62.5,y-65,.5),(x+62.5,y-65,.5),(x+62.5,y+85,.5),(x-62.5,y+85,.5),(x-62.5,y-65,.5)],'muted',.3)
    for y in [352,397]:
        for x in [44,84]:kit(x,y)
    label('4 original kits',(71,454,1),9,'white','CENTER')
    label('STARTING FIT',(0,554,0),24)
    label('Five 125 x 150 mm robots. All dimensions in mm.',(0,527,0),10,'muted')
    dim((0,480,0),(280,480,0),'280 clear',(0,25),10)
    dim((0,0,0),(0,480,0),'480',(-29,0),10)
    dim((7.5,0,0),(272.5,0,0),'265 occupied',(0,-25),10)
    label('7.5 edge margin  /  15 column gap  /  7.5 row gap',(140,-55,0),8.5,'white','CENTER')
    label('LAB reverses south first. The upper-left cell keeps the kit layout.',(140,-75,0),7.5,'muted','CENTER')
    camera((125,230,900),(125,230,0),710);lighting(1,(140,240,0))
    s.render.resolution_x=1600;s.render.resolution_y=1900
    s=newscene('03_shared_mechanism');scenes.append((s,'shared_mechanism.png'))
    deck,parts=robot('red',aperture=26)
    box('workbench',(0,0,-4),(430,350,8),'floor',2)
    # Measured axes and a bench sample make the working scale visible.
    cyl('patient_for_scale',(0,65,10),10,20,'red')
    dim((-62.5,-65,1),(62.5,-65,1),'125 overall',(0,-26),10)
    dim((62.5,-65,1),(62.5,85,1),'150',(35,0),10)
    dim((-55,0,2),(55,0,2),'110 track',(0,-112),9)
    label('SHARED CONTACT ROBOT',(-175,217,1),17)
    label('2 wheel motors + lift + coupled jaws',(-175,195,1),9,'muted')
    label('50 mm tires',(-187,-52,1),9)
    line('wheel_leader',[(-105,-52,1),(-70,-30,1),(-55,0,30)],r=.5)
    label('65 x 30 compute',(-185,103,1),9)
    line('compute_leader',[(-92,103,1),(-66,70,32),(-27,-40,57)],r=.5)
    label('45 mm lift',(78,105,1),9)
    line('lift_leader',[(105,97,1),(55,60,80),(14,20,84)],r=.5)
    label('18-68 mm aperture',(-172,-144,1),9)
    label('0.45 kg mass budget  |  110 mm height envelope',(0,-164,1),8,'muted','CENTER')
    camera((380,-430,370),(0,0,38),600);lighting(.8,(0,10,30))
    export_stl(deck,OUT/'chassis_fit_mockup_mm.stl')
    s=newscene('04_disc_clearance');scenes.append((s,'disc_clearance.png'))
    # Engineering section in XY for a direct, undistorted top-down drawing.
    box('section_base',(0,0,-2),(145,26,4),'floor',0)
    for x in (-49,49):box('plate_section',(x,1.5,1),(38,3,2),'muted',0)
    box('disc_section',(0,2.5,1),(56,5,2),'black',0)
    for sign in (-1,1):
        box('sample_pad_section',(sign*28.6,4.1,2),(1.2,1.4,2),'lab',0)
        box('support_section',(sign*31.2,4.1,1),(4,1.4,2),'silver',0)
        box('finger_section',(sign*32.5,10,1),(4,13.2,2),'silver',0)
    dim((-30,-2,3),(30,-2,3),'60 slot',(0,-15),4)
    dim((-28,7,3),(28,7,3),'56 disc',(0,16),4)
    label('DISC RELEASE SECTION',(-72,54,1),7)
    label('5 mm disc  /  3 mm plate  /  2 mm radial clearance',(-72,44,1),3.1,'muted')
    label('Pads 3.4-4.8 above floor',(-72,33,1),3.1,'white')
    line('pad_height_leader',[(-23,32,3),(-40,18,3),(-29.5,4.1,3)],'line',.2)
    label('0.4 mm nominal rim gap',(2,-32,1),3.3,'white','CENTER')
    label('Settle on slot floor, open above rim, lift clear.',(0,-40,1),3.0,'muted','CENTER')
    label('Gauge plate thickness and jaw sag before a hardware run.',(0,-48,1),2.8,'muted','CENTER')
    camera((0,3,400),(0,3,0),177);lighting(.3,(0,0,0))
    s.render.resolution_x=2100;s.render.resolution_y=1400
    s=newscene('05_lab_extension');scenes.append((s,'lab_extension.png'))
    robot('lab',aperture=56,extension=40)
    box('bench',(0,80,-4),(410,460,8),'floor',2)
    plate=box('lab_plate_detail',(0,105,1.5),(160,150,3),'muted',.2)
    hole=cyl('slot_cut',(0,105,1.5),30,12,'silver')
    mod=plate.modifiers.new('60mm slot','BOOLEAN');mod.object=hole;mod.operation='DIFFERENCE'
    bpy.context.view_layer.objects.active=plate;bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(hole,do_unlink=True)
    cyl('seated_disc',(0,105,2.5),28,5,'black')
    label('LAB / EXTEND AFTER START',(-180,270,1),16)
    label('40 mm horizontal stage keeps the wheels below the plate',(-180,248,1),8,'muted')
    dim((87,0,1),(87,105,1),'105 tool reach',(15,0),9)
    dim((-55,25,1),(-55,30,1),'5 gap',(-37,0),7)
    label('21 motors across the fleet',(0,-120,1),10,'white','CENTER')
    label('4 per courier. LAB adds one screw-driven extension motor.',(0,-139,1),8,'muted','CENTER')
    camera((340,-400,410),(0,70,30),640);lighting(.9,(0,60,20))
    # Bounding-box audit uses start scene robot meshes and reserved envelope.
    pack=bpy.data.scenes[PREFIX+'02_start_packing'];report={'units':'mm','robots':{},'original_scene_preserved':'Scene' in bpy.data.scenes,'design_sha256':hashlib.sha256((ROOT/'best_design.json').read_bytes()).hexdigest()}
    for r in D['robots']:
        objs=[o for o in pack.objects if o.get('robot_id')==r['id'] and o.type=='MESH']
        pts=[o.matrix_world@Vector(c) for o in objs for c in o.bound_box]
        lo=[min(p[i] for p in pts)*1000 for i in range(3)];hi=[max(p[i] for p in pts)*1000 for i in range(3)]
        report['robots'][r['id']]={'bounds_min':lo,'bounds_max':hi,'size':[hi[i]-lo[i] for i in range(3)],'inside_start':lo[0]>=0 and lo[1]>=0 and hi[0]<=280 and hi[1]<=480}
    (OUT/'cad_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    bpy.context.window.scene=scenes[0][0]
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'best_design.blend'))
    (OUT/'render_jobs.json').write_text(json.dumps([{'scene':s.name,'file':f} for s,f in scenes],indent=2)+'\n')
    print(json.dumps({'blend':str(OUT/'best_design.blend'),'scenes':[s.name for s,f in scenes],'geometry_audit':report}))
    return scenes


if __name__=='__main__':
    scenes=build()
    if '--render' in sys.argv:
        for scene,filename in scenes:
            bpy.context.window.scene=scene
            scene.render.filepath=str(OUT/filename)
            bpy.ops.render.render(write_still=True)
        bpy.context.window.scene=scenes[0][0]
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'best_design.blend'))
