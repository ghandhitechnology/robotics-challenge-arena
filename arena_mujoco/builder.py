"""Build self-contained MJCF from the dimension-audited arena specification."""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .geometry import lab_prisms, rectangle_grid, tape_strips
from .materials import friction5, load_profile, material_pair

ROOT = Path(__file__).resolve().parents[1]


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def build_arena(config="senior_preliminary", *, tape_mode="flex", robot=True,
                profile=None, spec=None):
    """Return (MJCF string, JSON-compatible metadata). Native runtime owns bond history."""
    profile = load_profile(profile)
    spec = json.loads((ROOT / "arena_spec.json").read_text()) if spec is None else spec
    if config not in spec["configurations"]:
        raise ValueError(f"Unknown arena configuration: {config}")
    if tape_mode not in {"flex", "rigid", "none"}:
        raise ValueError("tape_mode must be flex, rigid or none")
    settings, dim, place = spec["configurations"][config], spec["dimensions"], spec["placements"]
    p, tape = profile, profile["tape"]
    if tape["joins"] != "butt":
        raise ValueError("The validated MuJoCo arena supports butt tape joins. Overlapping flex layers failed stability checks; see docs/mujoco_physics_sources.md.")
    width, height = dim["surface_x"]*.001, dim["surface_y"]*.001
    thickness, density = dim["tape_thickness"]*.001, p["wood_density_kg_m3"]
    root = ET.Element("mujoco", model="robotics_challenge_arena")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", balanceinertia="false")
    ET.SubElement(root, "option", timestep=str(p["timestep_s"]), gravity=f"0 0 {-p['gravity_m_s2']}",
                  integrator="implicitfast", solver="CG", iterations="80", tolerance="1e-9",
                  cone="pyramidal", jacobian="sparse")
    ET.SubElement(root, "size", memory="256M")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "map", znear="0.005", zfar="10")
    default = ET.SubElement(root, "default")
    contact = p["contact"]
    solref = numbers([contact["timeconst_s"], contact["dampratio"]])
    solimp = numbers(contact["impedance"])
    ET.SubElement(default, "geom", contype="1", conaffinity="3", condim="6", solref=solref, solimp=solimp, margin="0", gap="0")
    asset, world, contacts = [ET.SubElement(root, tag) for tag in ("asset", "worldbody", "contact")]
    emblem=json.loads((Path(__file__).parent/"assets/biohazard.json").read_text())
    ET.SubElement(asset,"mesh",name="biohazard",vertex=numbers(emblem["vertices"]),
                  face=" ".join(map(str,np.asarray(emblem["triangles"]).ravel())))
    actuator, sensor = ET.SubElement(root, "actuator"), ET.SubElement(root, "sensor")
    ET.SubElement(world, "light", pos="0.4 0.1 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="1.2 1.5 1.5", dir="-0.3 -0.3 -1", diffuse="0.4 0.4 0.4")
    ET.SubElement(world, "camera", name="overview", pos="0.5715 0.5905 1.8", quat="1 0 0 0")
    geoms, objects, pairs = [], [], []
    metadata = {"configuration": config, "tape_mode": tape_mode, "profile": p,
                "surface_size_m": [width, height], "objects": objects,
                "contact_pairs": pairs, "tape_strips": [], "tape_nodes": [],
                "flatten_qpos": {}, "geometry_source": "arena_spec.json"}

    def geom(parent, name, material, *, body="world", dynamic=False, wall=None, **attributes):
        # For flex contacts the rigid geom's higher priority supplies its vinyl friction.
        values = material_pair(p, material, "vinyl")
        element = ET.SubElement(parent, "geom", name=name,
                                friction=numbers([values[0], values[2], values[3]]),
                                priority="1", **{key: str(value) for key, value in attributes.items()})
        geoms.append({"name": name, "material": material, "body": body,
                      "dynamic": dynamic, "wall": wall})
        return element

    floor = geom(world, "floor", "paper", type="box", size=numbers([width/2, height/2, .005]),
                 pos=numbers([width/2, height/2, -.005]), rgba="0.96 0.955 0.93 1")
    floor.set("priority", "0")
    floor.set("conaffinity", "1")
    # Position discs are printed ink with no raised collision surface.
    colors = {"yellow": [.95, .7, .02, 1], "green": [.12, .55, .18, 1], "red": [.85, .08, .06, 1]}

    def body_piece(name, xyz, category, volume, rgba, kind, size):
        body = ET.SubElement(world, "body", name=name, pos=numbers(xyz))
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        geom(body, f"{name}_geom", "wood", body=name, dynamic=True, type=kind,
             size=numbers(size), density=str(density), rgba=numbers(rgba))
        objects.append({"name": name, "category": category, "mass_kg": volume*density,
                        "initial_position_m": list(xyz)})
        return body

    for row_index, row in enumerate(place["cylinder_rows"]):
        for column, x in enumerate(place["cylinder_x"]):
            name = f"Cylinder_{row['color'].title()}_{row_index*2+column+1:02d}"
            radius, h = dim["cylinder_diameter"]*.0005, dim["cylinder_height"]*.001
            xyz = [x*.001, row["y"]*.001, h/2+.00002]
            body_piece(name, xyz, row["category"], math.pi*radius**2*h, colors[row["color"]], "cylinder", [radius,h/2])
            ET.SubElement(world, "geom", name=f"{name}_mark", type="cylinder", size=f"{radius} 0.000001",
                          pos=numbers([*xyz[:2],.000001]), rgba="0.02 0.02 0.02 1", contype="0", conaffinity="0", group="1")
    rows = settings.get("kit_rows", list(range(1,len(place["kit_y"])+1)))
    kit_count = 0
    for row in rows:
        for x in place["kit_x"]:
            if kit_count >= settings["kits"]:
                break
            kit_count += 1
            size = np.array([dim["kit_marked_face"], dim["kit_marked_face"], dim["kit_height_as_placed"]])*.001
            b = body_piece(f"Medical_Kit_{kit_count:02d}", [x*.001,place["kit_y"][row-1]*.001,size[2]/2+.00002],
                           "medical_kit", float(np.prod(size)), [1,.96,.87,1], "box", size/2)
            for i, half in enumerate(([.01,.0025,.000001],[.0025,.01,.000001])):
                ET.SubElement(b,"geom",name=f"Kit_{kit_count:02d}_cross_{i}",type="box",size=numbers(half),
                              pos=numbers([0,0,size[2]/2+.000002]),rgba="0.84 0.035 0.025 1",contype="0",conaffinity="0",mass="0",group="1")
    for i, xy in enumerate(place["samples"][:settings["samples"]]):
        radius, h = dim["sample_diameter"]*.0005, dim["sample_thickness"]*.001
        b=body_piece(f"Biological_Sample_{i+1:02d}", [xy[0]*.001,xy[1]*.001,h/2+.00002],
                   "sample", math.pi*radius**2*h, [.98,.75,.06,1], "cylinder", [radius,h/2])
        ET.SubElement(b,"geom",name=f"Sample_{i+1:02d}_biohazard",type="mesh",mesh="biohazard",
                      pos=numbers([0,0,h/2+.00001]),rgba="1 1 1 1",contype="0",conaffinity="0",mass="0",group="1")
    for beam in place["containment_beams"][:settings["containment_beams"]]:
        size=np.array([dim["beam_width"],beam["length"],dim["beam_height"]])*.001
        b=body_piece(beam["name"], [beam["center"][0]*.001,beam["center"][1]*.001,size[2]/2+.00002],
                   "containment_beam", float(np.prod(size)), [.87,.66,.32,1], "box", size/2)
        ET.SubElement(b,"geom",name=f"{beam['name']}_biohazard",type="mesh",mesh="biohazard",
                      pos=numbers([0,0,size[2]/2+.00001]),rgba="1 1 1 1",contype="0",conaffinity="0",mass="0",group="1")
    lab = ET.SubElement(world,"body",name="Laboratory")
    prisms = lab_prisms(spec)
    for prism in prisms:
        name = prism["name"]
        vertices = prism["vertices"]
        center = vertices.mean(axis=0)
        faces = []
        for face in prism["faces"]:
            faces.extend([[face[0], face[i], face[i+1]] for i in range(1,len(face)-1)])
        ET.SubElement(asset,"mesh",name=name,vertex=numbers(vertices-center),face=" ".join(map(str,np.asarray(faces).ravel())))
        geom(lab,f"{name}_geom","wood",body="Laboratory",type="mesh",mesh=name,pos=numbers(center),rgba=".78 .84 .91 1")
    metadata["laboratory"] = {"convex_pieces":len(prisms),"max_hole_error_m":max(x["approximation_error_m"] for x in prisms)}
    if settings["fence"]:
        t, h = dim["frame_thickness"]*.001, dim["frame_height"]*.001
        walls = {"west":([-t/2,height/2,h/2],[t/2,height/2+t,h/2]),
                 "east":([width+t/2,height/2,h/2],[t/2,height/2+t,h/2]),
                 "south":([width/2,-t/2,h/2],[width/2,t/2,h/2]),
                 "north":([width/2,height+t/2,h/2],[width/2,t/2,h/2])}
        for side,(pos,size) in walls.items():
            geom(world,f"wall_{side}","wood",wall=side,type="box",pos=numbers(pos),size=numbers(size),rgba=".58 .40 .22 1")
    if robot:
        if robot == "competition":
            root.find("option").set("cone", "elliptic")
            root.find("option").set("solver", "Newton")
            root.find("option").set("impratio", "100")
            from .competition_robot import add_competition_robot
            metadata["robot"] = add_competition_robot(root,p["robot"])
        else:
            from .robot import add_robot
            metadata["robot"] = add_robot(world,actuator,sensor,p["robot"])
        for item in metadata["robot"]["colliders"]:
            geoms.append({**item,"dynamic":True,"wall":None,"body":item.get("body","reference_robot")})
            element=next(g for g in world.iter("geom") if g.get("name")==item["name"])
            values=material_pair(p,item["material"],"vinyl")
            element.set("friction",numbers([values[0],values[2],values[3]]))
            element.set("priority","1")
    if tape_mode != "none":
        _add_tape(root,world,contacts,spec,p,metadata,tape_mode)
    if tape_mode == "rigid":
        geoms.extend({"name":strip["name"],"material":"vinyl","body":"world","dynamic":False,"wall":None}
                     for strip in metadata["tape_strips"])
    # Explicit rigid pairs preserve independently measured surface coefficients.
    # This also makes each wall tunable without engine max-friction mixing.
    robot_geoms = {item["name"] for item in metadata.get("robot",{}).get("colliders",[])}
    for i,first in enumerate(geoms):
        for second in geoms[i+1:]:
            if first["body"]==second["body"] or not(first["dynamic"] or second["dynamic"]):
                continue
            if first["name"] in robot_geoms and second["name"] in robot_geoms:
                continue
            values = list(material_pair(p,first["material"],second["material"]))
            side=first["wall"] or second["wall"]
            if side:
                values=[v*p["wall_friction_scale"][side] for v in values]
            name=f"dry_{len(pairs):05d}"
            grip_contact = first.get("grip_pad",False) or second.get("grip_pad",False)
            pair_solref = ".003 1" if grip_contact else solref
            pair_solimp = ".90 .99 .0003" if grip_contact else solimp
            ET.SubElement(contacts,"pair",name=name,geom1=first["name"],geom2=second["name"],
                          condim="6",friction=numbers(friction5(values)),solref=pair_solref,solimp=pair_solimp,adhesion="0")
            pairs.append({"name":name,"geom1":first["name"],"geom2":second["name"],"parameters":values})
    metadata["rigid_geoms"] = geoms
    ET.indent(root)
    return ET.tostring(root,encoding="unicode"), metadata


def _add_tape(root,world,contacts,spec,profile,metadata,mode):
    tape=profile["tape"]
    thickness=spec["dimensions"]["tape_thickness"]*.001
    radius=thickness/2
    deform=ET.SubElement(root,"deformable") if mode=="flex" else None
    body_elements, strip_nodes = {}, {}
    strips=sorted(tape_strips(spec,tape["joins"]),key=lambda x:x["layer"])
    for strip in strips:
        name=strip["name"]
        if mode=="rigid":
            x0,y0,x1,y1=strip["bounds_m"]
            if tape["joins"] != "butt":
                raise ValueError("Rigid approximation supports butt joins only")
            ET.SubElement(world,"geom",name=name,type="box",size=numbers([(x1-x0)/2,(y1-y0)/2,radius]),
                          pos=numbers([(x0+x1)/2,(y0+y1)/2,radius]),rgba=".015 .015 .015 1",
                          friction=".4 .0001 .00001",priority="0")
            metadata["tape_strips"].append(strip)
            continue
        grid=rectangle_grid(strip["bounds_m"],tape["spacing_m"],radius)
        flat=grid["vertices"].copy()
        flat[:,2]=radius
        supports=[]
        bridge_nodes=[]
        # Upper overlap nodes follow real lower bodies. No world welds through lower tape.
        for index,point in enumerate(flat):
            support=None
            for region in strip["overlap_regions"]:
                x0,y0,x1,y1=region["bounds_m"]
                if x0-1e-10<=point[0]<=x1+1e-10 and y0-1e-10<=point[1]<=y1+1e-10:
                    lower=strip_nodes[region["support_name"]]
                    support=min(lower,key=lambda x:np.linalg.norm(np.asarray(x["flat"])[:2]-point[:2]))
                    point[2]=support["flat"][2]+thickness
                    break
            bridge=False
            if support is None:
                for region in strip["overlap_regions"]:
                    x0,y0,x1,y1=region["bounds_m"]
                    distance=math.hypot(max(x0-point[0],0,point[0]-x1),max(y0-point[1],0,point[1]-y1))
                    if distance <= tape["spacing_m"]:
                        # Lift the approach node before the support edge. A mesh
                        # triangle sloping through the edge starts inside lower tape.
                        point[2]=radius+thickness
                        bridge=True
                        break
            supports.append(support)
            bridge_nodes.append(bridge)
        rest=flat.copy()
        axis=grid["length_axis"]
        curvature=float(tape["rest_curvature_m_inv"])
        # Curve only the final 30 mm: modest residual end curl avoids rolling a metre-long reference strip through itself.
        s=np.maximum(flat[:,axis]-(flat[:,axis].max()-.03),0)
        if abs(curvature)>1e-12:
            rest[:,axis]+=np.sin(curvature*s)/curvature-s
            rest[:,2]+=(1-np.cos(curvature*s))/curvature
        areas=grid["areas"]*grid["nominal_area_m2"]/grid["mesh_area_m2"]
        names,records=[],[]
        for i,(point,area,support) in enumerate(zip(rest,areas,supports)):
            body_name=f"{name}_node_{i:04d}"
            names.append(body_name)
            body=ET.SubElement(world,"body",name=body_name,pos=numbers(point))
            body_elements[body_name]=body
            for j in range(3):
                joint=f"{body_name}_{'xyz'[j]}"
                ET.SubElement(body,"joint",name=joint,type="slide",axis=numbers(np.eye(3)[j]),damping="0")
                shift=flat[i,j]-point[j]
                if j==2 and tape["edge_lift_m"]:
                    shift+=tape["edge_lift_m"]*(s[i]/.03)**2
                if abs(shift)>1e-15:
                    metadata["flatten_qpos"][joint]=float(shift)
            mass=float(area*thickness*tape["density_kg_m3"])
            ET.SubElement(body,"inertial",pos="0 0 0",mass=str(mass),diaginertia=numbers([mass*1e-8]*3))
            bead=f"{body_name}_adhesive"
            ET.SubElement(body,"geom",name=bead,type="sphere",size=str(radius),mass="0",
                          contype="0",conaffinity="0",group="3",rgba="0 0 0 0")
            support_body="world"
            support_geom="floor"
            support_anchor=flat[i]-[0,0,radius]
            if support:
                support_body=support["body_name"]
                support_geom=f"{body_name}_support"
                offset=flat[i]-np.asarray(support["flat"])-[0,0,thickness]
                ET.SubElement(body_elements[support_body],"geom",name=support_geom,type="sphere",
                              pos=numbers(offset),size=str(radius),mass="0",contype="0",conaffinity="0",group="3",rgba="0 0 0 0")
                support_anchor=offset+[0,0,radius]
            pair=f"{body_name}_bond"
            force=tape["peak_traction_pa"]*area
            failure=2*tape["fracture_energy_j_m2"]/tape["peak_traction_pa"]
            ET.SubElement(contacts,"pair",name=pair,geom1=support_geom,geom2=bead,condim="3",
                          adhesion=str(force),gap=str(failure),margin="0",
                          friction=numbers([tape["adhesive_friction"]]*2+[0,0,0]),
                          solref=numbers([max(profile["timestep_s"]*2,.0004),1]),solimp=".99 .999 .00001")
            record={"body_name":body_name,"pair_name":pair,"area_m2":float(area),"flat":flat[i].tolist(),
                    "support_body_name":support_body,"support_anchor_local_m":support_anchor.tolist(),
                    "rest_relative_m":[0,0,radius],"strip_name":name,"vertex_index":i}
            if bridge_nodes[i]:
                record["initial_damage"]=1.0
                record["unbonded_bridge"]=True
            metadata["tape_nodes"].append(record)
            records.append(record)
        strip_nodes[name]=records
        flex=ET.SubElement(deform,"flex",name=name,dim="2",body=" ".join(names),
                           vertex=numbers(np.zeros_like(rest)),element=" ".join(map(str,grid["triangles"].ravel())),
                           radius=str(radius),rgba=".012 .014 .018 1",flatskin="true")
        vinyl=material_pair(profile,"vinyl","vinyl")
        ET.SubElement(flex,"contact",contype="2",conaffinity="2",condim="6",selfcollide="auto",internal="false",passive="false",
                      friction=numbers([vinyl[0],vinyl[2],vinyl[3]]),solref=numbers([max(profile["timestep_s"]*2,.0005),1]),
                      solimp=".99 .999 .00001",priority="0")
        ET.SubElement(flex,"elasticity",young=str(tape["young_pa"]),poisson=str(tape["poisson"]),
                      thickness=str(thickness),elastic2d="both",damping=str(tape["rayleigh_damping_s"]))
        metadata["tape_strips"].append({**strip,"node_count":len(names),"mass_kg":float(areas.sum()*thickness*tape["density_kg_m3"]),
                                       "triangles":grid["triangles"].tolist(),"body_names":names})


def export_arena(path, **kwargs):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    xml,metadata=build_arena(**kwargs)
    path.write_text(xml)
    path.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata
