"""Dimensioned challenge assets, in metres, with body-centred origins.

Dimensions follow the national challenge drawings, supplemented by the official
2026-2027 international rulebook. The lab hole pitch is inferred; its 60 mm
diameter is explicit in the international rulebook. Decorations are separate
child meshes so callers can omit them from physics collision geometry.
"""

import math

import bpy
from mathutils import Matrix

_SEGMENTS = 192  # Divisible by four so cardinal diameter bounds are exact.
_BIOHAZARD_MESH = None


def _mesh_object(name, vertices, faces, location, material, collection):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = location
    if material is not None:
        mesh.materials.append(material)
    return obj


def _box(name, size, location, material, collection):
    x, y, z = (v / 2 for v in size)
    vertices = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(3, 2, 1, 0), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    return _mesh_object(name, vertices, faces, location, material, collection)


def make_cylinder(name, diameter, height, location, material, collection):
    """Create a Z-axis cylinder, smooth sides and flat caps, without bevels."""
    if diameter <= 0 or height <= 0:
        raise ValueError("Cylinder diameter and height must be positive")
    n = _SEGMENTS
    radius = diameter / 2
    vertices = [(radius * math.cos(2 * math.pi * i / n),
                 radius * math.sin(2 * math.pi * i / n), z)
                for z in (-height / 2, height / 2) for i in range(n)]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces.extend((i, (i + 1) % n, (i + 1) % n + n, i + n)
                 for i in range(n))
    obj = _mesh_object(name, vertices, faces, location, material, collection)
    for polygon in obj.data.polygons:
        polygon.use_smooth = polygon.index >= 2
    return obj


def _parent_decoration(obj, body, local_location=(0, 0, 0)):
    obj.parent = body
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = local_location
    obj["decoration"] = True


def make_kit(name, location, body_material, cross_material, collection):
    """Return a 25 x 25 x 20 mm kit and its upward-facing red cross."""
    body = _box(name, (0.025, 0.025, 0.020), location, body_material, collection)
    # One concave polygon: 20 mm overall, both strokes exactly 5 mm wide.
    a, b = 0.010, 0.0025
    outline = [(-b, -a), (b, -a), (b, -b), (a, -b), (a, b), (b, b),
               (b, a), (-b, a), (-b, b), (-a, b), (-a, -b), (-b, -b)]
    cross = _mesh_object(name + "_cross", [(x, y, 0) for x, y in outline],
                         [tuple(range(12))], (0, 0, 0), cross_material, collection)
    _parent_decoration(cross, body, (0, 0, 0.01001))
    return body, [cross]


def _boolean(target, cutter, operation):
    """Bake a Boolean through the dependency graph without selecting objects."""
    modifier = target.modifiers.new("construction", "BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.context.view_layer.update()
    evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result = bpy.data.meshes.new_from_object(evaluated)
    target.modifiers.remove(modifier)
    old = target.data
    target.data = result
    if old.users == 0:
        bpy.data.meshes.remove(old)
    cutter_mesh = cutter.data
    bpy.data.objects.remove(cutter, do_unlink=True)
    if cutter_mesh.users == 0:
        bpy.data.meshes.remove(cutter_mesh)


def _make_biohazard_mesh(collection):
    """Construct the three crescents and separated inner ring from circles.

    Uses the NIH geometric proportions reproduced in Auburn's technical diagram:
    https://cws.auburn.edu/shared/content/files/1621/biohazard-history.pdf
    Basic unit A=1, outer radius 15 and centre radius 11, inner radius 10.5
    and centre radius 15, central hole radius 3, ring radii 10 and 13.5.
    The challenge drawing shows the symbol rotated 180 degrees and about 31 mm
    across its 56 mm disc. The vector construction keeps that orientation.
    """
    angles = [math.radians(a) for a in (-90, 30, 150)]

    def disc(radius, centre, depth=1):
        return make_cylinder("_symbol_construction", 2 * radius, depth,
                             (*centre, 0), None, collection)

    # Work at drawing-unit scale to avoid tolerance problems at micrometre depth.
    crescent = None
    for a in angles:
        part = disc(15, (11 * math.cos(a), 11 * math.sin(a)))
        if crescent is None:
            crescent = part
        else:
            _boolean(crescent, part, "UNION")
    for a in angles:
        _boolean(crescent, disc(10.5, (15 * math.cos(a), 15 * math.sin(a)), 3),
                 "DIFFERENCE")
    _boolean(crescent, disc(3, (0, 0), 3), "DIFFERENCE")
    for a in angles:
        # Narrow centre slots continue through each open lobe mouth.
        slot = _box("_symbol_slot", (1, 34, 3),
                    (17 * math.cos(a), 17 * math.sin(a), 0), None, collection)
        slot.rotation_euler.z = a - math.pi / 2
        _boolean(crescent, slot, "DIFFERENCE")
        # The opening at each outer tip is four basic units wide.
        tip = _box("_symbol_tip", (4, 16, 3),
                   (30 * math.cos(a), 30 * math.sin(a), 0), None, collection)
        tip.rotation_euler.z = a - math.pi / 2
        _boolean(crescent, tip, "DIFFERENCE")

    ring = disc(13.5, (0, 0))
    _boolean(ring, disc(10, (0, 0), 3), "DIFFERENCE")
    # Keep only ring portions inside the inner lobes, with a one-unit air gap.
    mask = None
    for a in angles:
        piece = disc(9.5, (15 * math.cos(a), 15 * math.sin(a)), 3)
        if mask is None:
            mask = piece
        else:
            _boolean(mask, piece, "UNION")
    _boolean(ring, mask, "INTERSECT")
    _boolean(crescent, ring, "UNION")

    # Boolean object origins differ: bake its complete matrix before centring.
    mesh = crescent.data
    mesh.transform(crescent.matrix_world)
    # Centre the bounding box in XY to centre the emblem on the sample disc.
    low = [min(v.co[i] for v in mesh.vertices) for i in (0, 1)]
    high = [max(v.co[i] for v in mesh.vertices) for i in (0, 1)]
    scale = 0.031 / (high[0] - low[0])
    for vertex in mesh.vertices:
        vertex.co.x = (vertex.co.x - (low[0] + high[0]) / 2) * scale
        vertex.co.y = (vertex.co.y - (low[1] + high[1]) / 2) * scale
        vertex.co.z *= 0.00001
    for polygon in mesh.polygons:
        polygon.use_smooth = False
    mesh.name = "biohazard_vector_mesh"
    bpy.data.objects.remove(crescent, do_unlink=True)
    return mesh


def _biohazard_decoration(name, body, top_z, material, collection):
    """Copy the shared vector emblem and attach it just above a body's top."""
    global _BIOHAZARD_MESH
    if _BIOHAZARD_MESH is None or _BIOHAZARD_MESH.name not in bpy.data.meshes:
        _BIOHAZARD_MESH = _make_biohazard_mesh(collection)
    mesh = _BIOHAZARD_MESH.copy()
    mesh.materials.clear()
    if material is not None:
        mesh.materials.append(material)
    symbol = bpy.data.objects.new(name + "_biohazard", mesh)
    collection.objects.link(symbol)
    _parent_decoration(symbol, body, (0, 0, top_z + 0.00001))
    return symbol


def make_sample(name, location, body_material, symbol_material, collection):
    """Return a 56 mm diameter x 5 mm sample with a white biohazard mesh."""
    body = make_cylinder(name, 0.056, 0.005, location, body_material, collection)
    symbol = _biohazard_decoration(name, body, 0.0025, symbol_material, collection)
    return body, [symbol]


def make_containment_beam(name, center_m, length_m, bodymat, emblemmat, col):
    """Return a 60 mm X x length Y x 20 mm Z wooden containment beam.

    The official international rulebook p10 specifies 250 mm and 280 mm lengths.
    The body has exact rectangular bounds; its light biohazard emblem faces up.
    """
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("Containment beam length must be positive and finite")
    body = _box(name, (0.060, length_m, 0.020), center_m, bodymat, col)
    symbol = _biohazard_decoration(name, body, 0.010, emblemmat, col)
    return body, [symbol]


def make_lab_plate(name, location, material, collection,
                   hole_diameter=0.060, hole_pitch=0.100):
    """Return a 150 x 345 x 3 mm plate with three genuine through holes.

    X is the 150 mm width; holes lie along Y at -pitch, 0, +pitch. Diameter
    is specified in the international rulebook p9; pitch is a drawing estimate.
    Both remain configurable independently in metres. National plate length
    345 mm takes precedence over the international drawing's 440 mm.
    """
    if not 0 < hole_diameter < 0.150:
        raise ValueError("Hole diameter must be between zero and plate width")
    if hole_pitch <= hole_diameter or hole_pitch + hole_diameter / 2 >= 0.1725:
        raise ValueError("Holes must be separate and remain inside the plate")
    plate = _box(name, (150, 345, 3), (0, 0, 0), material, collection)
    for y in (-hole_pitch * 1000, 0, hole_pitch * 1000):
        cutter = make_cylinder("_lab_hole", hole_diameter * 1000, 9,
                               (0, y, 0), None, collection)
        _boolean(plate, cutter, "DIFFERENCE")
    for vertex in plate.data.vertices:
        vertex.co *= 0.001
    plate.data.update()
    plate.location = location
    # Smooth only cylindrical hole walls; rectangular perimeter faces stay flat.
    for polygon in plate.data.polygons:
        centre = polygon.center
        radial_distance = min(math.hypot(centre.x, centre.y - y)
                              for y in (-hole_pitch, 0, hole_pitch))
        polygon.use_smooth = (abs(polygon.normal.z) < 0.1
                              and abs(radial_distance - hole_diameter / 2) < 0.0001)
    plate["hole_diameter_m"] = hole_diameter
    plate["hole_pitch_m"] = hole_pitch
    plate["hole_diameter_inferred"] = False
    plate["hole_pitch_inferred"] = True
    return plate
