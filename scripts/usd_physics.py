"""Author portable USD physics after Blender export, without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

CONTACT_OFFSET_M = 0.0005
REST_OFFSET_M = 0.0
STATIC_FRICTION = 0.6
DYNAMIC_FRICTION = 0.5
DEFAULT_MASS_KG = 0.02
VISUAL_ROLES = {"zone_visual", "decoration", "marking", "cross", "biohazard"}
EXACT_STATIC_ROLES = {"floor", "lab_plate", "tape"}


def _attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind, custom=True).Set(value)


def _schema_attributes(prim, schema, attributes):
    """Keep PhysX schema tokens when the export host has only vanilla OpenUSD."""
    try:
        from pxr import PhysxSchema
    except ImportError:
        applied = prim.GetMetadata("apiSchemas")
        tokens = list(applied.GetAddedOrExplicitItems()) if applied else []
        if schema not in tokens:
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(tokens + [schema]))
    else:
        getattr(PhysxSchema, schema).Apply(prim)
    for name, kind, value in attributes:
        prim.CreateAttribute(name, kind, custom=False).Set(value)


def _resolve_objects(stage, manifest):
    prims = list(stage.Traverse())
    result = {}
    for item in manifest:
        name = item["name"]
        if name in result:
            raise ValueError(f"Duplicate object name in manifest: {name}")
        candidates = [p for p in prims if any(
            p.GetAttribute(key).Get() == name
            for key in ("userProperties:arena_source_name", "arena_source_name", "arena:sourceName")
        )]
        if not candidates:
            candidates = [p for p in prims if p.GetName() == Tf.MakeValidIdentifier(name)
                          and p.IsA(UsdGeom.Xformable)]
        # Blender can copy properties to both the object and its mesh child.
        candidates = [p for p in candidates if not any(
            p != other and p.GetPath().HasPrefix(other.GetPath()) for other in candidates
        )]
        if len(candidates) != 1:
            paths = [str(p.GetPath()) for p in candidates]
            raise ValueError(f"Expected one exported prim for {name!r}, found {paths}")
        if not candidates[0].GetPath().HasPrefix(Sdf.Path("/Arena")):
            raise ValueError(f"Object {name!r} is outside /Arena")
        result[name] = candidates[0]
    return result


def _own_geometry(prim, object_paths):
    return [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Gprim) and not any(
        p.GetPath().HasPrefix(path) and path != prim.GetPath()
        for path in object_paths if path.HasPrefix(prim.GetPath())
    )]


def _set_units(stage):
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)


def configure_usd(path, manifest, standalone_path=None):
    """Configure /Arena from object records and return a JSON-serializable report.

    body_type is static, dynamic, or none. collision='none' explicitly disables
    collision; omission on static bodies uses the original triangle mesh.
    Dynamic meshes default to convexHull. All mass/friction values are design
    assumptions unless a manifest record provides mass_source='measured'.
    """
    path = Path(path).resolve()
    manifest = list(manifest)
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetPrimAtPath("/Arena"):
        raise ValueError(f"Export must contain /Arena: {path}")
    objects = _resolve_objects(stage, manifest)
    object_paths = {p.GetPath() for p in objects.values()}
    _set_units(stage)
    arena = stage.GetPrimAtPath("/Arena")
    stage.SetDefaultPrim(arena)
    arena.SetCustomDataByKey("physicsAssumptions", "Masses and friction are unmeasured design defaults; tune against hardware.")
    for prim in reversed(list(stage.Traverse())):
        if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdLux.LightAPI):
            stage.RemovePrim(prim.GetPath())

    UsdGeom.Scope.Define(stage, "/Arena/PhysicsMaterials")
    material = UsdShade.Material.Define(stage, "/Arena/PhysicsMaterials/Default")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(STATIC_FRICTION)
    physics_material.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    physics_material.CreateRestitutionAttr(0.0)
    material.GetPrim().SetCustomDataByKey("assumption", "Unmeasured friction and restitution defaults")

    for item in manifest:
        prim = objects[item["name"]]
        role = item["role"]
        body = item.get("body_type", "none")
        if body not in {"static", "dynamic", "none"}:
            raise ValueError(f"Invalid body_type for {item['name']}: {body}")
        if role in VISUAL_ROLES and body != "none":
            raise ValueError(f"Visual object {item['name']} must use body_type='none'")
        if role == "tape" and body == "dynamic":
            raise ValueError(f"Tape {item['name']} must use body_type='static' or 'none'")
        for key, value in {"sourceName": item["name"], "role": role,
                           "category": item.get("category", role), "bodyType": body}.items():
            _attr(prim, f"arena:{key}", Sdf.ValueTypeNames.String, value)
        for key in ("dimensions_m", "position_m"):
            if key in item:
                values = item[key]
                if len(values) != 3 or not all(math.isfinite(float(x)) for x in values):
                    raise ValueError(f"Invalid {key} for {item['name']}")
                _attr(prim, f"arena:{key}", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*values))
        if item.get("parent"):
            parent = objects[item["parent"]]
            if not prim.GetPath().HasPrefix(parent.GetPath()) or prim == parent:
                raise ValueError(f"{item['name']} must be exported under {item['parent']}")
        geometry = _own_geometry(prim, object_paths)
        # Clear existing authored physics on this object only. Child objects are
        # independently configured from their own manifest entries.
        for target in {prim, *geometry}:
            for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI,
                           UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
                target.RemoveAPI(schema)
        if body == "none":
            continue
        if not geometry:
            raise ValueError(f"No collision geometry for {item['name']}")
        approximation = item.get("collision", "convexHull" if body == "dynamic" else "mesh")
        if approximation == "none":
            raise ValueError(f"Physical body {item['name']} has collision='none'; use body_type='none'")
        if approximation not in {"mesh", "convexHull", "boundingCube"}:
            raise ValueError(f"Unsupported collision approximation: {approximation}")
        if role in EXACT_STATIC_ROLES:
            approximation = "mesh"
        if body == "dynamic":
            if approximation == "mesh":
                raise ValueError(f"Dynamic triangle mesh unsupported: {item['name']}")
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateRigidBodyEnabledAttr(True)
            rigid.CreateKinematicEnabledAttr(False)
            mass = float(item.get("mass_kg", DEFAULT_MASS_KG))
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"Mass must be positive for {item['name']}")
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
            _attr(prim, "arena:massIsAssumption", Sdf.ValueTypeNames.Bool,
                  item.get("mass_source") != "measured")
            _schema_attributes(prim, "PhysxRigidBodyAPI", [
                ("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool, True),
                ("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 16),
                ("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 4),
            ])
        for geom in geometry:
            UsdPhysics.CollisionAPI.Apply(geom).CreateCollisionEnabledAttr(True)
            if geom.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr(
                    "none" if approximation == "mesh" else approximation)
            UsdShade.MaterialBindingAPI.Apply(geom).Bind(material, materialPurpose="physics")
            _schema_attributes(geom, "PhysxCollisionAPI", [
                ("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, CONTACT_OFFSET_M),
                ("physxCollision:restOffset", Sdf.ValueTypeNames.Float, REST_OFFSET_M),
            ])
    stage.GetRootLayer().Save()
    report = verify_usd(path, manifest)
    if standalone_path:
        create_standalone(path, standalone_path)
        report["standalone"] = verify_usd(standalone_path, standalone=True)
    return report


def create_standalone(asset_path, standalone_path):
    """Reference the reusable asset and add one scene and preview lighting."""
    standalone_path = Path(standalone_path).resolve()
    if standalone_path == Path(asset_path).resolve():
        raise ValueError("Standalone path must differ from asset path")
    standalone_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(standalone_path))
    _set_units(stage)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    arena = UsdGeom.Xform.Define(stage, "/World/Arena")
    relative_asset = os.path.relpath(Path(asset_path).resolve(), standalone_path.parent)
    arena.GetPrim().GetReferences().AddReference(relative_asset, "/Arena")
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    _schema_attributes(physics.GetPrim(), "PhysxSceneAPI", [
        ("physxScene:enableCCD", Sdf.ValueTypeNames.Bool, True),
        ("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt, 240),
        ("physxScene:solverType", Sdf.ValueTypeNames.Token, "TGS"),
    ])
    dome = UsdLux.DomeLight.Define(stage, "/World/PreviewDome")
    dome.CreateIntensityAttr(700)
    sun = UsdLux.DistantLight.Define(stage, "/World/PreviewSun")
    sun.CreateIntensityAttr(2000)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(30, -25, -20))
    stage.GetRootLayer().Save()


def verify_usd(path, manifest=None, *, standalone=False):
    """Fail on missing physics, collisions on decals, or broken references."""
    stage = Usd.Stage.Open(str(Path(path).resolve()))
    if not stage:
        raise ValueError(f"Cannot open USD: {path}")
    errors = []
    usd_validation = {"status": "unavailable", "validators": 0, "warnings": []}
    try:
        from pxr import UsdValidation
    except ImportError:
        pass
    else:
        validators = UsdValidation.ValidationRegistry().GetOrLoadAllValidators()
        diagnostics = UsdValidation.ValidationContext(validators).Validate(stage)
        usd_validation.update(status="passed", validators=len(validators))
        for diagnostic in diagnostics:
            message = f"{diagnostic.GetIdentifier()}: {diagnostic.GetMessage()}"
            if diagnostic.GetType() == UsdValidation.ValidationErrorType.Error:
                errors.append(message)
            else:
                usd_validation["warnings"].append(message)
    if stage.GetCompositionErrors():
        errors.extend(str(x) for x in stage.GetCompositionErrors())
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1:
        errors.append("Stage must use meters and kilograms")
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        errors.append("Stage must be Z up")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or default_prim.GetName() != ("World" if standalone else "Arena"):
        errors.append("Incorrect defaultPrim")
    prims = list(stage.Traverse())
    rigid = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    scenes = [p for p in prims if p.IsA(UsdPhysics.Scene)]
    lights = [p for p in prims if p.HasAPI(UsdLux.LightAPI)]
    if len(scenes) != int(standalone) or (not standalone and lights):
        errors.append("Reusable asset must have no scene/lights; standalone must have one scene")
    tagged = [p for p in prims if p.GetAttribute("arena:sourceName")]
    object_paths = {p.GetPath() for p in tagged}
    for prim in tagged:
        body = prim.GetAttribute("arena:bodyType").Get()
        role = prim.GetAttribute("arena:role").Get()
        if role == "tape" and body == "dynamic":
            errors.append(f"Tape must not be dynamic: {prim.GetPath()}")
        shapes = _own_geometry(prim, object_paths)
        is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if is_rigid != (body == "dynamic"):
            errors.append(f"Wrong rigid-body state: {prim.GetPath()}")
        for shape in shapes:
            if shape.HasAPI(UsdPhysics.CollisionAPI) != (body in {"static", "dynamic"}):
                errors.append(f"Wrong collision state: {shape.GetPath()}")
            if role in EXACT_STATIC_ROLES and body == "static" and shape.IsA(UsdGeom.Mesh):
                if UsdPhysics.MeshCollisionAPI(shape).GetApproximationAttr().Get() != "none":
                    errors.append(f"Exact collision mesh required: {shape.GetPath()}")
    for prim in rigid:
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if mass is None or not math.isfinite(mass) or mass <= 0:
            errors.append(f"Missing positive mass: {prim.GetPath()}")
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                errors.append(f"Nested rigid body: {prim.GetPath()}")
            parent = parent.GetParent()
    for prim in colliders:
        binding, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not binding or not binding.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            errors.append(f"Missing physics material: {prim.GetPath()}")
        contact = prim.GetAttribute("physxCollision:contactOffset").Get()
        rest = prim.GetAttribute("physxCollision:restOffset").Get()
        if contact is None or rest is None or not (0 <= rest < contact <= 0.001):
            errors.append(f"Invalid small-part collision offsets: {prim.GetPath()}")
    if manifest is not None:
        objects = _resolve_objects(stage, manifest)
        if len(tagged) != len(manifest):
            errors.append("Manifest and tagged object counts differ")
        for item in manifest:
            if objects[item["name"]].GetAttribute("arena:bodyType").Get() != item.get("body_type", "none"):
                errors.append(f"Manifest body type mismatch: {item['name']}")
    if errors:
        raise ValueError("USD verification failed:\n" + "\n".join(errors))
    return {"path": str(Path(path).resolve()), "objects": len(tagged),
            "rigid_bodies": len(rigid), "colliders": len(colliders),
            "physics_scenes": len(scenes), "lights": len(lights),
            "openusd_validation": usd_validation,
            "static_validation": "passed", "isaac_runtime_validation": "not_run",
            "rigid_body_paths": [str(p.GetPath()) for p in rigid]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text()) if args.manifest else None
    if isinstance(records, dict):
        records = records["objects"]
    print(json.dumps(verify_usd(args.usd, records, standalone=args.standalone), indent=2))
