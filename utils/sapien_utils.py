import numpy as np
import sapien.core as sapien
from gym import spaces
import numpy as np
from copy import deepcopy

def parse_urdf_config(config_dict: dict, scene: sapien.Scene) :
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = deepcopy(config_dict)

    # Create the global physical material for all links
    mtl_cfg = urdf_config.pop("material", None)
    if mtl_cfg is not None:
        urdf_config["material"] = scene.create_physical_material(**mtl_cfg)

    # Create link-specific physical materials
    materials = {}
    for k, v in urdf_config.pop("_materials", {}).items():
        materials[k] = scene.create_physical_material(**v)

    # Specify properties for links
    for link_config in urdf_config.get("link", {}).values():
        # Substitute with actual material
        link_config["material"] = materials[link_config["material"]]

    return urdf_config

def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for SAPIEN.

    Args:
        urdf_config (dict): dict passed to `sapien.URDFLoader.load`.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
            )

    allowed_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            # In fact, it should support specifying collision-shape-level materials.
            if kk not in allowed_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
                )

def get_entity_by_name(entities, name: str, is_unique=True):
    """Get a Sapien.Entity given the name.

    Args:
        entities (List[sapien.Entity]): entities (link, joint, ...) to query.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        sapien.Entity or List[sapien.Entity]:
            matched entity or entities. None if no matches.
    """
    matched_entities = [x for x in entities if x.get_name() == name]
    if len(matched_entities) > 1:
        if not is_unique:
            return matched_entities
        else:
            raise RuntimeError(f"Multiple entities with the same name {name}.")
    elif len(matched_entities) == 1:
        return matched_entities[0]
    else:
        return None
    
'''
1. obtain the name of each collision_shape via link.get_visual_bodies()
2. obtain the geometry of each collision shape via link.get_collision_shapes()
3. assume that orders of visual_bodies and collision_shapes are the same
'''
def get_part_mesh_and_pose(link : sapien.Link):
    link_visual_bodies = link.get_visual_bodies()
    link_collision_shapes = link.get_collision_shapes()

    part_vs = {}
    global_part_vs = {}
    part_fs = {}
    vid = 0

    visual_names = []
    for visual_body in link_visual_bodies :
        visual_names.append(visual_body.get_name())

    if len(link_collision_shapes) == len(link_visual_bodies) :
        for (visual_body, collision_shape) in zip(link_visual_bodies, link_collision_shapes):
            v = np.array(collision_shape.geometry.vertices, dtype=np.float32)
            f = np.array(collision_shape.geometry.indices, dtype=np.uint32).reshape(-1, 3)
            vscale = collision_shape.geometry.scale
            v[:, 0] *= vscale[0]
            v[:, 1] *= vscale[1]
            v[:, 2] *= vscale[2]

            ones = np.ones((v.shape[0], 1), dtype=np.float32)
            v_ones = np.concatenate([v, ones], axis=1)
            pose = collision_shape.get_local_pose()
            transmat = pose.to_transformation_matrix()
            v = (v_ones @ transmat.T)[:, :3]

            if visual_body.get_name() not in part_fs.keys():
                part_vs[visual_body.get_name()] = []
                part_fs[visual_body.get_name()] = []
                vid = 0
            else:
                vid = part_vs[visual_body.get_name()][0].shape[0]

            part_vs[visual_body.get_name()].append(v)
            part_vs[visual_body.get_name()] = [np.concatenate(part_vs[visual_body.get_name()], axis=0)]
            part_fs[visual_body.get_name()].append(f + vid)
            part_fs[visual_body.get_name()] = [np.concatenate(part_fs[visual_body.get_name()], axis=0)]
    else :

        assert(len(link_visual_bodies) == 1)

        for collision_shape in link_collision_shapes :

            v = np.array(collision_shape.geometry.vertices, dtype=np.float32)
            f = np.array(collision_shape.geometry.indices, dtype=np.uint32).reshape(-1, 3)
            vscale = collision_shape.geometry.scale
            v[:, 0] *= vscale[0]
            v[:, 1] *= vscale[1]
            v[:, 2] *= vscale[2]

            ones = np.ones((v.shape[0], 1), dtype=np.float32)
            v_ones = np.concatenate([v, ones], axis=1)
            pose = collision_shape.get_local_pose()
            transmat = pose.to_transformation_matrix()
            v = (v_ones @ transmat.T)[:, :3]

            name = visual_names[0]
            if name not in part_fs.keys():
                part_vs[name] = []
                part_fs[name] = []
                vid = 0
            else:
                vid = part_vs[visual_body.get_name()][0].shape[0]

            part_vs[name].append(v)
            part_vs[name] = [np.concatenate(part_vs[name], axis=0)]
            part_fs[name].append(f + vid)
            part_fs[name] = [np.concatenate(part_fs[name], axis=0)]

    for k in part_vs.keys():
        part_vs[k] = np.concatenate(part_vs[k], axis=0)
        part_fs[k] = np.concatenate(part_fs[k], axis=0)
    
    # if global_transform==True, return vertices in world frame
    part_transmat = link.get_pose().to_transformation_matrix()
    for k in part_vs.keys():
        ones = np.ones((part_vs[k].shape[0], 1), dtype=np.float32)
        vs_ones = np.concatenate([part_vs[k], ones], axis=1)
        global_part_vs[k] = (vs_ones @ part_transmat.T)[:, :3]
    
    return part_vs, global_part_vs, part_fs, part_transmat

def get_part_mesh(link : sapien.Link, global_transform=True):
    final_vs = []
    final_fs = []
    vid = 0
    vs = []
    for s in link.get_collision_shapes():
        v = np.array(s.geometry.vertices, dtype=np.float32)
        f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
        vscale = s.geometry.scale
        v[:, 0] *= vscale[0]
        v[:, 1] *= vscale[1]
        v[:, 2] *= vscale[2]
        ones = np.ones((v.shape[0], 1), dtype=np.float32)
        v_ones = np.concatenate([v, ones], axis=1)
        pose = s.get_local_pose()
        transmat = pose.to_transformation_matrix()
        v = (v_ones @ transmat.T)[:, :3]
        vs.append(v)
        final_fs.append(f + vid)
        vid += v.shape[0]
    part_transmat = None
    if len(vs) > 0:
        vs = np.concatenate(vs, axis=0)
        part_transmat = link.get_pose().to_transformation_matrix()
        if global_transform :
            ones = np.ones((vs.shape[0], 1), dtype=np.float32)
            vs_ones = np.concatenate([vs, ones], axis=1)
            vs = (vs_ones @ part_transmat.T)[:, :3]
        final_vs.append(vs)
    if(final_fs!=[] and final_fs!=[]):
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
    return final_vs, final_fs, part_transmat