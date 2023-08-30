#
# Joachim Tesch, Max Planck Institute for Intelligent Systems, Perceiving Systems
#
# Render OBJ sequences from folder or cutlist into images
#
# Version: 20191030
#
# Notes:
#
#   + Expects OBJ files which show up correctly when object rotation is set to identity after OBJ import
#     + Can be adjusted in render_file()
#
#   + Cycles (--render cycles)
#     + Soft shadows
#     + Wireframe overlay rendering
#     + Use GPU rendering if possible: --use_gpu 1
#     + Fast low-quality preview render: --quality_preview 1
#     + Slow high-quality final render: --quality_preview 0
#
#   + Code can also run from Blender application Text Editor via 'Run Script'
#

# pylint: disable=invalid-name
import os
import sys
import time
import trimesh
import argparse
import numpy as np
from math import radians
from loguru import logger
import bpy
import mathutils
import skimage.io as io


##################################################
# Globals
##################################################

render_only_one_image = False

camera_location = (-1.0, -0.0, 0.3)

diffuse_color = [0.85882353, 0.74117647, 0.65098039, 1.0]  # (0.4, 0.6, 1.0, 1.0)

source_fps = 25
tracking_camera_pelvis_id = 3500
views = 1

# Command line parameters
input_path = "/tmp/bpyrender/render_movie_images/input/"
output_path = "/tmp/bpyrender/render_movie_images/render/"

object_height = 0.0

render = "cycles"

use_gpu = True

fps = 30

quality_preview = False

samples_preview = 32
samples_final = 256

resolution_x = 1080
resolution_y = 1080

shadows = True

smooth = True

wireframe = False
line_thickness = 0.3
quads = False

object_transparent = False
mouth_transparent = True

compositor_background_image = False
compositor_image_scale = 1.0
compositor_alpha = 0.7

tracking_camera = False
tracking_camera_pelvis = False

floor = True

verbose = False

table_position = [0.0, 0.0, 0.0]

##################################################
# Helper functions
##################################################


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)),
        axis=2,
    )

    return r_mat


def get_world_mesh_list(planeWidth=4, axisHeight=0.7, axisRadius=0.02, add_plane=True):
    groundColor = [220, 220, 220, 255]  # face_colors: [R, G, B, transparency]
    xColor = [255, 0, 0, 128]
    yColor = [0, 255, 0, 128]
    zColor = [0, 0, 255, 128]

    if add_plane:
        ground = trimesh.primitives.Box(
            center=[0, 0, -0.0001], extents=[planeWidth, planeWidth, 0.0002]
        )
        ground.visual.face_colors = groundColor

    xAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    xAxis.apply_transform(
        matrix=np.mat(
            ((0, 0, 1, axisHeight / 2), (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 1))
        )
    )
    xAxis.visual.face_colors = xColor
    yAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    yAxis.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 0, -1, axisHeight / 2), (0, 1, 0, 0), (0, 0, 0, 1))
        )
    )
    yAxis.visual.face_colors = yColor
    zAxis = trimesh.primitives.Cylinder(
        radius=axisRadius,
        height=axisHeight,
    )
    zAxis.apply_transform(
        matrix=np.mat(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, axisHeight / 2), (0, 0, 0, 1))
        )
    )
    zAxis.visual.face_colors = zColor
    xBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    xBox.apply_translation((axisHeight, 0, 0))
    xBox.visual.face_colors = xColor
    yBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    yBox.apply_translation((0, axisHeight, 0))
    yBox.visual.face_colors = yColor
    zBox = trimesh.primitives.Box(
        extents=[axisRadius * 3, axisRadius * 3, axisRadius * 3]
    )
    zBox.apply_translation((0, 0, axisHeight))
    zBox.visual.face_colors = zColor
    if add_plane:
        worldMeshList = [ground, xAxis, yAxis, zAxis, xBox, yBox, zBox]
    else:
        worldMeshList = [xAxis, yAxis, zAxis, xBox, yBox, zBox]
    return worldMeshList


def add_world_mesh(loc, name="world_mesh"):
    world_mesh = get_world_mesh_list(add_plane=False, axisHeight=0.1, axisRadius=0.002)
    world_mesh = trimesh.util.concatenate(world_mesh)

    m = bpy.data.meshes.new(name)
    world_mesh_obj = bpy.data.objects.new(m.name, m)
    world_mesh_obj.location = loc
    col = bpy.data.collections.get("Collection")
    col.objects.link(world_mesh_obj)
    bpy.context.view_layer.objects.active = world_mesh_obj
    m.from_pydata(world_mesh.vertices.tolist(), [], world_mesh.faces.tolist())


def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices) * 3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))


def add_table_mesh(table_id=1, obj_location=None):
    # global table_position

    bpy.ops.import_scene.obj(filepath="table_mesh.obj")

    table_object = bpy.data.objects["table_mesh"]
    mat = (
        mathutils.Matrix.Translation(table_object.location)
        @ mathutils.Euler(table_object.rotation_euler, "XYZ").to_matrix().to_4x4()
    )
    mat = np.array(mat)
    verts = read_verts(table_object.data)
    wverts = trimesh.transform_points(verts, mat)
    table_top_center = [
        (wverts[:, 0].min() + wverts[:, 0].max()) / 2,
        (wverts[:, 1].min() + wverts[:, 1].max()) / 2,
        wverts[:, 2].max(),
    ]

    table_location = [0.0, 0.0, 0.0]
    if obj_location is not None:
        # table_top_center = np.array(
        #     [table_object.dimensions[0] / 2, -table_object.dimensions[1] / 2, table_object.dimensions[2]]
        # )
        table_location[0] = obj_location[0] - table_top_center[0]
        table_location[1] = obj_location[1] - table_top_center[1]
        table_location[2] = obj_location[2] - table_top_center[2]

    # add_world_mesh(table_location, name="table_axis")
    # add_world_mesh(table_location[:2]+[table_object.dimensions[2]], name="table_axis")
    # add_world_mesh(table_location[:2] + [table_object.dimensions[2] + table_location[2]], name="table_axis_2")
    # print(table_location)
    # print(table_top_center)
    # print(obj_location)
    # mport IPython; IPython.embed(); exit()
    table_object.location = table_location


def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)


def look_at_blender(obj_camera, point):
    """
    # Test
    obj_camera = bpy.data.objects["Camera"]
    obj_other = bpy.data.objects["Cube"]

    obj_camera.location = (5.0, 2.0, 3.0)
    look_at(obj_camera, obj_other.matrix_world.to_translation())
    """
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = mathutils.Vector(point) - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat("-Z", "Y")

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def setup_diffuse_transparent_material(
    target, color, object_transparent, backface_transparent
):
    """Sets up diffuse/transparent material with backface culling in cycles"""

    mat = target.active_material
    if mat is None:
        # Create material
        mat = bpy.data.materials.new(name="Material")
        target.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    node_geometry = nodes.new("ShaderNodeNewGeometry")

    node_diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    node_diffuse.inputs[0].default_value = color

    node_transparent = nodes.new("ShaderNodeBsdfTransparent")
    node_transparent.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

    node_emission = nodes.new("ShaderNodeEmission")
    node_emission.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)

    node_mix = nodes.new(type="ShaderNodeMixShader")
    if object_transparent:
        node_mix.inputs[0].default_value = color[-1]  # 1.0
    else:
        node_mix.inputs[0].default_value = 0.0

    node_mix_mouth = nodes.new(type="ShaderNodeMixShader")
    if object_transparent or backface_transparent:
        node_mix_mouth.inputs[0].default_value = color[-1]  # 1.0
    else:
        node_mix_mouth.inputs[0].default_value = 0.0

    node_mix_backface = nodes.new(type="ShaderNodeMixShader")

    node_output = nodes.new(type="ShaderNodeOutputMaterial")

    links = mat.node_tree.links

    links.new(node_geometry.outputs[6], node_mix_backface.inputs[0])

    links.new(node_diffuse.outputs[0], node_mix.inputs[1])
    links.new(node_transparent.outputs[0], node_mix.inputs[2])
    links.new(node_mix.outputs[0], node_mix_backface.inputs[1])

    links.new(node_emission.outputs[0], node_mix_mouth.inputs[1])
    links.new(node_transparent.outputs[0], node_mix_mouth.inputs[2])
    links.new(node_mix_mouth.outputs[0], node_mix_backface.inputs[2])

    links.new(node_mix_backface.outputs[0], node_output.inputs[0])
    return


def setup_diffuse_grid_material(target, color1, color2, scale):
    """Sets up diffuse grid material in cycles"""

    mat = target.data.materials.get("MaterialDiffuseGrid")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="MaterialDiffuseGrid")
        target.data.materials.append(mat)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    nodeGrid = nodes.new("ShaderNodeTexChecker")
    nodeGrid.inputs[1].default_value = color1
    nodeGrid.inputs[2].default_value = color2
    nodeGrid.inputs[3].default_value = scale

    node_diffuse = nodes.new("ShaderNodeBsdfDiffuse")

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    links = mat.node_tree.links
    links.new(nodeGrid.outputs[0], node_diffuse.inputs[0])
    links.new(node_diffuse.outputs[0], node_output.inputs[0])
    return


######################################################################
# Setup scene for rendering
######################################################################


def setup_scene():
    global render
    global quality_preview
    global resolution_x
    global resolution_y
    global camera_location
    global shadows
    global wireframe
    global line_thickness
    global compositor_background_image
    global tracking_camera
    global floor
    global use_gpu

    scene = bpy.data.scenes["Scene"]

    if render == "cycles":
        scene.render.engine = "CYCLES"

    ###########################
    # Engine independent setup
    ###########################

    # Remove default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100

    # Setup camera
    camera = bpy.data.objects["Camera"]
    camera.location = camera_location
    camera.rotation_euler = (radians(82.5), 0.0, 0)
    bpy.data.cameras["Camera"].lens = 35
    if tracking_camera:
        bpy.context.view_layer.objects.active = camera
        if camera.constraints.active is None:
            bpy.ops.object.constraint_add(type="COPY_LOCATION")
            camera.constraints["Copy Location"].use_z = False
            camera.constraints["Copy Location"].use_offset = True
            bpy.ops.object.constraint_add(type="DAMPED_TRACK")
            camera.constraints["Damped Track"].track_axis = "TRACK_NEGATIVE_Z"
            camera.constraints["Damped Track"].influence = 0.5

    # add table
    # add_table(1)

    if floor:
        if "Plane" not in bpy.data.objects:
            bpy.ops.mesh.primitive_plane_add()

            # Fix UV map
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.uv.unwrap()
            bpy.ops.object.mode_set(mode="OBJECT")

            plane = bpy.context.active_object
            plane.location = (0.0, 0.0, 0.0)
            plane.scale = (10.0, 10.0, 1)

        else:
            bpy.data.objects["Plane"].hide_render = False
            plane = bpy.data.objects["Plane"]
    else:
        if "Plane" in bpy.data.objects:
            bpy.data.objects["Plane"].hide_render = True

    # Engine specific setup

    ######################################################################
    # Cycles
    ######################################################################

    if render == "cycles":
        scene.render.film_transparent = True
        if quality_preview:
            scene.cycles.samples = samples_preview
        else:
            scene.cycles.samples = samples_final

        # Setup Cycles CUDA GPU acceleration if requested
        if use_gpu:
            print("Activating GPU acceleration")
            bpy.context.preferences.addons[
                "cycles"
            ].preferences.compute_device_type = "CUDA"

            if bpy.app.version[0] >= 3:
                cuda_devices = bpy.context.preferences.addons[
                    "cycles"
                ].preferences.get_devices_for_type(compute_device_type="CUDA")
            else:
                (cuda_devices, opencl_devices) = bpy.context.preferences.addons[
                    "cycles"
                ].preferences.get_devices()

            if len(cuda_devices) < 1:
                print("ERROR: CUDA GPU acceleration not available")
                sys.exit(1)

            for cuda_device in cuda_devices:
                if cuda_device.type == "CUDA":
                    cuda_device.use = True
                    print("Using CUDA device: " + str(cuda_device.name))
                else:
                    cuda_device.use = False
                    print("Ignoring device: " + str(cuda_device.name))

            scene.cycles.device = "GPU"
            if bpy.app.version[0] < 3:
                scene.render.tile_x = 256
                scene.render.tile_y = 256
        else:
            scene.cycles.device = "CPU"
            if bpy.app.version[0] < 3:
                scene.render.tile_x = 64
                scene.render.tile_y = 64

        if floor:
            # color0=[0.8, 0.9, 0.9], color1=[0.6, 0.7, 0.7]
            # Ground plane
            setup_diffuse_grid_material(
                plane, (1.0, 1.0, 1.0, 1.0), (0.6, 0.7, 0.7, 1.0), 30
            )
            # setup_diffuse_grid_material(plane, (0.8, 0.9, 0.9, 1.0), (0.6, 0.7, 0.7, 1.0), 20)
            # Exclude plane from diffuse cycles contribution to avoid bright pixel noise in body rendering
            if bpy.app.version[0] < 3:
                plane.cycles_visibility.diffuse = False

        # Setup freestyle mode for wireframe overlay rendering
        if wireframe:
            scene.render.use_freestyle = True
            scene.render.line_thickness = line_thickness
            bpy.context.view_layer.freestyle_settings.linesets[
                0
            ].select_edge_mark = True

            # Disable border edges so that we don't see contour of shadow catcher plane
            bpy.context.view_layer.freestyle_settings.linesets[0].select_border = False

            if floor:
                # Unmark freestyle edges of plane
                bpy.context.view_layer.objects.active = plane
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.mark_freestyle_edge(clear=True)
                bpy.ops.object.mode_set(mode="OBJECT")

        else:
            scene.render.use_freestyle = False
    else:
        print("ERROR: Undefined render engine" + render)
        sys.exit(1)

    if compositor_background_image:
        # Setup compositing when using background image
        setup_compositing()
    else:
        # Output transparent image when no background is used
        scene.render.image_settings.color_mode = "RGBA"
    return scene


##################################################


def setup_compositing():
    global compositor_image_scale
    global compositor_alpha

    # Node editor compositing setup
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Create input image node
    image_node = tree.nodes.new(type="CompositorNodeImage")

    scale_node = tree.nodes.new(type="CompositorNodeScale")
    scale_node.inputs[1].default_value = compositor_image_scale
    scale_node.inputs[2].default_value = compositor_image_scale

    blend_node = tree.nodes.new(type="CompositorNodeAlphaOver")
    blend_node.inputs[0].default_value = compositor_alpha

    # Link nodes
    links = tree.links
    links.new(image_node.outputs[0], scale_node.inputs[0])

    links.new(scale_node.outputs[0], blend_node.inputs[1])
    links.new(tree.nodes["Render Layers"].outputs[0], blend_node.inputs[2])

    links.new(blend_node.outputs[0], tree.nodes["Composite"].inputs[0])


######################################################################
# Render
######################################################################


def render_file(input_file, input_dir, output_file, output_dir, yaw):
    """Render image of given model file"""
    global object_height
    global render
    global smooth
    global object_transparent
    global mouth_transparent
    global compositor_background_image
    global quads
    global verbose

    path = input_dir + input_file

    # Import object into scene
    bpy.ops.import_scene.obj(filepath=path)
    object = bpy.context.selected_objects[0]
    object.location = (0.0, 0.0, object_height)
    object.rotation_euler = (
        radians(0.0),
        0.0,
        radians(yaw),
    )  # Adjust if your imported OBJ shows up wrong

    if quads:
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.object.mode_set(mode="OBJECT")

    if smooth:
        bpy.ops.object.shade_smooth()

    if render == "cycles":
        # Mark freestyle edges
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.mark_freestyle_edge(clear=False)
        bpy.ops.object.mode_set(mode="OBJECT")

    if tracking_camera:
        # Create vertex group so that camera can track the mesh vertices instead if pivot
        bpy.context.view_layer.objects.active = object
        bpy.ops.object.vertex_group_add()
        bpy.ops.object.mode_set(mode="EDIT")

        if tracking_camera_pelvis:
            # Track only pelvis id by deselecting all vertices and then selecting only pelvis id vertex
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")
            object.data.vertices[tracking_camera_pelvis_id].select = True
            bpy.ops.object.mode_set(mode="EDIT")

        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.data.objects["Camera"].constraints["Copy Location"].target = object
        bpy.data.objects["Camera"].constraints["Copy Location"].subtarget = "Group"
        bpy.data.objects["Camera"].constraints["Damped Track"].target = object
        bpy.data.objects["Camera"].constraints["Damped Track"].subtarget = "Group"

        # import IPython; IPython.embed(); exit()
        # print(bpy.data.objects['Camera'])

        setup_diffuse_transparent_material(
            object, diffuse_color, object_transparent, mouth_transparent
        )

    if compositor_background_image:
        # Set background image
        imagePath = input_dir + input_file.replace(".obj", "_original.png")
        bpy.context.scene.node_tree.nodes["Image"].image = bpy.data.images.load(
            imagePath
        )

    # Render
    bpy.context.scene.render.filepath = os.path.join(output_dir, output_file)

    if verbose:
        # Silence console output of bpy.ops.render.render by redirecting stdout to /dev/null
        sys.stdout.flush()
        old = os.dup(1)
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)

    # Render
    bpy.ops.render.render(write_still=True)

    if verbose:
        # Remove temporary output redirection
        sys.stdout.flush()
        os.close(1)
        os.dup(old)
        os.close(old)

    # Delete last selected object from scene
    object.select_set(True)
    bpy.ops.object.delete()

    return


def process_file(input_file, input_dir, output_file, output_dir):
    global views
    global quality_preview

    if not input_file.endswith(".obj"):
        print("ERROR: Invalid input: " + input_file)
        return

    print("Processing: " + input_file)
    if output_file == "":
        output_file = input_file + ".png"

    if quality_preview:
        output_file = output_file.replace(".png", "-preview.png")

    angle = 360.0 / views

    for view in range(0, views):
        print("  View: " + str(view))
        output_file_view = output_file.replace(".png", "-%03d.png" % (view))
        yaw = view * angle
        render_file(input_file, input_dir, output_file_view, output_dir, yaw)

    #    angle = 360.0/views
    #    for view in range(0, views):
    #        print('  View: ' + str(view))
    #        yaw = view * angle
    #        output_file_view = output_file.replace('.png', '-%03d.png' % (view))
    #        render_file(input_file, input_dir, output_file_view, output_dir, yaw)

    # render_file(input_file, input_dir, output_file, output_dir, 0)

    return


def render_single(output_file):
    if not verbose:
        sys.stdout.flush()
        old = os.dup(1)
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)

    if not verbose:
        sys.stdout.flush()
        os.close(1)
        os.dup(old)
        os.close(old)

    if smooth:
        for obj in bpy.data.objects:
            obj.select_set(True)
        bpy.ops.object.shade_smooth()

    # bpy.context.scene.render.filepath = os.path.join(output_path, output_file)
    bpy.context.scene.render.filepath = output_file

    if not verbose:
        sys.stdout.flush()
        old = os.dup(1)
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)

    # Render
    bpy.ops.render.render(write_still=True)

    # set the bg to white
    img = io.imread(bpy.context.scene.render.filepath)
    mask = img[:, :, 3] / 255.0
    mask = mask[..., None]
    bg = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    img[:, :, :3] = ((mask * img[:, :, :3]) + ((1 - mask) * bg)).astype(np.uint8)
    io.imsave(bpy.context.scene.render.filepath, img)

    if not verbose:
        sys.stdout.flush()
        os.close(1)
        os.dup(old)
        os.close(old)


def blender_viewer_fn(
    hand_r_fname,
    hand_l_fname,
    obj_bottom_fname,
    obj_top_fname,
    fx,
    fy,
    cx,
    cy,
    look_at_point=(0.0, 0.0, 0.0),
    # obj_transform=None,
    img_width=1080,
    img_height=1080,
    add_ground=False,
    out_p="render_outputs/",
    wireframe=False,
    num_views=1,
    preview=False,
    cam_dexycb=False,
):
    global resolution_x
    global resolution_y
    global floor
    global output_path
    global views
    global quality_preview
    global camera_location

    quality_preview = preview

    resolution_x = img_width
    resolution_y = img_height

    floor = add_ground
    views = num_views

    scene = setup_scene()

    ####### Import object into scene #######
    bpy.ops.import_scene.obj(filepath=obj_top_fname)
    bpy.ops.import_scene.obj(filepath=obj_bottom_fname)
    bpy.ops.import_scene.obj(filepath=hand_r_fname)
    bpy.ops.import_scene.obj(filepath=hand_l_fname)

    obj_bottom_color = [0.46, 0.89, 0.14, 1.0]
    obj_top_color = [0.89, 0.46, 0.14, 1.0]

    # obj_bottom_color = [68.0/255, 158.0/255, 230.0/255, 1.0]
    obj_top_color = [0.1, 0.2, 0.8, 1.0]
    # obj_top_color = [0.46, 0.89, 0.14, 1.0]
    hand_color = [0.9, 0.8, 0.9, 1.0]  # (0.4, 0.6, 1.0, 1.0)
    # hand_color = [0.85882353, 0.74117647, 0.65098039, 1.0]  # (0.4, 0.6, 1.0, 1.0)

    setup_diffuse_transparent_material(
        bpy.data.objects[hand_l_fname.split("/")[-1].replace(".obj", "")],
        hand_color,
        object_transparent,
        mouth_transparent,
    )
    setup_diffuse_transparent_material(
        bpy.data.objects[hand_r_fname.split("/")[-1].replace(".obj", "")],
        hand_color,
        object_transparent,
        mouth_transparent,
    )

    setup_diffuse_transparent_material(
        bpy.data.objects[obj_bottom_fname.split("/")[-1].replace(".obj", "")],
        obj_bottom_color,
        object_transparent,
        mouth_transparent,
    )
    setup_diffuse_transparent_material(
        bpy.data.objects[obj_top_fname.split("/")[-1].replace(".obj", "")],
        obj_top_color,
        object_transparent,
        mouth_transparent,
    )

    ####### set camera look at #######
    if cam_dexycb:
        camera_location = (-0.5, -0.5, 0.3)

    obj_camera = bpy.data.objects["Camera"]
    obj_camera.location = (0.0, 0.0, 0.0)
    obj_camera.rotation_euler = (np.radians(270.0), 0.0, 0.0)

    # Use horizontal sensor fit
    obj_camera.data.sensor_fit = "HORIZONTAL"

    # Focal length
    obj_camera.data.lens_unit = "MILLIMETERS"
    sensor_width_in_mm = obj_camera.data.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    obj_camera.data.shift_x = -(cx / w - 0.5)
    obj_camera.data.shift_y = (cy - 0.5 * h) / w

    obj_camera.data.lens = fx / w * sensor_width_in_mm

    pixel_aspect = fy / fx
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = pixel_aspect

    ####### SETUP LIGHTS #######

    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN")
        light_sun = bpy.context.active_object
        light_sun.location = (0.0, -3, 0.0)
        light_sun.rotation_euler = (radians(-45.0), 0.0, radians(30))
    else:
        light_sun = bpy.data.objects["Sun"]

    bpy.data.lights["Sun"].energy = 3

    light = bpy.data.objects["Light"]
    bpy.data.lights["Light"].type = "SUN"
    bpy.data.lights["Light"].energy = 1
    light_offset = np.array((0.0, 3.0, 0.0))  # + look_at_point
    light.location = light_offset.tolist()
    light_rotmat = look_at(light_offset, look_at_point)
    light.rotation_euler = mathutils.Matrix(light_rotmat[0].tolist()).to_euler("XYZ")

    # Cycles lights
    # light.data.cycles.cast_shadow = True
    # light_sun.data.cycles.cast_shadow = True
    light.data.cycles.cast_shadow = False
    light_sun.data.cycles.cast_shadow = False

    logger.debug(f"Scene objects: {bpy.data.objects.keys()}")
    print(f"out_p: {out_p}")

    render_single(out_p)

    for obj_key in bpy.data.objects.keys():
        if obj_key in ["Camera", "Light", "Sun"]:
            continue
        else:
            bpy.data.objects.remove(bpy.data.objects[obj_key])


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":
    texture_dir = ""

    try:
        if bpy.app.background:
            texture_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "textures"
            )

            parser = argparse.ArgumentParser(
                description="Render image sequence from from obj directory",
                formatter_class=argparse.RawTextHelpFormatter,
                epilog="Usage: %s --input /source/dir/ --output /output/dir\n       %s --input /source/dir/cutlist.csv --output /output/dir"
                % (__file__, __file__),
            )
            parser.add_argument(
                "--input", dest="input_path", type=str, help="Input file or directory"
            )
            parser.add_argument(
                "--output",
                dest="output_path",
                type=str,
                help="Output file or directory",
            )
            parser.add_argument(
                "--object_height",
                type=float,
                default=object_height,
                help="Height position of imported object",
            )
            parser.add_argument(
                "--views", type=int, default=views, help="Number of views to render"
            )

            parser.add_argument(
                "--quality_preview",
                type=int,
                default=quality_preview,
                help="Use preview quality",
            )
            parser.add_argument(
                "--resolution_x",
                type=int,
                default=resolution_x,
                help="Render image X resolution",
            )
            parser.add_argument(
                "--resolution_y",
                type=int,
                default=resolution_y,
                help="Render image Y resolution",
            )
            parser.add_argument(
                "--shadows", type=int, default=shadows, help="Show shadows"
            )

            parser.add_argument(
                "--smooth", type=int, default=smooth, help="Smooth mesh"
            )
            parser.add_argument(
                "--wireframe",
                type=int,
                default=wireframe,
                help="Show wireframe overlay",
            )
            parser.add_argument(
                "--line_thickness",
                type=float,
                default=line_thickness,
                help="Wireframe line thickness",
            )
            parser.add_argument(
                "--quads", type=int, default=quads, help="Convert triangles to quads"
            )

            parser.add_argument(
                "--object_transparent",
                type=int,
                default=object_transparent,
                help="Render face transparent",
            )
            parser.add_argument(
                "--mouth_transparent",
                type=int,
                default=mouth_transparent,
                help="Render mouth transparent",
            )

            parser.add_argument(
                "--compositor_background_image",
                type=int,
                default=compositor_background_image,
                help="Use background image",
            )
            parser.add_argument(
                "--compositor_image_scale",
                type=float,
                default=compositor_image_scale,
                help="Input image scale",
            )
            parser.add_argument(
                "--compositor_alpha",
                type=float,
                default=compositor_alpha,
                help="Rendered object alpha value",
            )

            parser.add_argument("--fps", type=int, default=fps, help="Target framerate")
            parser.add_argument(
                "--tracking_camera",
                type=int,
                default=tracking_camera,
                help="Use tracking camera",
            )
            parser.add_argument(
                "--tracking_camera_pelvis",
                type=int,
                default=tracking_camera_pelvis,
                help="Track only pelvis vertex instead of all vertices",
            )
            parser.add_argument(
                "--floor", type=int, default=floor, help="Use floor plane"
            )

            parser.add_argument(
                "--verbose", type=int, default=verbose, help="Use verbose output"
            )

            parser.add_argument(
                "--render", type=str, default=render, help="Render engine (cycles)"
            )

            parser.add_argument(
                "--use_gpu", type=int, default=0, help="Use CUDA GPU acceleration"
            )

            args = parser.parse_args()

            # TODO make input/output positional and not optional

            if (args.input_path is None) or (args.output_path is None):
                parser.print_help()
                print("-----\n")
                sys.exit(1)

            input_path = args.input_path
            output_path = args.output_path

            if not os.path.exists(input_path):
                print("ERROR: Invalid input path")
                sys.exit(1)

            object_height = args.object_height
            views = args.views
            quality_preview = args.quality_preview
            resolution_x = args.resolution_x
            resolution_y = args.resolution_y
            shadows = args.shadows

            smooth = args.smooth
            wireframe = args.wireframe
            line_thickness = args.line_thickness

            # Always use quads in wireframe mode
            if wireframe:
                quads = True
            else:
                quads = args.quads

            object_transparent = args.object_transparent
            mouth_transparent = args.mouth_transparent

            compositor_background_image = args.compositor_background_image
            compositor_image_scale = args.compositor_image_scale
            compositor_alpha = args.compositor_alpha

            fps = args.fps
            tracking_camera = args.tracking_camera
            tracking_camera_pelvis = args.tracking_camera_pelvis
            floor = args.floor

            verbose = args.verbose

            render = args.render
            use_gpu = args.use_gpu
        # end if bpy.app.background

        startTime = time.perf_counter()

        setup_scene(texture_dir)

        # Process data
        cwd = os.getcwd()

        # Check if we use cutlist or folder processing
        use_cutlist = False
        if input_path.endswith(".csv"):
            if not os.path.isfile(input_path):
                print("ERROR: Cutlist input file not existing:" + input_path)
                sys.exit(1)
            else:
                use_cutlist = True
        else:
            if not os.path.isdir(input_path):
                print("ERROR: Invalid input directory: " + input_path)
                sys.exit(1)

            # Ensure that input_path ends with path separator so that os.path.dirname returns proper path
            if not input_path.endswith(os.path.sep):
                input_path += os.path.sep

        if not input_path.startswith(os.path.sep):
            input_dir = os.path.join(cwd, input_path)
        else:
            input_dir = os.path.dirname(input_path)

        if not output_path.endswith(os.path.sep):
            output_path += os.path.sep

        if not output_path.startswith(os.path.sep):
            output_dir = os.path.join(cwd, output_path)
        else:
            output_dir = os.path.dirname(output_path)

        if not input_dir.endswith(os.path.sep):
            input_dir += os.path.sep

        if not output_dir.endswith(os.path.sep):
            output_dir += os.path.sep

        if fps > source_fps:
            fps = source_fps

        imagerate = int(source_fps / fps)

        print("Input path: " + input_path)
        print("Input directory: " + input_dir)
        print("Output path: " + output_path)
        print("Output directory: " + output_dir)
        print("Texture directory: " + texture_dir)
        print("Target frames-per-second: " + str(fps))

        print("--------------------------------------------------")

        images_processed = 0

        if use_cutlist:
            with open(input_path) as cutlist:
                for line in cutlist:
                    line = line.rstrip()
                    if line.startswith("Name") or (line.startswith("#")) or (not line):
                        continue

                    values = line.split()
                    name = values[0]
                    start = int(values[1])
                    end = int(values[2])
                    print("Generating: %s [%d-%d]" % (name, start, end))

                    input_dir = input_path
                    output_dir = os.path.join(output_dir, name)

                    filelist = sorted(os.listdir(input_dir))
                    filelist = filelist[start : (end + 1)]
                    filelist = filelist[::imagerate]

                    for input_file in filelist:
                        if input_file.endswith(".obj"):
                            output_file = os.path.basename(output_path)
                            process_file(input_file, input_dir, "", output_dir)
                            images_processed += 1
                            if render_only_one_image:
                                break
        else:
            input_dir = input_path
            filelist = sorted(os.listdir(input_dir))
            filelist = filelist[::imagerate]

            for input_file in filelist:
                if input_file.endswith(".obj"):
                    output_file = os.path.basename(output_path)
                    process_file(input_file, input_dir, "", output_dir)
                    images_processed += 1
                    if render_only_one_image:
                        break

        print("--------------------------------------------------")
        print("Rendering finished.")
        print("Processing time : %0.2f s" % (time.perf_counter() - startTime))
        print("Images processed: " + str(images_processed))
        print(
            "Time per image  : %0.2f s"
            % ((time.perf_counter() - startTime) / images_processed)
        )
        print("--------------------------------------------------")

        from utils.humor_utils import create_video

        if quality_preview:
            create_video(
                f"{output_path}/%06d.obj-preview.png",
                out_path=input_path.replace("/", "") + ".mp4",
                fps=25,
            )
        else:
            create_video(
                f"{output_path}/%06d.obj.png",
                out_path=input_path.replace("/", "") + ".mp4",
                fps=25,
            )
        sys.exit(0)

    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else:
            exit_status = ex.code

        print("Exiting. Exit status: " + str(exit_status))

        # Only exit to OS when we are not running in Blender GUI
        if bpy.app.background:
            sys.exit(exit_status)
