import torch.nn as nn
import smplx
import torch
import os
import numpy as np

MODEL_DIR_R = os.environ["MANO_MODEL_DIR_R"]
MODEL_DIR_L = os.environ["MANO_MODEL_DIR_L"]

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)


class MANOLayer(nn.Module):
    def __init__(self, use_pca, is_rhand, num_pca_comp, flat_hand_mean):
        super().__init__()
        if is_rhand:
            model_path = MODEL_DIR_R
        else:
            model_path = MODEL_DIR_L
        self.layer = smplx.create(
            model_path=model_path,
            model_type="mano",
            use_pca=use_pca,
            is_rhand=is_rhand,
            flat_hand_mean=flat_hand_mean,
        )

        seal_faces = torch.LongTensor(np.array(SEAL_FACES_R))
        if not is_rhand:
            # left hand
            seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal

        self.faces = torch.LongTensor(self.layer.faces.astype(np.int32))
        self.sealed_faces = torch.cat((self.faces, seal_faces))

    def forward(self, global_orient, hand_pose, betas, transl):
        out = self.layer(
            global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl
        )
        joints = out.joints
        vertices = out.vertices

        out_dict = {}
        out_dict["joints"] = joints
        out_dict["vertices"] = vertices

        centers = vertices[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
        sealed_vertices = torch.cat((vertices, centers), dim=1)
        out_dict["vertices.sealed"] = sealed_vertices
        return out_dict

    def to(self, dev):
        self.layer.to(dev)
