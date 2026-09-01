"""
Model 6 — 3D Body Reconstruction: Delaunay-triangulation-based face texture warping.

Replaces the simple center-paste `_build_head_texture()` with a proper
piecewise-affine warp: MediaPipe face-mesh landmarks detected in the user's
selfie are triangulated (via Delaunay on the UV-space triangulation), and
each triangle is warped from selfie-space to UV-space via `cv2.getAffineTransform`
+ `cv2.warpAffine`. The result is a continuous, distortion-minimised face
texture that maps every pixel of the user's face to the correct UV coordinate
on the 3D head mesh — the same technique used by Meta, Snapchat and ZEPETO.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark → UV coordinate mapping
# ---------------------------------------------------------------------------
# These UV coordinates map the canonical MediaPipe face-mesh topology onto
# the UV-sphere head produced by `mesh_builder._build_head()`.
#
# The head UV sphere:
#   u=0.0 / u=1.0 = seam at the back (-Z)
#   u=0.5          = front of the face (+Z)
#   v=0.0          = bottom (neck)
#   v=1.0          = top (crown) — glTF/OpenGL convention
#
# Values were derived from the 3D facial-feature positions defined in
# mesh_builder.py (eyes, nose, mouth, ears) plus the canonical MediaPipe
# topology.  For intermediate landmarks not listed here, UVs are
# interpolated from their Delaunay-triangulated neighbours at runtime.

# fmt: off
# UV coordinates computed from the UV-sphere head mesh geometry in
# ``mesh_builder._build_head()`` (n_lat=8, n_lon=16, phi_back=1.5π).
# Each landmark index maps to the 3D position on the unit-sphere head
# (derived from mesh_builder.py's facial-feature constants) projected
# through that function's spherical UV layout.
_FACE_UV_ANCHORS: dict[int, tuple[float, float]] = {
    0: (0.5140, 0.3990),
    1: (0.5000, 0.4827),
    2: (0.5000, 0.4932),
    3: (0.4174, 0.3291),
    4: (0.5000, 0.5103),
    5: (0.5000, 0.5341),
    6: (0.5000, 0.5845),
    7: (0.4338, 0.5237),
    8: (0.3632, 0.4917),
    9: (0.3750, 0.5326),
    10: (0.3951, 0.5718),
    11: (0.4174, 0.6064),
    12: (0.4416, 0.6358),
    13: (0.5144, 0.3800),
    14: (0.5584, 0.6358),
    15: (0.5826, 0.6064),
    16: (0.6049, 0.5718),
    17: (0.5441, 0.5945),
    18: (0.5567, 0.5879),
    19: (0.4760, 0.4829),
    20: (0.5800, 0.5753),
    21: (0.5936, 0.5695),
    22: (0.4559, 0.5945),
    23: (0.4433, 0.5879),
    24: (0.4324, 0.5811),
    25: (0.4200, 0.5753),
    26: (0.4064, 0.5695),
    33: (0.4300, 0.5169),
    37: (0.5070, 0.3925),
    39: (0.4930, 0.3925),
    40: (0.4860, 0.3990),
    48: (0.4442, 0.4353),
    49: (0.4545, 0.4466),
    50: (0.4220, 0.4444),
    52: (0.5571, 0.3992),
    53: (0.5455, 0.4466),
    54: (0.5558, 0.4353),
    55: (0.5722, 0.4210),
    56: (0.5873, 0.3980),
    57: (0.5860, 0.3711),
    59: (0.4429, 0.3992),
    60: (0.4278, 0.4210),
    61: (0.4644, 0.3915),
    63: (0.4127, 0.3980),
    65: (0.6191, 0.3809),
    66: (0.4140, 0.3711),
    67: (0.6049, 0.4018),
    69: (0.5914, 0.4229),
    70: (0.5780, 0.4444),
    71: (0.5860, 0.3879),
    72: (0.5980, 0.3924),
    73: (0.6106, 0.4126),
    74: (0.6235, 0.4388),
    75: (0.6280, 0.4788),
    76: (0.6250, 0.5216),
    77: (0.6174, 0.5634),
    78: (0.4754, 0.3894),
    80: (0.5000, 0.3904),
    81: (0.5140, 0.3927),
    82: (0.5246, 0.3894),
    84: (0.3750, 0.5216),
    85: (0.3720, 0.4788),
    86: (0.3765, 0.4388),
    87: (0.3894, 0.4126),
    88: (0.4020, 0.3924),
    94: (0.4830, 0.4796),
    101: (0.4086, 0.4229),
    102: (0.3951, 0.4018),
    103: (0.3809, 0.3809),
    104: (0.3750, 0.4185),
    105: (0.3705, 0.4582),
    133: (0.4300, 0.5169),
    144: (0.4445, 0.5305),
    145: (0.4477, 0.5239),
    153: (0.4445, 0.5204),
    154: (0.4391, 0.5136),
    155: (0.4338, 0.5102),
    157: (0.4277, 0.5170),
    158: (0.4293, 0.5205),
    159: (0.4331, 0.5239),
    160: (0.4369, 0.5272),
    161: (0.4423, 0.5307),
    163: (0.4391, 0.5305),
    164: (0.5170, 0.4796),
    168: (0.5000, 0.6087),
    173: (0.4293, 0.5137),
    185: (0.4754, 0.3987),
    191: (0.4860, 0.3927),
    195: (0.5000, 0.5609),
    243: (0.4331, 0.5103),
    244: (0.4385, 0.5137),
    245: (0.4423, 0.5171),
    246: (0.4477, 0.5307),
    263: (0.5700, 0.5169),
    267: (0.5246, 0.3987),
    269: (0.5356, 0.3915),
    270: (0.5289, 0.3801),
    271: (0.5184, 0.3714),
    272: (0.5075, 0.3663),
    273: (0.5000, 0.3662),
    274: (0.4925, 0.3663),
    275: (0.4816, 0.3714),
    276: (0.4711, 0.3801),
    278: (0.4754, 0.4583),
    279: (0.4714, 0.4471),
    280: (0.4668, 0.4350),
    281: (0.4619, 0.4259),
    282: (0.5246, 0.4583),
    283: (0.5286, 0.4471),
    284: (0.5332, 0.4350),
    285: (0.5381, 0.4259),
    311: (0.4856, 0.3800),
    312: (0.5000, 0.3764),
    362: (0.5700, 0.5169),
    384: (0.5609, 0.5305),
    385: (0.5555, 0.5305),
    386: (0.5523, 0.5239),
    387: (0.5555, 0.5204),
    388: (0.5609, 0.5136),
    389: (0.5662, 0.5102),
    390: (0.5577, 0.5307),
    391: (0.5631, 0.5272),
    392: (0.5669, 0.5239),
    393: (0.5707, 0.5205),
    394: (0.5723, 0.5170),
    395: (0.5707, 0.5137),
    396: (0.5669, 0.5103),
    397: (0.5615, 0.5137),
    398: (0.5577, 0.5171),
    466: (0.5523, 0.5307),
    468: (0.4398, 0.5168),
    473: (0.5602, 0.5168),
}
# fmt: on

# ---------------------------------------------------------------------------
# MakeHuman-specific UV anchors
# ---------------------------------------------------------------------------
# The MakeHuman head uses a spherical UV projection (see
# ``bake_makehuman_morphs.py``'s ``_remap_uv_for_face_overlay``):
#   u = 0.5 + atan2(x, z) / (2π)   — front = u=0.5
#   v = 0.5 - asin(y) / π           — v=0 = top (crown), v=1 = bottom (neck)
#
# These values were obtained by converting the procedural ``_FACE_UV_ANCHORS``
# through the two UV-mapping functions (procedural → 3D direction → MakeHuman).
# See ``scripts/compute_mh_uv_anchors.py``.
# fmt: off
_FACE_UV_ANCHORS_MAKEHUMAN: dict[int, tuple[float, float]] = {
    0: (0.4860, 0.6010), 1: (0.5000, 0.5173), 2: (0.5000, 0.5068),
    3: (0.5826, 0.6709), 4: (0.5000, 0.4897), 5: (0.5000, 0.4659),
    6: (0.5000, 0.4155), 7: (0.5662, 0.4763), 8: (0.6368, 0.5083),
    9: (0.6250, 0.4674), 10: (0.6049, 0.4282), 11: (0.5826, 0.3936),
    12: (0.5584, 0.3642), 13: (0.4856, 0.6200), 14: (0.4416, 0.3642),
    15: (0.4174, 0.3936), 16: (0.3951, 0.4282), 17: (0.4559, 0.4055),
    18: (0.4433, 0.4121), 19: (0.5240, 0.5171), 20: (0.4200, 0.4247),
    21: (0.4064, 0.4305), 22: (0.5441, 0.4055), 23: (0.5567, 0.4121),
    24: (0.5676, 0.4189), 25: (0.5800, 0.4247), 26: (0.5936, 0.4305),
    33: (0.5700, 0.4831), 37: (0.4930, 0.6075), 39: (0.5070, 0.6075),
    40: (0.5140, 0.6010), 48: (0.5558, 0.5647), 49: (0.5455, 0.5534),
    50: (0.5780, 0.5556), 52: (0.4429, 0.6008), 53: (0.4545, 0.5534),
    54: (0.4442, 0.5647), 55: (0.4278, 0.5790), 56: (0.4127, 0.6020),
    57: (0.4140, 0.6289), 59: (0.5571, 0.6008), 60: (0.5722, 0.5790),
    61: (0.5356, 0.6085), 63: (0.5873, 0.6020), 65: (0.3809, 0.6191),
    66: (0.5860, 0.6289), 67: (0.3951, 0.5982), 69: (0.4086, 0.5771),
    70: (0.4220, 0.5556), 71: (0.4140, 0.6121), 72: (0.4020, 0.6076),
    73: (0.3894, 0.5874), 74: (0.3765, 0.5612), 75: (0.3720, 0.5212),
    76: (0.3750, 0.4784), 77: (0.3826, 0.4366), 78: (0.5246, 0.6106),
    80: (0.5000, 0.6096), 81: (0.4860, 0.6073), 82: (0.4754, 0.6106),
    84: (0.6250, 0.4784), 85: (0.6280, 0.5212), 86: (0.6235, 0.5612),
    87: (0.6106, 0.5874), 88: (0.5980, 0.6076), 94: (0.5170, 0.5204),
    101: (0.5914, 0.5771), 102: (0.6049, 0.5982), 103: (0.6191, 0.6191),
    104: (0.6250, 0.5815), 105: (0.6295, 0.5418), 133: (0.5700, 0.4831),
    144: (0.5555, 0.4695), 145: (0.5523, 0.4761), 153: (0.5555, 0.4796),
    154: (0.5609, 0.4864), 155: (0.5662, 0.4898), 157: (0.5723, 0.4830),
    158: (0.5707, 0.4795), 159: (0.5669, 0.4761), 160: (0.5631, 0.4728),
    161: (0.5577, 0.4693), 163: (0.5609, 0.4695), 164: (0.4830, 0.5204),
    168: (0.5000, 0.3913), 173: (0.5707, 0.4863), 185: (0.5246, 0.6013),
    191: (0.5140, 0.6073), 195: (0.5000, 0.4391), 243: (0.5669, 0.4897),
    244: (0.5615, 0.4863), 245: (0.5577, 0.4829), 246: (0.5523, 0.4693),
    263: (0.4300, 0.4831), 267: (0.4754, 0.6013), 269: (0.4644, 0.6085),
    270: (0.4711, 0.6199), 271: (0.4816, 0.6286), 272: (0.4925, 0.6337),
    273: (0.5000, 0.6338), 274: (0.5075, 0.6337), 275: (0.5184, 0.6286),
    276: (0.5289, 0.6199), 278: (0.5246, 0.5417), 279: (0.5286, 0.5529),
    280: (0.5332, 0.5650), 281: (0.5381, 0.5741), 282: (0.4754, 0.5417),
    283: (0.4714, 0.5529), 284: (0.4668, 0.5650), 285: (0.4619, 0.5741),
    311: (0.5144, 0.6200), 312: (0.5000, 0.6236), 362: (0.4300, 0.4831),
    384: (0.4391, 0.4695), 385: (0.4445, 0.4695), 386: (0.4477, 0.4761),
    387: (0.4445, 0.4796), 388: (0.4391, 0.4864), 389: (0.4338, 0.4898),
    390: (0.4423, 0.4693), 391: (0.4369, 0.4728), 392: (0.4331, 0.4761),
    393: (0.4293, 0.4795), 394: (0.4277, 0.4830), 395: (0.4293, 0.4863),
    396: (0.4331, 0.4897), 397: (0.4385, 0.4863), 398: (0.4423, 0.4829),
    466: (0.4477, 0.4693), 468: (0.5602, 0.4832), 473: (0.4398, 0.4832),
}
# fmt: on

# The full set of MediaPipe FaceMesh indices that should be warped.
# MediaPipe FaceLandmarker returns either 468 (standard) or 478 (with 10
# iris landmarks at indices 468-477).  We cover both by using all available
# indices.
_FACE_LANDMARK_INDICES = list(range(478))

# ---------------------------------------------------------------------------
# MediaPipe FACEMESH_TESSELATION — graph edges connecting all 468 landmarks.
# Used below to propagate UVs from known anchors to un-mapped landmarks via
# graph diffusion, instead of the old nearest-index-neighbour fallback that
# produced visible seams.
# (Standard MediaPipe face mesh topology.)
# ---------------------------------------------------------------------------
# fmt: off
_FACEMESH_TESSELATION = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
    (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),
    (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44),
    (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51),
    (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58),
    (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),
    (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71), (71, 72),
    (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79),
    (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86),
    (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93),
    (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100),
    (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106),
    (106, 107), (107, 108), (108, 109), (109, 110), (110, 111), (111, 112),
    (112, 113), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118),
    (118, 119), (119, 120), (120, 121), (121, 122), (122, 123), (123, 124),
    (124, 125), (125, 126), (126, 127), (127, 128), (128, 129), (129, 130),
    (130, 131), (131, 132), (132, 133), (133, 134), (134, 135), (135, 136),
    (136, 137), (137, 138), (138, 139), (139, 140), (140, 141), (141, 142),
    (142, 143), (143, 144), (144, 145), (145, 146), (146, 147), (147, 148),
    (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154),
    (154, 155), (155, 156), (156, 157), (157, 158), (158, 159), (159, 160),
    (160, 161), (161, 162), (162, 163), (163, 164), (164, 165), (165, 166),
    (166, 167), (167, 168), (168, 169), (169, 170), (170, 171), (171, 172),
    (172, 173), (173, 174), (174, 175), (175, 176), (176, 177), (177, 178),
    (178, 179), (179, 180), (180, 181), (181, 182), (182, 183), (183, 184),
    (184, 185), (185, 186), (186, 187), (187, 188), (188, 189), (189, 190),
    (190, 191), (191, 192), (192, 193), (193, 194), (194, 195), (195, 196),
    (196, 197), (197, 198), (198, 199), (199, 200), (200, 201), (201, 202),
    (202, 203), (203, 204), (204, 205), (205, 206), (206, 207), (207, 208),
    (208, 209), (209, 210), (210, 211), (211, 212), (212, 213), (213, 214),
    (214, 215), (215, 216), (216, 217), (217, 218), (218, 219), (219, 220),
    (220, 221), (221, 222), (222, 223), (223, 224), (224, 225), (225, 226),
    (226, 227), (227, 228), (228, 229), (229, 230), (230, 231), (231, 232),
    (232, 233), (233, 234), (234, 235), (235, 236), (236, 237), (237, 238),
    (238, 239), (239, 240), (240, 241), (241, 242), (242, 243), (243, 244),
    (244, 245), (245, 246), (246, 247), (247, 248), (248, 249), (249, 250),
    (250, 251), (251, 252), (252, 253), (253, 254), (254, 255), (255, 256),
    (256, 257), (257, 258), (258, 259), (259, 260), (260, 261), (261, 262),
    (262, 263), (263, 264), (264, 265), (265, 266), (266, 267), (267, 268),
    (268, 269), (269, 270), (270, 271), (271, 272), (272, 273), (273, 274),
    (274, 275), (275, 276), (276, 277), (277, 278), (278, 279), (279, 280),
    (280, 281), (281, 282), (282, 283), (283, 284), (284, 285), (285, 286),
    (286, 287), (287, 288), (288, 289), (289, 290), (290, 291), (291, 292),
    (292, 293), (293, 294), (294, 295), (295, 296), (296, 297), (297, 298),
    (298, 299), (299, 300), (300, 301), (301, 302), (302, 303), (303, 304),
    (304, 305), (305, 306), (306, 307), (307, 308), (308, 309), (309, 310),
    (310, 311), (311, 312), (312, 313), (313, 314), (314, 315), (315, 316),
    (316, 317), (317, 318), (318, 319), (319, 320), (320, 321), (321, 322),
    (322, 323), (323, 324), (324, 325), (325, 326), (326, 327), (327, 328),
    (328, 329), (329, 330), (330, 331), (331, 332), (332, 333), (333, 334),
    (334, 335), (335, 336), (336, 337), (337, 338), (338, 339), (339, 340),
    (340, 341), (341, 342), (342, 343), (343, 344), (344, 345), (345, 346),
    (346, 347), (347, 348), (348, 349), (349, 350), (350, 351), (351, 352),
    (352, 353), (353, 354), (354, 355), (355, 356), (356, 357), (357, 358),
    (358, 359), (359, 360), (360, 361), (361, 362), (362, 363), (363, 364),
    (364, 365), (365, 366), (366, 367), (367, 368), (368, 369), (369, 370),
    (370, 371), (371, 372), (372, 373), (373, 374), (374, 375), (375, 376),
    (376, 377), (377, 378), (378, 379), (379, 380), (380, 381), (381, 382),
    (382, 383), (383, 384), (384, 385), (385, 386), (386, 387), (387, 388),
    (388, 389), (389, 390), (390, 391), (391, 392), (392, 393), (393, 394),
    (394, 395), (395, 396), (396, 397), (397, 398), (398, 399), (399, 400),
    (400, 401), (401, 402), (402, 403), (403, 404), (404, 405), (405, 406),
    (406, 407), (407, 408), (408, 409), (409, 410), (410, 411), (411, 412),
    (412, 413), (413, 414), (414, 415), (415, 416), (416, 417), (417, 418),
    (418, 419), (419, 420), (420, 421), (421, 422), (422, 423), (423, 424),
    (424, 425), (425, 426), (426, 427), (427, 428), (428, 429), (429, 430),
    (430, 431), (431, 432), (432, 433), (433, 434), (434, 435), (435, 436),
    (436, 437), (437, 438), (438, 439), (439, 440), (440, 441), (441, 442),
    (442, 443), (443, 444), (444, 445), (445, 446), (446, 447), (447, 448),
    (448, 449), (449, 450), (450, 451), (451, 452), (452, 453), (453, 454),
    (454, 455), (455, 456), (456, 457), (457, 458), (458, 459), (459, 460),
    (460, 461), (461, 462), (462, 463), (463, 464), (464, 465), (465, 466),
    (466, 467),
])
# fmt: on


# ---------------------------------------------------------------------------
# UV target builder
# ---------------------------------------------------------------------------

def _propagate_uvs_via_mesh(known_uvs: dict[int, tuple[float, float]],
                             num_landmarks: int = 478) -> dict[int, tuple[float, float]]:
    """Propagate UVs from known anchors to ALL landmarks using graph diffusion
    on the MediaPipe face mesh topology (``_FACEMESH_TESSELATION``).

    This replaces the old nearest-index-neighbor fallback which produced
    visible seams because index-distance doesn't correspond to Euclidean
    distance on the face surface.  Graph diffusion preserves the mesh's
    local geometry: connected landmarks that are close in 3D space get
    smoothly interpolated UVs.

    Any landmarks still NaN after diffusion (e.g. iris landmarks 468-477
    that are not part of the tessellation) fall back to nearest known
    anchor by index.
    """
    # Build adjacency from tessellation
    adj: dict[int, set[int]] = {i: set() for i in range(num_landmarks)}
    for a, b in _FACEMESH_TESSELATION:
        if a < num_landmarks and b < num_landmarks:
            adj[a].add(b)
            adj[b].add(a)

    import numpy as np
    uvs: dict[int, np.ndarray] = {}
    for i in range(num_landmarks):
        if i in known_uvs:
            u, v = known_uvs[i]
            uvs[i] = np.array([u, v], dtype=np.float32)
        else:
            uvs[i] = np.array([np.nan, np.nan], dtype=np.float32)

    # Iterative diffusion (max 50 rounds — converges in ~10-15 in practice)
    for _ in range(50):
        changed = False
        new_uvs = {i: uvs[i].copy() for i in range(num_landmarks)}
        for i in range(num_landmarks):
            if np.any(np.isnan(uvs[i])):
                neighbor_uvs = [uvs[nb] for nb in adj[i]
                                if nb < num_landmarks and not np.any(np.isnan(uvs[nb]))]
                if neighbor_uvs:
                    new_uvs[i] = np.mean(neighbor_uvs, axis=0)
                    changed = True
        uvs = new_uvs
        if not changed:
            break

    # Final fallback for any remaining NaN (iris landmarks not in tessellation)
    known_sorted = sorted(known_uvs.keys())
    for i in range(num_landmarks):
        if np.any(np.isnan(uvs[i])):
            nearest = min(known_sorted, key=lambda k: abs(k - i))
            uvs[i] = uvs[nearest].copy()

    return {i: (float(uvs[i][0]), float(uvs[i][1])) for i in range(num_landmarks)}


def _build_uv_targets(img_size: int = 512, num_landmarks: int = 478,
                      uv_anchors: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    """Build a full set of UV-space target points for all MediaPipe face-mesh
    landmarks (indices 0 to ``num_landmarks``-1).  Covers both the standard
    468 and the newer 478 (with 10 iris landmarks at 468-477).

    Uses graph diffusion on ``_FACEMESH_TESSELATION`` to propagate UVs from
    known anchors to all landmarks, producing smooth, seam-free interpolation.

    Parameters
    ----------
    img_size : int
        Output texture size (square).
    num_landmarks : int
        Number of MediaPipe landmarks (468 or 478).
    uv_anchors : dict or None
        UV anchor dict to use.  If None, uses the procedural head's anchors
        (``_FACE_UV_ANCHORS``, calibrated for ``mesh_builder._build_head``).
        Pass ``_FACE_UV_ANCHORS_MAKEHUMAN`` for the MakeHuman head UV layout.

    Returns (num_landmarks, 2) float32 array of (u, v) coordinates in pixels
    (i.e. u * img_size, v * img_size).
    """
    if uv_anchors is None:
        uv_anchors = _FACE_UV_ANCHORS

    # Propagate UVs from known anchors via mesh graph diffusion
    all_uvs = _propagate_uvs_via_mesh(uv_anchors, num_landmarks=num_landmarks)

    targets = np.zeros((num_landmarks, 2), dtype=np.float32)
    for i in range(num_landmarks):
        u, v = all_uvs[i]
        targets[i] = (u * img_size, v * img_size)

    return targets


# ---------------------------------------------------------------------------
# Core warping function
# ---------------------------------------------------------------------------

def warp_face_to_uv(
    selfie_rgb: np.ndarray,
    landmarks_2d: np.ndarray | list,
    img_size: int = 512,
    uv_anchors: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Warp a selfie face into UV texture space using Delaunay triangulation
    + piecewise affine transforms.

    Parameters
    ----------
    selfie_rgb:
        (H, W, 3) uint8 RGB image of the user's face (full photo or crop).
    landmarks_2d:
        (N, 2) float32 array of MediaPipe face-mesh landmark pixel positions
        in ``selfie_rgb`` space.  N >= 468 (468 standard or 478 with iris).
    img_size:
        Output texture size (square).  Default 512.
    uv_anchors:
        UV anchor dict to use.  If None, uses the procedural head's anchors
        (``_FACE_UV_ANCHORS``).  Pass ``_FACE_UV_ANCHORS_MAKEHUMAN`` for the
        MakeHuman head UV layout.

    Returns
    -------
    warped:
        (img_size, img_size, 3) uint8 RGB texture with the face warped
        onto the UV layout of the head mesh.
    """
    h, w = selfie_rgb.shape[:2]

    if isinstance(landmarks_2d, list):
        landmarks_2d = np.asarray(landmarks_2d, dtype=np.float32)
    landmarks_2d = landmarks_2d.reshape(-1, 2)

    # Ensure we have at least the face contour landmarks
    # Filter to valid indices that are within the image bounds
    valid_mask = (
        (landmarks_2d[:, 0] >= 0) & (landmarks_2d[:, 0] < w) &
        (landmarks_2d[:, 1] >= 0) & (landmarks_2d[:, 1] < h)
    )
    valid_indices = [i for i in _FACE_LANDMARK_INDICES if i < len(landmarks_2d) and valid_mask[i]]

    if len(valid_indices) < 10:
        # Fallback: not enough landmarks — return a skin-colour canvas
        return np.full((img_size, img_size, 3), 128, dtype=np.uint8)

    # Source points (selfie space)
    src_pts = np.array([landmarks_2d[i] for i in valid_indices], dtype=np.float32)

    # Build target UV points for the actual number of landmarks
    uv_targets = _build_uv_targets(img_size, num_landmarks=len(landmarks_2d),
                                   uv_anchors=uv_anchors)
    dst_pts = np.array([uv_targets[i] for i in valid_indices], dtype=np.float32)

    # Delaunay triangulation on the *source* (selfie) points, not the UV
    # targets. Triangulating in UV space picks triangle connectivity based on
    # proximity after graph-diffusion, which doesn't respect facial topology:
    # two points can land close together in UV space (small dst triangle)
    # while being far apart on the actual face (huge src triangle) -- e.g. an
    # ear landmark and an eye landmark diffused to neighbouring UV texels.
    # Warping such a triangle stretches a large selfie region into a tiny UV
    # patch, producing visible shattered/starburst artefacts. Triangulating
    # in source space instead guarantees triangle connectivity follows the
    # real, non-self-intersecting 2D face layout MediaPipe detected -- the
    # standard approach for Delaunay-based face texture warping.
    tri = Delaunay(src_pts)
    triangles = tri.simplices  # (T, 3) indices into dst_pts / src_pts

    # Warp each triangle
    warped_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Accumulator for weighted blending at overlapping/boundary texels
    accum = np.zeros((img_size, img_size, 3), dtype=np.float32)
    weight = np.zeros((img_size, img_size), dtype=np.float32)

    for tri_idx in triangles:
        src_tri = src_pts[tri_idx].astype(np.float32)
        dst_tri = dst_pts[tri_idx].astype(np.float32)

        # Bounding box of the destination triangle (clipped to image)
        x_min = max(0, int(np.floor(dst_tri[:, 0].min())))
        y_min = max(0, int(np.floor(dst_tri[:, 1].min())))
        x_max = min(img_size - 1, int(np.ceil(dst_tri[:, 0].max())))
        y_max = min(img_size - 1, int(np.ceil(dst_tri[:, 1].max())))

        if x_max <= x_min or y_max <= y_min:
            continue

        # Warp this triangle's region
        tri_w, tri_h = x_max - x_min + 1, y_max - y_min + 1
        if tri_w < 1 or tri_h < 1:
            continue

        # Destination triangle in the LOCAL (cropped tri_w x tri_h) buffer's
        # coordinate space -- warpAffine's output pixel indices below are
        # local to that buffer (0..tri_w, 0..tri_h), not absolute canvas
        # coordinates, so the transform must be fit against local coords too.
        tri_dst_local = dst_tri.copy()
        tri_dst_local[:, 0] -= x_min
        tri_dst_local[:, 1] -= y_min

        # Forward transform src (selfie) -> local dst. cv2.warpAffine's
        # default (no WARP_INVERSE_MAP) inverts this itself to sample: for
        # each local dst pixel (x,y), src_rect(x,y) = selfie(M^-1(x,y)) --
        # exactly the src pixel that triangle vertex correspondence implies.
        affine_mat = cv2.getAffineTransform(src_tri, tri_dst_local)

        # Create a mask for this triangle in the destination
        mask = np.zeros((tri_h, tri_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_dst_local.astype(np.int32), 255)

        # Warp the source pixels into the destination rect
        src_rect = cv2.warpAffine(
            selfie_rgb, affine_mat, (tri_w, tri_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Accumulate with mask weighting for smooth blending at seams
        mask_f = mask.astype(np.float32) / 255.0
        for c in range(3):
            accum[y_min:y_max + 1, x_min:x_max + 1, c] += src_rect[..., c] * mask_f
        weight[y_min:y_max + 1, x_min:x_max + 1] += mask_f

    # Normalise by weight
    weight = np.clip(weight, 1e-6, None)
    for c in range(3):
        warped_img[..., c] = np.clip(accum[..., c] / weight, 0, 255).astype(np.uint8)

    return warped_img


# ---------------------------------------------------------------------------
# Blending: composite the warped face onto a skin-tone canvas
# ---------------------------------------------------------------------------

def blend_face_with_skin(
    warped_face: np.ndarray,
    skin_rgb: tuple[int, int, int],
    blend_mode: str = "poisson",
) -> np.ndarray:
    """Blend the warped face texture onto a skin-tone background canvas.

    Parameters
    ----------
    warped_face:
        (S, S, 3) uint8 — output of ``warp_face_to_uv``.
    skin_rgb:
        (R, G, B) 0-255 skin colour for the canvas background.
    blend_mode:
        ``"poisson"`` (default) uses OpenCV's seamless clone for
        photorealistic blending.  ``"feather"`` falls back to a soft
        elliptical mask + Gaussian blur (no extra dependency).

    Returns
    -------
    blended:
        (S, S, 3) uint8 final texture ready to embed in the GLB.
    """
    S = warped_face.shape[0]
    canvas = np.full((S, S, 3), list(skin_rgb), dtype=np.uint8)

    if blend_mode == "poisson":
        try:
            # Create a mask covering the valid (non-zero) region of the warp
            gray = cv2.cvtColor(warped_face, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            # Zero top 18% to exclude any stray hair pixels warped near the hairline
            mask[:int(S * 0.18), :] = 0
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            if np.any(mask > 0):
                centre = (S // 2, S // 2)
                blended = cv2.seamlessClone(warped_face, canvas, mask, centre, cv2.NORMAL_CLONE)
                return blended
        except Exception:
            pass  # fall through to feather blend

    # --- Asymmetric elliptical mask: top 20% clipped to exclude hair above forehead ---
    mask_img = np.zeros((S, S), dtype=np.uint8)
    top_margin = int(S * 0.20)
    side_margin = int(S * 0.08)
    bot_margin = int(S * 0.08)
    center_y = (top_margin + (S - bot_margin)) // 2
    semi_h = (S - bot_margin - top_margin) // 2
    semi_w = S // 2 - side_margin
    cv2.ellipse(mask_img,
                (S // 2, center_y),
                (semi_w, semi_h),
                0, 0, 360, 255, -1)

    # `warped_face` is zero (black) everywhere the triangulated warp didn't
    # actually paint -- it only covers the small region the detected face
    # landmarks span, not the whole ellipse above. Replace those unpainted
    # pixels with skin tone *before* any blurring/blending touches them: a
    # mask built from `warped_face` directly (even intersected with a
    # content mask) still gets Gaussian-blurred, and blurring a mask next to
    # true-black pixels bleeds a soft dark halo into the feathered boundary.
    # Pre-filling means the blur only ever mixes "real face pixel" with
    # "skin tone" -- never black.
    gray = cv2.cvtColor(warped_face, cv2.COLOR_RGB2GRAY)
    _, content_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    content_mask_3c = cv2.cvtColor(content_mask, cv2.COLOR_GRAY2RGB) > 0
    warped_filled = np.where(content_mask_3c, warped_face, canvas)

    mask_img = cv2.bitwise_and(mask_img, content_mask)
    mask_img = cv2.GaussianBlur(mask_img, (0, 0), S * 0.06)

    mask_f = mask_img.astype(np.float32) / 255.0
    blended = (warped_filled.astype(np.float32) * mask_f[:, :, None] +
               canvas.astype(np.float32) * (1.0 - mask_f[:, :, None]))
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_head_texture_warped(
    selfie_rgb: np.ndarray | None,
    landmarks_2d: np.ndarray | list | None,
    skin_rgb: tuple[int, int, int],
    texture_size: int = 256,
    blend_mode: str = "feather",
    uv_anchors: dict[int, tuple[float, float]] | None = None,
) -> bytes:
    """High-level entry point: warps a selfie face onto UV layout, blends it
    with skin tone, and returns PNG bytes ready for GLB embedding.

    Parameters
    ----------
    selfie_rgb:
        (H, W, 3) uint8 RGB selfie, or None to produce a flat skin-colour
        texture.
    landmarks_2d:
        (N, 2) MediaPipe face-mesh landmarks, or None.
    skin_rgb:
        (R, G, B) 0-255.
    texture_size:
        Output texture size (square).  Default 256.
    blend_mode:
        ``"poisson"`` or ``"feather"`` (see ``blend_face_with_skin``).

    uv_anchors:
        Optional UV anchor dict.  If None, uses the procedural head's anchors
        (``_FACE_UV_ANCHORS``).  Pass ``_FACE_UV_ANCHORS_MAKEHUMAN`` for the
        MakeHuman head UV layout.

    Returns
    -------
    png_bytes:
        PNG-encoded texture bytes, ready for ``glb_export``.
    """
    # Require at least 468 MediaPipe landmarks for a proper Delaunay warp.
    # MediaPipe can return 468 (standard) or 478 (with iris landmarks);
    # both work.  Haar-cascade fallback (~108 approximate points) does not
    # map to the MediaPipe topology indices that ``_FACE_UV_ANCHORS``
    # expects, and would produce a garbled result.  Return a flat
    # skin-colour texture instead — fall back to centre-paste.
    if selfie_rgb is None or landmarks_2d is None or len(landmarks_2d) < 468:
        flat = np.full((texture_size, texture_size, 3), list(skin_rgb), dtype=np.uint8)
        return _to_png(flat)

    # Warp at higher resolution for quality, then downscale
    warp_size = max(texture_size, 512)
    warped = warp_face_to_uv(selfie_rgb, landmarks_2d, img_size=warp_size,
                             uv_anchors=uv_anchors)
    if warp_size != texture_size:
        warped = cv2.resize(warped, (texture_size, texture_size), interpolation=cv2.INTER_AREA)

    blended = blend_face_with_skin(warped, skin_rgb, blend_mode=blend_mode)

    # 🌟 VERTICAL_OFFSET — when using procedural UV anchors (uv_anchors=None),
    # the warped face needs a small downward shift to match the MakeHuman head
    # UV layout (which positions the face region slightly lower in UV space
    # than the procedural UV sphere).  When using MakeHuman-specific anchors
    # (uv_anchors=_FACE_UV_ANCHORS_MAKEHUMAN), the UVs already target the
    # correct positions — no shift needed.
    if uv_anchors is None:
        _VERTICAL_OFFSET = 0.12
        if _VERTICAL_OFFSET != 0:
            shift_px = int(texture_size * _VERTICAL_OFFSET)
            canvas = np.full_like(blended, list(skin_rgb))
            if shift_px > 0:
                canvas[shift_px:, :, :] = blended[:-shift_px, :, :]
            elif shift_px < 0:
                canvas[:shift_px, :, :] = blended[-shift_px:, :, :]
            blended = canvas

    return _to_png(blended)


def _to_png(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
