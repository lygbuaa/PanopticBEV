
import numpy as np

g_grid_sample_table = {
#GridSample_148
"756": {"elem_type": 1, "shape": [1, 128, 192, 192]},
"854": {"elem_type": 1, "shape": [1, 192, 225, 2]},
"855": {"elem_type": 1, "shape": [1, 128, 192, 225]},
#GridSample_224
"867": {"elem_type": 1, "shape": [1, 128, 112, 192]},
"998": {"elem_type": 1, "shape": [1, 192, 224, 2]},
"999": {"elem_type": 1, "shape": [1, 128, 192, 224]},
#GridSample_357
"1030": {"elem_type": 1, "shape": [1, 1, 192, 224]},
"1177": {"elem_type": 1, "shape": [1, 112, 192, 2]},
"1178": {"elem_type": 1, "shape": [1, 1, 112, 192]},
#GridSample_462
"1191": {"elem_type": 1, "shape": [1, 128, 224, 192]},
"1295": {"elem_type": 1, "shape": [1, 192, 225, 2]},
"1296": {"elem_type": 1, "shape": [1, 128, 192, 225]},
#GridSample_941
"1426": {"elem_type": 1, "shape": [1, 256, 224, 384]},
"1831": {"elem_type": 1, "shape": [1, 224, 384, 2]},
"1832": {"elem_type": 1, "shape": [1, 256, 224, 384]},

#GridSample_1091
"1898": {"elem_type": 1, "shape": [1, 128, 96, 96]},
"1996": {"elem_type": 1, "shape": [1, 96, 113, 2]},
"1997": {"elem_type": 1, "shape": [1, 128, 96, 113]},
#GridSample_1167
"2009": {"elem_type": 1, "shape": [1, 128, 56, 96]},
"2140": {"elem_type": 1, "shape": [1, 96, 112, 2]},
"2141": {"elem_type": 1, "shape": [1, 128, 96, 112]},
#GridSample_1300
"2172": {"elem_type": 1, "shape": [1, 1, 96, 112]},
"2319": {"elem_type": 1, "shape": [1, 56, 96, 2]},
"2320": {"elem_type": 1, "shape": [1, 1, 56, 96]},
#GridSample_1405
"2333": {"elem_type": 1, "shape": [1, 128, 112, 96]},
"2437": {"elem_type": 1, "shape": [1, 96, 113, 2]},
"2438": {"elem_type": 1, "shape": [1, 128, 96, 113]},
#GridSample_1884
"2568": {"elem_type": 1, "shape": [1, 256, 112, 192]},
"2973": {"elem_type": 1, "shape": [1, 112, 192, 2]},
"2974": {"elem_type": 1, "shape": [1, 256, 112, 192]},

#GridSample_2034
"3040": {"elem_type": 1, "shape": [1, 128, 48, 48]},
"3138": {"elem_type": 1, "shape": [1, 48, 57, 2]},
"3139": {"elem_type": 1, "shape": [1, 128, 48, 57]},
#GridSample_2109
"3151": {"elem_type": 1, "shape": [1, 128, 28, 48]},
"3292": {"elem_type": 1, "shape": [1, 48, 56, 2]},
"3293": {"elem_type": 1, "shape": [1, 128, 48, 56]},
#GridSample_2242
"3324": {"elem_type": 1, "shape": [1, 1, 48, 56]},
"3471": {"elem_type": 1, "shape": [1, 28, 48, 2]},
"3472": {"elem_type": 1, "shape": [1, 1, 28, 48]},
#GridSample_2347
"3485": {"elem_type": 1, "shape": [1, 128, 56, 48]},
"3589": {"elem_type": 1, "shape": [1, 48, 57, 2]},
"3590": {"elem_type": 1, "shape": [1, 128, 48, 57]},
#GridSample_2826
"3720": {"elem_type": 1, "shape": [1, 256, 56, 96]},
"4125": {"elem_type": 1, "shape": [1, 56, 96, 2]},
"4126": {"elem_type": 1, "shape": [1, 256, 56, 96]},

#GridSample_2976
"4192": {"elem_type": 1, "shape": [1, 128, 24, 24]},
"4290": {"elem_type": 1, "shape": [1, 24, 29, 2]},
"4291": {"elem_type": 1, "shape": [1, 128, 24, 29]},
#GridSample_3052
"4303": {"elem_type": 1, "shape": [1, 128, 14, 24]},
"4434": {"elem_type": 1, "shape": [1, 24, 28, 2]},
"4435": {"elem_type": 1, "shape": [1, 128, 24, 28]},
#GridSample_3185
"4466": {"elem_type": 1, "shape": [1, 1, 24, 28]},
"4613": {"elem_type": 1, "shape": [1, 14, 24, 2]},
"4614": {"elem_type": 1, "shape": [1, 1, 14, 24]},
#GridSample_3290
"4627": {"elem_type": 1, "shape": [1, 128, 28, 24]},
"4731": {"elem_type": 1, "shape": [1, 24, 29, 2]},
"4732": {"elem_type": 1, "shape": [1, 128, 24, 29]},
#GridSample_3769
"4862": {"elem_type": 1, "shape": [1, 256, 28, 48]},
"5267": {"elem_type": 1, "shape": [1, 28, 48, 2]},
"5268": {"elem_type": 1, "shape": [1, 256, 28, 48]},
}