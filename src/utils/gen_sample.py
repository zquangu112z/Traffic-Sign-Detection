import Augmentor

# General transform
# datadir = ['00000', '00014', '00017', '00032',
#            '00033', '00034', '00051', '00052', '00099']

datadir = ['00060']

for d in datadir:
    p = Augmentor.Pipeline(
        "data/raw/training/" + d,
        "../augmented/" + d)
    # p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.skew_left_right(probability=0.8, magnitude=0.2)
    p.shear(probability=0.8, max_shear_left=15, max_shear_right=15)
    p.sample(500)


# Mirror left-right transfrom
p = Augmentor.Pipeline(
    "data/raw/training/00060",
    "../augmented/00060")
p.flip_left_right(probability=1)
p.sample(500)

p = Augmentor.Pipeline(
    "data/raw/training/00060",
    "../augmented/00060")
p.flip_top_bottom(probability=1)
p.sample(500)

# p = Augmentor.Pipeline(
#     "data/raw/training/00054",
#     "../augmented/00053")
# p.flip_left_right(probability=1)
# p.sample(5000)
