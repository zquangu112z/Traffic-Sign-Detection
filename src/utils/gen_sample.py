import Augmentor

# General transform
datadir = ['00000', '00014', '00017',
           '00033', '00034']

for d in datadir:
    p = Augmentor.Pipeline(
        "data/raw/training/" + d,
        "../augmented/" + d)
    # p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.skew_left_right(probability=0.8, magnitude=0.2)
    p.shear(probability=0.8, max_shear_left=15, max_shear_right=15)
    p.sample(20000)


# Mirror left-right transfrom
p = Augmentor.Pipeline(
    "data/raw/training/00033",
    "../augmented/00034")
p.flip_left_right(probability=1)
p.sample(10000)

# Mirror left-right transfrom
p = Augmentor.Pipeline(
    "data/raw/training/00034",
    "../augmented/00033")
p.flip_left_right(probability=1)
p.sample(10000)

p = Augmentor.Pipeline(
    "data/raw/training/00017",
    "../augmented/00017")
p.flip_left_right(probability=1)
p.sample(5000)

p = Augmentor.Pipeline(
    "data/raw/training/00017",
    "../augmented/00017")
p.flip_top_bottom(probability=1)
p.sample(5000)