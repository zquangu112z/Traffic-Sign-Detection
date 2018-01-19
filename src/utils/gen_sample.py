import Augmentor
# p = Augmentor.Pipeline("datasets/traffic-sign/Argument")
p = Augmentor.Pipeline("data/raw/training/00052")
# p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.skew_left_right(probability=0.8, magnitude=0.2)
p.shear(probability=0.8, max_shear_left=15, max_shear_right=15)
p.sample(5000)

# datadir = ['00000/']

# for d in datadir:
    
