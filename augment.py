import Augmentor

p=Augmentor.Pipeline('D:\\base400\\train\\TA')
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.zoom(probability=0.2,min_factor=5,max_factor=1.2)
p.skew(probability=0.2)
p.shear(probability=0.2,max_shear_left=0.95189873417721522,max_shear_right=2)
p.crop_random(probability=0.5,percentage_area=0.8)
p.flip_random(probability=0.2)
p.sample(420)
p.random_distortion(probability=1,grid_width=4,grid_height=4,magnitude=8)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)

p.process()