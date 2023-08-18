import pybullet as p

p.connect(p.DIRECT)
for i in range(0, 9):
    name_in = f'../resources/objects/blocks/obj/{i}.obj'
    name_out = name_in.replace('.obj', '_vhacd.obj')
    name_log = "../logs/gen_vhacd/log.txt"
    p.vhacd(name_in, name_out, name_log)
