import os
import glob
from PIL import Image, ImageOps


if __name__ == '__main__':
    path_in = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/synthetic-stamps-20180525/'
    path_out = '/Users/dmitryduev/_caltech/python/deep-asteroids/data-raw/synthetic-streaks'

    nn = 1
    for subf in ('tmp-png-1', 'tmp-png-2'):
        f_stamps = glob.glob(os.path.join(path_in, subf, '*.png'))
        for f_stamp in f_stamps:
            print(nn)
            im = Image.open(f_stamp)
            # region = im.crop((14, 14, 382, 382)).resize((144, 144), Image.BILINEAR).convert('RGB')
            region = im.crop((14, 14, 382, 382)).resize((157, 157)).convert('RGB')
            # gray = ImageOps.grayscale(region)
            # region.show()
            # gray.show()
            # input()
            region.save(os.path.join(path_out, f'{nn:06d}.jpg'), "jpeg", quality=100)
            nn += 1
