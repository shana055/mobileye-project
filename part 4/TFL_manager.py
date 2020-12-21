from PIL import Image

import SFM
from run_attention import get_lights_attention
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

class TFL_manager:

    def __init__(self , pp , focal):
        self.pp = pp
        self.focal = focal

    def light_attention(self, data_frame):
        data_frame.candidates, data_frame.auxiliary = get_lights_attention(data_frame.img)
        return data_frame.candidates, data_frame.auxiliary

    def crop(self,img , candidates):
        crops = []
        np.pad(img, pad_width=((0, 0), (41, 41), (41, 41)), mode='constant', constant_values=0)
        for i in candidates:
            crops.append(img[i[0] - 40: i[0] + 41 ,i[1] - 40:i[1] + 41])

        matching = [i for i in crops if i.shape == (81,81,3)]

            # print(img[i[0] - 40: i[0] + 41 ,i[1] - 40:i[1] + 41])
            # img_ = Image.fromarray(img, 'RGB')
            # crops.append(img_.crop(((i[1] - 40 , i[0] - 40), (i[1] + 41 ,i[0] + 41))))

            # plt.imshow(crops[0])
        return matching

    def tfl_detection(self, curr_container,candidates,auxiliary):
        loaded_model = load_model("CNN_data/model.h5")
        img = plt.imread(curr_container.img)
        croped_imgs = self.crop(img,candidates)
        # plt.imshow(croped_imgs[0])

        l_predictions = loaded_model.predict(np.array(croped_imgs))
        # croped_imgs_ = np.vstack(croped_imgs)
        # l_predictions = loaded_model.predict_classes(croped_imgs_, batch_size=10)
        # data_frame.tfls, data_frame.auxiliary = run_part_2(data_frame.img, data_frame.candidates, data_frame.auxiliary)

    def calc_distance(self, curr_data_frame, prev_data_frame):
        return SFM.calc_TFL_dist(prev_data_frame, curr_data_frame, self.focal,self.pp)

