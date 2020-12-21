import pickle
import numpy as np
from DataContainer import DataContainer
from TFL_manager import TFL_manager


class Controller:
    def __init__(self, pls):
        # read pls

        with open(pls, "r") as plsfile:
            data = [line[:-1] for line in plsfile]

        self.pkl = data[0]
        self.frames = data[1:]


    def init_prev_container(self, first_frame, focal, pp, tfl_man , tfl):
        first_img_path = first_frame
        first = DataContainer(first_img_path ,tfl)
        candidates, auxiliary = tfl_man.light_attention(first)
        tfl_man.tfl_detection(first,candidates,auxiliary)
        return first

    def read_pkl_file(self):
        pkl_path = 'dusseldorf_000049.pkl'
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        focal = data['flx']
        pp = data['principle_point']
        return data, focal, pp

    def read_EM(self,prev_frame_id,curr_frame_id,data):
        return np.array(data['egomotion_' + str(prev_frame_id) + '-' + str(curr_frame_id)])

    def run(self):
        data, focal, pp = self.read_pkl_file()
        tfl_man = TFL_manager(pp , focal)

        prev_container = self.init_prev_container(self.frames[0], focal, pp, tfl_man,np.array(data['points_' + str(self.frames[0][31:33])][0]))

        for i in range(1, len(self.frames)):
            curr_frame = self.frames[i]
            prev_frame = self.frames[i - 1]
            curr_img_path = curr_frame

            EM = self.read_EM(prev_frame[31:33],curr_frame[31:33],data)

            curr_container = DataContainer(curr_img_path,np.array(data['points_' + str(curr_frame[31:33])][0]), EM)

            candidates, auxiliary = tfl_man.light_attention(curr_container)
            # print(candidates,"\n", auxiliary)
            tfl_man.tfl_detection(curr_container , candidates,auxiliary)
            curr_container = tfl_man.calc_distance(curr_container, prev_container)
            print(prev_container ,"\n", curr_container)
            # visualize(prev_container, curr_container, focal, pp)

            prev_container = curr_container
