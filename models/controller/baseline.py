import os
from models.controller.base_controller import BaseController
from abc import abstractclassmethod
from models.manipulation.open_cabinet import OpenCabinetManipulation
from models.manipulation.open_drawer import OpenDrawerManipulation
from models.manipulation.open_pot import OpenPotManipulation
from models.manipulation.pick_mug import PickMugManipulation
import numpy as np

class BaselineController(BaseController) :

    def run(self, setting, action) :

        self.env.load(setting)
        # print(setting, action)
        # exit()
        center = action[None, :3]
        direction = action[None, 3:]
        x_ = np.zeros_like(direction)
        x_[:, 0] = 1
        y_ = np.zeros_like(direction)
        y_[:, 1] = 1
        z_ = np.zeros_like(direction)
        z_[:, 2] = 1

        axis = np.zeros((1, 3, 3))
        
        if isinstance(self.manipulation, OpenCabinetManipulation) :
            axis[0, 0] = -action[3:]
        elif isinstance(self.manipulation, OpenDrawerManipulation) :
            axis[0, 0] = -action[3:]
        elif isinstance(self.manipulation, OpenPotManipulation) :
            axis[0, 1, 1] = 1
            axis[0, 2, 0] = 1
        elif isinstance(self.manipulation, PickMugManipulation) :
            axis[0, 1] = action[3:]
        
        self.manipulation.plan_pathway(center, axis)
        pass