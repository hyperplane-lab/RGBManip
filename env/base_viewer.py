from abc import abstractclassmethod

class BaseViewer:
    def __init__(self, name, get_rgba_method, get_depth_method, get_pos_method, get_segmentation_method, update_param_method, get_param_method, get_norm_method):
        self.name = name
        self.get_rgba_method = get_rgba_method
        self.get_depth_method = get_depth_method
        self.get_pos_method = get_pos_method
        self.get_segmentation_method = get_segmentation_method
        self.update_param_method = update_param_method
        self.get_param_method = get_param_method
        self.get_norm_method = get_norm_method

    def get_rgba(self, *args, **kwargs) :
        '''
        Take picture from the viewer.
        return: RGBA image array of W*H*4.
        '''
        return self.get_rgba_method(*args, **kwargs)

    def get_depth(self, *args, **kwargs) :
        '''
        Take picture from the viewer.
        return: depth array of W*H.
        '''
        return self.get_depth_method(*args, **kwargs)
    
    def get_pos(self, *args, **kwargs) :
        '''
        Take picture from the viewer.
        return: depth array of W*H*3.
        '''
        return self.get_pos_method(*args, **kwargs)
    
    def get_segmentation(self, *args, **kwargs) :
        '''
        Take picture from the viewer.
        return: segmentation array of W*H*2,
            the two demensions are for mesh-level and actor-level segmentation.
        '''
        return self.get_segmentation_method(*args, **kwargs)

    def get_norm(self) :
        '''
        Take picture from the viewer.
        return: norm array of W*H*3.
        '''

        return self.get_norm_method()

    def update_viewer(self, camera_extrinsics) :
        '''
        Update camera extrinsics and intrisics.
        camera_extrinsics: transformation matrix.
        '''
        return self.update_param_method(camera_extrinsics)

    def get_param(self) :
        '''
        Get camera intrinsics and extrinsics.
        return: camera intrinsics and extrinsics.
        '''
        return self.get_param_method()