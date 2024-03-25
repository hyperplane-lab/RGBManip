import numpy as np
import sapien.core as sapien
from gym import spaces
import numpy as np
from copy import deepcopy
import torch
import cv2

def show_image(img) :

    mat = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if len(img.shape) == 2:
        mat[:, :, 0] = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        mat = img
    else :
        raise ValueError("Invalid image shape")
    
    cv2.imshow("tmp", mat)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def split_obs(data, num_envs) :

    res = None
    if isinstance(data, np.ndarray) :
        res = []
        assert(data.shape[0] == num_envs)
        for i in range(num_envs) :
            res.append(data[i])
    elif isinstance(data, dict) :
        res = [{} for i in range(num_envs)]
        for k, v in data.items() :
            tmp = split_obs(v, num_envs)
            for i, item in enumerate(tmp) :
                res[i][k] = item
    elif isinstance(data, str) :
        res = [data for i in range(num_envs)]
    else :
        raise TypeError(data)
    
    return res

def merge_obs(lst) :

    res = None
    for d in lst :
        if isinstance(d, tuple) :
            if type(res) == type(None) :
                res = []
            for i in range(len(d)) :
                res.append(merge_obs([a[i] for a in lst]))
            return res
        if isinstance(d, dict) :
            if type(res) == type(None) :
                res = {}
            for k, v in d.items() :
               res[k] = merge_obs([a[k] for a in lst])
            return res
        elif isinstance(d, sapien.Pose) :
            if type(res) == type(None) :
                res = np.concatenate([d.p, d.q])[np.newaxis, ...]
            else :
                res = np.concatenate([res, np.concatenate([d.p, d.q])[np.newaxis, ...]])
        elif isinstance(d, np.ndarray) :
            if type(res) == type(None) :
                res = d[np.newaxis, ...]
            else :
                res = np.concatenate([res, d[np.newaxis, ...]])
        elif isinstance(d, bool) or isinstance(d, int) or isinstance(d, float) :
            if type(res) == type(None) :
                res = np.asarray([d], dtype=np.int32)
            else :
                res = np.concatenate([res, np.asarray([d], dtype=np.int32)])
        else :
            raise TypeError(d)

        
            # for k, v in d.items() :

            #     if isinstance(v, sapien.Pose) :
            #         v = np.concatenate([v.p, v.q])[np.newaxis, ...]
            #         if k not in res :
            #             res[k] = v
            #         else :
            #             res[k] = np.concatenate([res[k], v])
            #     elif isinstance(v, np.ndarray) :
            #         v = v[np.newaxis, ...]
            #         if k not in res :
            #             res[k] = v
            #         else :
            #             res[k] = np.concatenate([res[k], v])
            #     elif isinstance(v, dict) :
            #         res[k] = merge_obs([a[k] for a in lst])
            #     else :
            #         raise TypeError(v)

    return res

def numpy_dict_to_list_dict(d) :
    res = {}
    for k, v in d.items() :
        if isinstance(v, dict) :
            res[k] = numpy_dict_to_list_dict(v)
        elif isinstance(v, np.ndarray) :
            res[k] = v.tolist()
        elif isinstance(v, str) :
            res[k] = v
        else :
            raise TypeError(res)
    return res    

def numpy_dict_to_tensor_dict(d) :
    res = {}
    for k, v in d.items() :
        if isinstance(v, sapien.Pose) :
            res[k] = torch.tensor(np.concatenate([v.p, v.q]))
        elif isinstance(v, dict) :
            res[k] = numpy_dict_to_tensor_dict(v)
        elif isinstance(v, np.ndarray) :
            res[k] = torch.from_numpy(v)
        else :
            raise TypeError(res)
    return res

def flatten_dict(d) :
    res = np.array([])
    for k, v in d.items() :
        if isinstance(v, sapien.Pose) :
            res = np.concatenate([res, v.p])
            res = np.concatenate([res, v.q])
        elif isinstance(v, dict) :
            res = np.concatenate([res, flatten_dict(v)])
        else :
            res = np.concatenate([res, np.asarray(v).flatten()])
    return res

def create_np_buffer(space: spaces.Space, n: int):
    if isinstance(space, spaces.Dict):
        return {
            key: create_np_buffer(subspace, n) for key, subspace in space.spaces.items()
        }
    elif isinstance(space, spaces.Box):
        return np.zeros((n,) + space.shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )

def concat_spaces(space: spaces.Space) :
    '''
    Concat a Dict space into a Box space
    '''
    if isinstance(space, spaces.Dict) :
        res = []
        for k, v in space.spaces.items() :
            res.append(concat_spaces(v).shape[0])
        return spaces.Box(low=-np.inf, high=np.inf, shape=(sum(res),), dtype=np.float32)
    elif isinstance(space, spaces.Box) :
        return space
    else :
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )

def concat_tensor_dict(d) :

    res = []
    for k, v in d.items() :
        if isinstance(v, dict) :
            res.append(concat_tensor_dict(v))
        elif isinstance(v, torch.Tensor) :
            res.append(v)
        elif isinstance(v, np.ndarray) :
            res.append(torch.from_numpy(v))
        else :
            raise TypeError(v)
    return torch.cat(res, dim=-1).float()

def regularize_dict(d) :

    res = {}
    for k, v in d.items() :
        if isinstance(v, sapien.Pose) :
            tmp = pose_to_array(v)
            res[k] = tmp
        elif isinstance(v, list) :
            res[k] = np.asarray(v)
        elif isinstance(v, dict) :
            res[k] = regularize_dict(v)
        elif isinstance(v, np.ndarray) :
            res[k] = v
        else :
            raise TypeError(v)
    return res

def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)

def pose_to_array(pq) :

    assert(isinstance(pq, sapien.Pose))
    return np.concatenate([pq.p, pq.q])

def convert_observation_to_space(observation, prefix=""):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        space = spaces.Dict(
            {
                k: convert_observation_to_space(v, prefix + "/" + k)
                for k, v in observation.items()
            }
        )
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

if __name__ == "__main__" :

    a = {
        'a' : np.asarray([1, 2, 3]),
        'b' : np.asarray([4, 5, 6]),
        'c' : {
            'd' : np.asarray([7, 8, 9])
        }
    }
    res = split_obs(a)
    print(res)