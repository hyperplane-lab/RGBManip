import numpy as np

def normalize(x) :
    '''
    normalize vectors
    '''

    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

def quat_mul(q1, q2) :
    '''
    multiply two quaternions
    '''
    q1 = np.array(q1)
    q2 = np.array(q2)
    shape = q1.shape
    assert(shape[-1] == 4)
    assert(q2.shape == shape)

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]

    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

    # w1, x1, y1, z1 = q1
    # w2, x2, y2, z2 = q2

    # w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    # x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    # z = w1*z2 + z1*w2 + x1*y2 - y1*x2

    # return np.array([w, x, y, z])

def lookat_quat(direction) :

    shape = direction.shape
    assert(shape[-1] == 3)
    direction = direction.reshape(-1, 3)

    direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-9)
    # upaxis = upaxis / np.linalg.norm(upaxis)

    x_ = np.asarray([1., 0, 0])
    y_ = np.asarray([0, 1., 0])
    z_ = np.asarray([0, 0, 1.])

    res = np.zeros((direction.shape[0], 4))

    for id, dir in enumerate(direction) :

        dot = (z_ * dir).sum(axis=-1)

        if abs(np.linalg.norm(direction)) < 1e-6 :

            x = x_
            y = y_
            z = z_

        elif abs(dot - (-1.0)) < 1e-6 :

            x = -z_
            y = y_
            z = x_
        
        elif abs(dot - (1.0)) < 1e-6 :

            x = z_
            y = y_
            z = -x_
        
        else :

            y = np.cross(z_, dir)
            y = y / np.linalg.norm(y, axis=-1, keepdims=True)
            z = np.cross(dir, y)
            z = z / np.linalg.norm(z, axis=-1, keepdims=True)
            x = dir

        quat = get_quaternion([x_, y_, z_], [x, y, z])

        res[id] = quat

    return res.reshape(*shape[:-1], 4)

    # dot = (np.array([[0, 0, 1]]) * direction).sum(axis=-1)

    # res = np.zeros((direction.shape[0], 4))

    # res[np.where(np.abs(dot - (-1.0)) < 0.000001)] = np.array([0, 1, 0, np.pi])
    # res[np.where(np.abs(dot - (1.0)) < 0.000001)] = np.array([1, 0, 0, 0])

    # rot_angle = np.arccos(dot)
    # rot_axis = np.cross(np.array([[0, 0, 1]]), direction)
    # rot_axis = rot_axis / np.linalg.norm(rot_axis, axis=-1, keepdims=True)

    # return axis_angle_to_quat(rot_axis, rot_angle).reshape(*shape[:-1], 4)

# public static Quaternion LookAt(Vector3 sourcePoint, Vector3 destPoint)
# {
#     Vector3 forwardVector = Vector3.Normalize(destPoint - sourcePoint);

#     float dot = Vector3.Dot(Vector3.forward, forwardVector);

#     if (Math.Abs(dot - (-1.0f)) < 0.000001f)
#     {
#         return new Quaternion(Vector3.up.x, Vector3.up.y, Vector3.up.z, 3.1415926535897932f);
#     }
#     if (Math.Abs(dot - (1.0f)) < 0.000001f)
#     {
#         return Quaternion.identity;
#     }

#     float rotAngle = (float)Math.Acos(dot);
#     Vector3 rotAxis = Vector3.Cross(Vector3.forward, forwardVector);
#     rotAxis = Vector3.Normalize(rotAxis);
#     return CreateFromAxisAngle(rotAxis, rotAngle);
# }

def axis_angle_to_quat(axis, angle):
    '''
    axis: [[x, y, z]] or [x, y, z]
    angle: rad
    return: a quat that rotates angle around axis
    '''
    axis = np.array(axis)
    shape = axis.shape
    assert(shape[-1] == 3)
    axis = axis.reshape(-1, 3)

    angle = np.array(angle)
    angle = angle.reshape(-1, 1)

    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9)
    quat1 = np.concatenate([np.cos(angle/2), axis[:, 0:1]*np.sin(angle/2), axis[:, 1:2]*np.sin(angle/2), axis[:, 2:3]*np.sin(angle/2)], axis=-1)
    return quat1.reshape(*shape[:-1], 4)

def batch_get_quaternion(lst1, lst2) :
    '''
    lst1: batch fo list of 3d vectors
    lst2: list of 3d vectors
    returns: quaternion that rotates lst1 to lst2
    '''

    ret = []

    for tmp1, tmp2 in zip(lst1, lst2) :

        ret.append(get_quaternion(tmp1, tmp2))
    
    return ret

def get_quaternion(lst1, lst2, matchlist=None):
    '''
    lst1: list of 3d vectors
    lst2: list of 3d vectors
    matchlist: list of indices of lst2 that correspond to lst1
    returns: quaternion that rotates lst1 to lst2
    '''
    if not matchlist:
        matchlist=range(len(lst1))
    M=np.matrix([[0,0,0],[0,0,0],[0,0,0]])

    for i,coord1 in enumerate(lst1):
        x=np.matrix(np.outer(coord1,lst2[matchlist[i]]))
        M=M+x

    N11=float(M[0][:,0]+M[1][:,1]+M[2][:,2])
    N22=float(M[0][:,0]-M[1][:,1]-M[2][:,2])
    N33=float(-M[0][:,0]+M[1][:,1]-M[2][:,2])
    N44=float(-M[0][:,0]-M[1][:,1]+M[2][:,2])
    N12=float(M[1][:,2]-M[2][:,1])
    N13=float(M[2][:,0]-M[0][:,2])
    N14=float(M[0][:,1]-M[1][:,0])
    N21=float(N12)
    N23=float(M[0][:,1]+M[1][:,0])
    N24=float(M[2][:,0]+M[0][:,2])
    N31=float(N13)
    N32=float(N23)
    N34=float(M[1][:,2]+M[2][:,1])
    N41=float(N14)
    N42=float(N24)
    N43=float(N34)

    N=np.matrix([[N11,N12,N13,N14],\
              [N21,N22,N23,N24],\
              [N31,N32,N33,N34],\
              [N41,N42,N43,N44]])


    values,vectors=np.linalg.eig(N)
    w=list(values)
    mw=max(w)
    quat= vectors[:,w.index(mw)]
    quat=np.array(quat).reshape(-1,)
    return quat

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((-a[:, :3], a[:, -1:]), axis=-1).reshape(shape)

def quat_to_axis(q, id) :
    '''
    q: a quat
    id: 0, 1, 2
    '''
    
    q = np.array(q)
    shape = q.shape
    q = q.reshape(-1, 4)
    
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    
    if id == 0 :
        return np.concatenate([2*q0**2+2*q1**2-1, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2], axis=-1).reshape(shape[:-1] + (3,))
    elif id == 1 :
        return np.concatenate([2*q1*q2-2*q0*q3, 2*q0**2+2*q2**2-1, 2*q2*q3+2*q0*q1], axis=-1).reshape(shape[:-1] + (3,))
    elif id == 2 :
        return np.concatenate([2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*q0**2+2*q3**2-1], axis=-1).reshape(shape[:-1] + (3,))

def compute_quat_err(targ, curr) :

    cc = quat_conjugate(curr)
    q_r = quat_mul(targ, cc)
    return q_r[0:3] * np.sign(q_r[3])