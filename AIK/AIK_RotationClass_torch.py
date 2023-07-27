import torch


class CustomRotation(object):
    def __init__(self, quat, normalize=True, copy=True):
        self._single = False
        quat = quat.clone().to(torch.float32) #torch.tensor(quat, dtype=float, requires_grad=True)
        if quat.ndim not in [1, 2] or quat.shape[-1] != 4:
            raise ValueError("Expected `quat` to have shape (4,) or (N x 4), "
                             "got {}.".format(quat.shape))

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if quat.shape == (4,):
            quat = quat[None, :]
            self._single = True

        if normalize:
            self._quat = quat.copy()
            norms = torch.norm(quat, axis=1)

            zero_norms = norms == 0
            if zero_norms.any():
                raise ValueError("Found zero norm quaternions in `quat`.")

            # Ensure norm is broadcasted along each column.
            self._quat[~zero_norms] /= norms[~zero_norms][:, None]
        else:
            self._quat = quat.copy() if copy else quat

    def __len__(self):
        return self._quat.shape[0]
    
    @classmethod
    def from_matrix(cls, matrix):
        is_single = False
        matrix = matrix.clone().to(matrix.device) #torch.tensor(matrix, dtype=torch.float64, requires_grad=True)

        if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
            raise ValueError("Expected `matrix` to have shape (3, 3) or "
                             "(N, 3, 3), got {}".format(matrix.shape))

        # If a single matrix is given, convert it to 3D 1 x 3 x 3 matrix but
        # set self._single to True so that we can return appropriate objects in
        # the `to_...` methods
        if matrix.shape == (3, 3):
            matrix = matrix.reshape((1, 3, 3))
            is_single = True

        num_rotations = matrix.shape[0]

        decision_matrix = torch.empty((num_rotations, 4)).to(matrix.device) #torch.empty((num_rotations, 4),dtype=torch.float64, requires_grad=True)
        decision_matrix[:, :3] = torch.diagonal(matrix, dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
        choices = decision_matrix.argmax(axis=1)

        quat = torch.empty((num_rotations, 4)).to(matrix.device)

        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
        quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
        quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
        quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
        quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
        quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]
        quat = quat / (torch.norm(quat, dim=1)[:, None])

        if is_single:
            return cls(quat[0], normalize=False, copy=False)
        else:
            return cls(quat, normalize=False, copy=False)


    def as_rotvec(self):
        quat = self._quat.clone()
        # w > 0 to ensure 0 <= angle <= pi
        quat[quat[:, 3] < 0] *= -1

        angle = 2 * torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])

        small_angle = (angle <= 1e-3)
        large_angle = ~small_angle

        num_rotations = len(self)
        scale = torch.empty(num_rotations).to(torch.float32).to(quat.device)
        scale[small_angle] = (2 + angle[small_angle] ** 2 / 12 +
                              7 * angle[small_angle] ** 4 / 2880)
        scale[large_angle] = (angle[large_angle] /
                              torch.sin(angle[large_angle] / 2))

        rotvec = scale[:, None] * quat[:, :3]

        if self._single:
            return rotvec[0]
        else:
            return rotvec