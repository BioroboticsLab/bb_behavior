import numpy as np

def relative_rotation(a, z_rot_a, b):
    """
    relative_rotation calculates the relative rotation of one tag to the position of another tag.
    It takes two vecors representing positions of tags and the z_rotation of the first tag.
    It returns the scalar product between the normalized vector pointing from a to b and a normalized vector
    representing the first tag's rotation.
    Assumption: z_rot_a is given in radians with value 0 pointing east, 1/2*pi south, -1/2*pi north and 1*pi west.
    """
    rotation = np.array((np.cos(z_rot_a),np.sin(z_rot_a)))
    normalized_rotation = rotation/np.linalg.norm(rotation)
    normalized_a_to_b = (b-a)/np.linalg.norm(b-a)
    return np.dot(normalized_a_to_b, normalized_rotation)

def is_valid_relative_rotation(xy1, r1, xy2, r2, minimum_relative_orientation):
    """Checks whether the relative_rotation of two positions is above a certain threshold.

    Arguments:
        xy1: np.array
            x, y coordinate of the first individual.
        r1: float
            Orientation on the plane of the first individual.
        xy2: np.array
            x, y coordinate of the second individual.
        r2: float
            Orientation on the plane of the second individual.
    
    Returns:
        bool
        Whether the relative_rotation if above the set threshold.
    """
    rot1 = relative_rotation(xy1, r1, xy2)
    rot2 = relative_rotation(xy2, r2, xy1)
    return (rot1 >= minimum_relative_orientation) and (rot2 >= minimum_relative_orientation)