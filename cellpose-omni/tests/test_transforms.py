from cellpose_omni.transforms import *
# so the test was is actually broken in that it passes in an image without a channel, 
# so the imwarp was acting on slices of the image not the full image. 


def test_random_rotate_and_resize__default():
    nimg = 2
    X = [np.random.rand(64, 64) for i in range(nimg)]

    random_rotate_and_resize(X)

def test_random_rotate_and_resize__use_skel():
    nimg = 2
    X = [np.random.rand(64, 64)[np.newaxis] for i in range(nimg)]

    random_rotate_and_resize(X, omni=True)
