from .imports import *


# formerly named network
def run_network(self, x, return_conv=False, to_numpy=True):
    """convert imgs to torch/mxnet and run network model and return numpy"""
    X = self._to_device(x)
    if self.torch:
        self.net.eval()
        with torch.no_grad():
            y, style = self.net(X)
    else:
        y, style = self.net(X)
    del X
    if to_numpy:
        y = self._from_device(y)
        style = self._from_device(style)

    if return_conv:  # pragma: no cover
        print("cc")
        conv = self._from_device(conv)
        y = np.concatenate((y, conv), axis=1)

    return y, style



def _run_tiled(self, imgi, augment=False, normalize=True,
               bsize=224, tile_overlap=0.1,
               return_conv=False, to_numpy=True):
    """run network in tiles of size [bsize x bsize]

    Image is split into overlapping tiles of size [bsize x bsize].
    If augment, tiles have 50% overlap and are flipped at overlaps.
    The average of the network output over tiles is returned.

    Parameters
    --------------

    imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    Returns
    ------------------

    yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
        yf is averaged over tiles
        yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

    styles: array [64]
        1D array summarizing the style of the image, averaged over tiles

    """

    if imgi.ndim == 4 and self.dim == 2:
        batch_size = self.batch_size
        Lz, nchan = imgi.shape[:2]
        IMG, subs, shape, inds = transforms.make_tiles_ND(
            imgi[0],
            bsize=bsize,
            augment=augment,
            tile_overlap=tile_overlap,
            normalize=normalize,
        )
        ntiles, nchan = IMG.shape[:2]
        ly, lx = IMG.shape[-2:]
        batch_size *= max(4, (bsize**2 // (ly * lx))**0.5)
        yf = np.zeros((Lz, self.nclasses, imgi.shape[-2], imgi.shape[-1]), np.float32)
        styles = []
        if ntiles > batch_size:
            ziterator = trange(Lz, file=tqdm_out)
            for i in ziterator:
                yfi, stylei = self._run_tiled(
                    imgi[i],
                    augment=augment,
                    bsize=bsize,
                    normalize=normalize,
                    tile_overlap=tile_overlap,
                    return_conv=return_conv,
                    to_numpy=to_numpy,
                )
                yf[i] = yfi
                styles.append(stylei)
        else:
            nimgs = max(2, int(np.round(batch_size / ntiles)))
            niter = int(np.ceil(Lz / nimgs))
            ziterator = trange(niter, file=tqdm_out)
            for k in ziterator:
                IMGa = np.zeros((ntiles * nimgs, nchan, ly, lx), np.float32)
                for i in range(min(Lz - k * nimgs, nimgs)):
                    IMG, subs, shape, inds = transforms.make_tiles_ND(
                        imgi[k * nimgs + i],
                        bsize=bsize,
                        augment=augment,
                        tile_overlap=tile_overlap,
                        normalize=normalize,
                    )
                    IMGa[i * ntiles:(i + 1) * ntiles] = IMG
                ya, stylea = self.run_network(IMGa,
                    return_conv=return_conv,
                    to_numpy=to_numpy,
                )
                for i in range(min(Lz - k * nimgs, nimgs)):
                    y = ya[i * ntiles:(i + 1) * ntiles]
                    if augment:
                        y = transforms.unaugment_tiles_ND(y, inds, self.unet)
                    yfi = transforms.average_tiles_ND(y, subs, shape)
                    yfi = yfi[(Ellipsis,) + tuple(slice(s) for s in shape)]
                    yfi = yfi[:, :imgi.shape[2], :imgi.shape[3]]
                    yf[k * nimgs + i] = yfi
                    stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
                    stylei /= (stylei**2).sum()**0.5
                    styles.append(stylei)

        return yf, np.array(styles)

    IMG, subs, shape, inds = transforms.make_tiles_ND(
        imgi,
        bsize=bsize,
        augment=augment,
        normalize=normalize,
        tile_overlap=tile_overlap,
    )
    batch_size = self.batch_size
    niter = int(np.ceil(IMG.shape[0] / batch_size))
    nout = self.nclasses + 32 * return_conv
    y = np.zeros((IMG.shape[0], nout) + tuple(IMG.shape[-self.dim:]))
    for k in range(niter):
        irange = np.arange(batch_size * k, min(IMG.shape[0], batch_size * k + batch_size))
        y0, style = self.run_network(IMG[irange], return_conv=return_conv, to_numpy=to_numpy)
        arg = (len(irange),) + y0.shape[-(self.dim + 1):]
        y[irange] = y0.reshape(arg)
        if k == 0:
            styles = style[0]
        styles += style.sum(axis=0)
    styles /= IMG.shape[0]
    if augment:
        y = transforms.unaugment_tiles_ND(y, inds, self.unet)

    yf = transforms.average_tiles_ND(y, subs, shape)
    slc = tuple([slice(s) for s in shape])
    yf = yf[(Ellipsis,) + slc]
    styles /= (styles**2).sum()**0.5
    return yf, styles

def _run_nets(self, img, net_avg=True, augment=False, tile=False, normalize=True,
              tile_overlap=0.1, bsize=224, return_conv=False, to_numpy=True, progress=None):
    """run network (if more than one, loop over networks and average results

    Parameters
    --------------

    img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    net_avg: bool (optional, default True)
        runs the 4 built-in networks and averages them if True, runs one network if False

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    tile: bool (optional, default True)
        tiles image to ensure GPU memory usage limited (recommended)

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI

    Returns
    ------------------

    y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
        y is output (averaged over networks);
        y[0] is Y flow; y[1] is X flow; y[2] is cell probability

    style: array [64]
        1D array summarizing the style of the image,
        if tiled it is averaged over tiles,
        but not averaged over networks.

    """
    if isinstance(self.pretrained_model, str) or not net_avg:
        y, style = self._run_net(
            img,
            augment=augment,
            tile=tile,
            normalize=normalize,
            tile_overlap=tile_overlap,
            bsize=bsize,
            return_conv=return_conv,
            to_numpy=to_numpy,
        )
    else:
        for j in range(len(self.pretrained_model)):
            if j > 0:
                print("multi model averaging not working correctly, contact Kevin")
                if self.torch and self.gpu:
                    net = self.net.module
                else:
                    net = self.net

                net.load_model(self.pretrained_model[j], cpu=(not self.gpu))

                if not self.torch:
                    net.collect_params().grad_req = "null"

            y0, style = self._run_net(
                img,
                augment=augment,
                tile=tile,
                normalize=normalize,
                tile_overlap=tile_overlap,
                bsize=bsize,
                return_conv=return_conv,
                to_numpy=to_numpy,
            )

            if j == 0:
                y = y0
            else:
                y += y0

            if progress is not None:
                progress.setValue(10 + 10 * j)
        y = y / len(self.pretrained_model)

    return y, style



def _run_net(self, imgs,
             augment=False, tile=True, normalize=True,
             tile_overlap=0.1, bsize=224,
             return_conv=False, to_numpy=True):
    """run network on image or stack of images

    (faster if augment is False)

    Parameters
    --------------

    imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    rsz: float (optional, default 1.0)
        resize coefficient(s) for image

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    tile: bool (optional, default True)
        tiles image to ensure GPU/CPU memory usage limited (recommended);
        cannot be turned off for 3D segmentation

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]

    Returns
    ------------------

    y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
        y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

    style: array [64]
        1D array summarizing the style of the image,
        if tiled it is averaged over tiles

    """
    transpose = False
    if imgs.ndim == 4 and self.dim == 2:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        detranspose = (0, 2, 3, 1)
        return_conv = False
        transpose = True
    elif imgs.ndim > self.dim:
        order = (self.dim,) + tuple([k for k in range(self.dim)])
        imgs = np.transpose(imgs, order)
        transpose = True
        detranspose = tuple([k for k in range(1, self.dim + 1)]) + (0,)

    imgs, subs = transforms.pad_image_ND(imgs, dim=self.dim)
    slc = [slice(0, imgs.shape[n] + 1) for n in range(imgs.ndim)]
    slc[-(self.dim + 1)] = slice(0, self.nclasses + 32 * return_conv + 1)
    for k in range(1, self.dim + 1):
        slc[-k] = slice(subs[-k][0], subs[-k][-1] + 1)
    slc = tuple(slc)

    if tile or augment or (imgs.ndim == 4 and self.dim == 2):
        y, style = self._run_tiled(
            imgs,
            augment=augment,
            bsize=bsize,
            tile_overlap=tile_overlap,
            normalize=normalize,
            return_conv=return_conv,
            to_numpy=to_numpy,
        )
    else:
        imgs = np.expand_dims(imgs, axis=0)
        y, style = self.run_network(imgs, return_conv=return_conv, to_numpy=to_numpy)
        y, style = y[0], style[0]
    style /= (style**2).sum()**0.5

    y = y[slc]

    if transpose:
        y = np.transpose(y, detranspose) if isinstance(y, np.ndarray) else torch.transpose(y, detranspose)

    return y, style



def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=True,
            augment=False, tile=True, tile_overlap=0.1,
            normalize=True, bsize=224, progress=None):
    """run network on stack of images

    (faster if augment is False)

    Parameters
    --------------

    imgs: array [Lz x Ly x Lx x nchan]

    rsz: float (optional, default 1.0)
        resize coefficient(s) for image

    anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

    net_avg: bool (optional, default True)
        runs the 4 built-in networks and averages them if True, runs one network if False

    augment: bool (optional, default False)
        tiles image with overlapping tiles and flips overlapped regions to augment

    tile: bool (optional, default True)
        tiles image to ensure GPU/CPU memory usage limited (recommended);
        cannot be turned off for 3D segmentation

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles when computing flows

    bsize: int (optional, default 224)
        size of tiles to use in pixels [bsize x bsize]

    progress: pyqt progress bar (optional, default None)
        to return progress bar status to GUI


    Returns
    ------------------

    yf: array [Lz x Ly x Lx x 3]
        y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

    style: array [64]
        1D array summarizing the style of the image,
        if tiled it is averaged over tiles

    """
    sstr = ["YX", "ZY", "ZX"]
    if anisotropy is not None:
        rescaling = [[rsz, rsz],
                     [rsz * anisotropy, rsz],
                     [rsz * anisotropy, rsz]]
    else:
        rescaling = [rsz] * 3
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 1, 2, 0)]
    yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
    for p in range(3 - 2 * self.unet):
        xsl = imgs.copy().transpose(pm[p])
        shape = xsl.shape
        if rescaling[p] is not None and rescaling[p] != 1.0:
            xsl = transforms.resize_image(xsl, rsz=rescaling[p])
        core_logger.info("running %s: %d planes of size (%d, %d)" % (sstr[p], shape[0], shape[1], shape[2]))
        y, style = self._run_nets(
            xsl,
            net_avg=net_avg,
            augment=augment,
            tile=tile,
            normalize=normalize,
            bsize=bsize,
            tile_overlap=tile_overlap,
        )
        y = transforms.resize_image(y, shape[1], shape[2])
        yf[p] = y.transpose(ipm[p])
        if progress is not None:
            progress.setValue(25 + 15 * p)
    return yf, style
