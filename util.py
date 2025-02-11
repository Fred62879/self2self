
import os
import cv2
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.compat.v1 as tf

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from skimage.metrics import structural_similarity


def get_header(fits_fn, sz):
    hdu = fits.open(os.path.join(fits_fn))[1]
    header = hdu.header
    cutout = Cutout2D(hdu.data, position=(sz//2, sz//2),
                      size=sz, wcs=WCS(header))
    return cutout.wcs.to_header()


def add_gaussian_noise(img, model_path, sigma):
    index = model_path.rfind("/")
    if sigma > 0:
        noise = np.random.normal(scale=sigma / 255., size=img.shape).astype(np.float32)
        sio.savemat(model_path[0:index] + '/noise.mat', {'noise': noise})
        noisy_img = (img + noise).astype(np.float32)
    else:
        noisy_img = img.astype(np.float32)
    cv2.imwrite(model_path[0:index] + '/noisy.png',
                np.squeeze(np.int32(np.clip(noisy_img, 0, 1) * 255.)))
    return noisy_img


def load_np_image(path):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0).astype(float32)
    return img


# img [n,h,w,n_dim]
# 0-masked, 1-use for train
def mask_pixel(img, recon_dir, ratio, mask_path, current_bands):
    assert(os.path.exists(mask_path))

    n_dim = img.shape[-1]
    masked_img = img.copy()

    # load full mask
    print(f'    load mask from {mask_path}')
    mask = np.load(mask_path) # [npixls,full_nchls]
    print(mask.shape)
    smpl_pixls = [np.count_nonzero(mask[:,i])
                  for i in range(mask.shape[1])]
    print('    sampled pixls for full mask', smpl_pixls)

    # slide only bands needed
    mask = mask[:,current_bands]
    (npixls, nchls) = mask.shape

    img_sz = int(np.sqrt(npixls))
    mask = mask.reshape((img_sz, img_sz, nchls))[np.newaxis]
    mask = mask.astype(np.float32)
    masked_img *= mask

    mask_fn = os.path.join(recon_dir,'mask.npy')
    masked_img_fn = os.path.join(recon_dir,'masked_img.npy')
    np.save(mask_fn, mask)
    np.save(masked_img_fn, masked_img)

    return masked_img, mask

def restore_unmasked(recon, gt, mask):
    ''' fill recon with unmasked pixels from gt
        recon/gt/mask [nbands,sz,sz]
    '''
    (nbands,sz,sz) = recon.shape
    npixls = sz**2

    recon = recon.reshape((nbands, -1))
    gt = gt.reshape((nbands, -1))

    for i in range(nbands):
        cur_mask = mask[i]
        is_train_band = np.count_nonzero(cur_mask) == npixls
        if is_train_band:
            recon[i] = gt[i]
        else:
            ids = np.arange(npixls).reshape((sz,sz))
            unmasked_ids = (ids[cur_mask == 1]).flatten()
            recon[i][unmasked_ids] = gt[i][unmasked_ids]

    recon = recon.reshape((-1,sz,sz))
    return recon

# gt/recon, [c,h,w]
def reconstruct(gt, recon, recon_path, loss_dir, mask=None, header=None):
    sz = gt.shape[1]
    np.save(recon_path + '.npy', recon)

    print('    GT max', np.round(np.max(gt, axis=(1,2)), 2) )
    print('    Recon pixl max ', np.round(np.max(recon, axis=(1,2)), 2) )
    #print('    Recon stat ', round(np.min(recon), 3), round(np.median(recon), 6),
    #      round(np.mean(recon), 3), round(np.max(recon), 3))

    if mask is not None:
        recon_restored = restore_unmasked(recon, gt, mask)
        np.save(recon_path + '_restored.npy', recon_restored)

    if header is not None:
        hdu = fits.PrimaryHDU(data=recon, header=header)
        hdu.writeto(recon_path + '.fits', overwrite=True)

    losses = get_losses(gt, recon, None, [1,2,4])

    for nm, loss in zip(['mse','psnr','ssim'], losses):
        fn = nm + '.npy'
        loss = np.expand_dims(loss, axis=0)
        if nm == 'ssim': rd = 5
        else: rd = 2
        print(f'    {nm}', np.round(loss, rd))
        np.save(os.path.join(loss_dir, fn), loss)

def calculate_ssim(gen, gt):
    rg = np.max(gt)-np.min(gt)
    return structural_similarity(gt, gen, data_range=rg)
                                 #win_size=len(org_img))
# image shape should be [sz,sz,nchls]
def calculate_sam(pred_img, org_img, convert_to_degree=False):
    numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
    denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = sam_angles * 180.0 / np.pi
    return np.mean(np.nan_to_num(sam_angles))

def calculate_psnr(gen, gt):
    mse = calculate_mse(gen, gt)
    return 20 * np.log10(np.max(gt) / np.sqrt(mse + 1e-10))

def calculate_mse(gen, gt):
    mse = np.mean((gen - gt)**2)
    return mse

# calculate normalized cross correlation between given 2 imgs
def calculate_ncc(img1, img2):
    a, b = img1.flatten(), img2.flatten()
    n = len(a)
    return 1/n * np.sum( (a-np.mean(a)) * (b-np.mean(b)) /
                         np.sqrt(np.var(a)*np.var(b)) )

def get_loss(gt, gen, mx, j, option):
    if option == 0:
        loss = np.abs(gen[j] - gt[j]).mean()
    elif option == 1:
        loss = calculate_mse(gen[j], gt[j])
    elif option == 2:
        loss = calculate_psnr(gen[j], gt[j])
    elif option == 3:
        loss = calculate_sam(gen[:,:,j:j+1], gt[:,:,j:j+1])
    elif option == 4:
        loss = calculate_ssim(gen[j], gt[j])
    elif option == 5: # min
        loss = np.min(gen[j])
    elif option == 6: # max
        loss = np.max(gen[j])
    elif option == 7: # meam
        loss = np.mean(gen[j])
    elif option == 8: # median
        loss = np.median(gen[j])
    return loss

# calculate losses between gt and gen based on options
def get_losses(gt, gen, mx, options):
    nchl = gen.shape[0]
    losses = np.zeros((len(options), nchl))

    for i, option in enumerate(options):
        for j in range(nchl):
            losses[i, j] = get_loss(gt, gen, mx, j, option)
    return losses
