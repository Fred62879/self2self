
import os
import cv2
import util
import time
import argparse
import numpy as np
import network.Punet
import configargparse
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from astropy.io import fits
from parser import parse_args


def train(dropout_rate, N_PREDICTION, args):
    # load data
    header = util.get_header(args.fits_fn, args.img_sz)
    gt = np.load(args.gt_img_fn).astype(np.float32) # [n,h,w,c]
    gt = gt.transpose((1,2,0))
    gt = np.expand_dims(gt, axis=0)
    _, w, h, c = np.shape(gt)

    masked_img, mask = util.mask_pixel\
        (gt, args.recon_dir, 1 - args.sample_ratio,
         args.sampled_pixl_id_fn, args.current_bands)

    #print('Training on %d percent pixls' % (args.sample_ratio*100))
    #print('GT max', np.round(np.max(gt, axis=(0,1,2)), 3) )
    #print('GT & mask shape', gt.shape, gt.dtype, mask.shape, mask.dtype,
    #      masked_img.shape, masked_img.dtype)

    # train
    tf.reset_default_graph()
    model = network.Punet.build_inpainting_unet(masked_img, mask, 1 - dropout_rate)

    saver = model['saver']
    avg_op = model['avg_op']
    summay = model['summary']
    non_zero = model['non_zero']
    loss = model['training_error']
    our_image = model['our_image']
    slice_avg = model['slice_avg']

    optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss)

    id, avg_loss = 0, 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.num_epochs):
            cur_non_zero, _, _op, loss_value, merged, o_image = \
                    sess.run([non_zero, optimizer, avg_op, loss, summay, our_image])
            avg_loss += loss_value

            if step == 0 or (step + 1) % args.model_smpl_intvl == 0:
                print("[Iteration/Total]:[%d/%d], [avg_loss]:[%.4f]" %
                      (step + 1, args.num_epochs, avg_loss / args.model_smpl_intvl))
                print('    Iteration {}, non zero mask {}'.format(step, cur_non_zero))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))

                for j in range(N_PREDICTION):
                    o_avg, o_image = sess.run([slice_avg, our_image])
                    sum += o_image

                recon = sum/N_PREDICTION
                recon = recon.reshape((args.img_sz, args.img_sz,-1)).transpose(2,0,1)
                recon_path = os.path.join(args.recon_dir, str(id+1))

                mask = None if not args.recon_restore else \
                    np.load(args.sampled_pixl_id_fn).T.reshape((-1,args.img_sz,args.img_sz))

                util.reconstruct(gt[0].transpose(2,0,1), recon, recon_path,
                                 args.metric_dir, mask=mask, header=header)
                id += 1

if __name__ == '__main__':
    start = time.time()

    parser = configargparse.ArgumentParser()
    config = parse_args(parser)
    args = argparse.Namespace(**config)

    N_PREDICTION = 1#100
    dropout_rate = 0.3
    train(dropout_rate, N_PREDICTION, args)

    duration = time.time() - start
    print('Timing ', duration )
