
import os
import cv2
import util
import time
import argparse
import numpy as np
import network.Punet
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from astropy.io import fits

# python3 demo_inpainting.py --nfls 5 --imgsz 512 --ratio 10 --niters 20

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nfls', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--ratio', type=float, default=100)
    parser.add_argument('--niters', type=int, default=100)
    parser.add_argument('--spectral', action='store_true')

    args = parser.parse_args()

    TF_DATA_TYPE = tf.float32
    N_PREDICTION = 1#100

    learning_rt = 1e-4
    dropout_rate = 0.3

    nfls = args.nfls      # num bands
    ratio = float(args.ratio)    # ratio %
    img_sz = args.imgsz
    n_iters = args.niters
    spectral = args.spectral
    save_intvl = n_iters // 4

    loss = 'l1_'
    dim = '2d_'+ str(nfls)
    data_dir = '../../data'
    sz_str = str(img_sz) + ('_spectra' if spectral else '')

    mask_dir = os.path.join(data_dir, 'pdr3_output/sampled_id',
                            'spectral' if spectral else 'spatial')

    output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/S2S',
                              sz_str, loss + str(ratio))

    loss_dir = os.path.join(output_dir, 'losses')
    model_dir = os.path.join(output_dir, 'models')
    recon_dir = os.path.join(output_dir, 'reconstr_imgs')
    mask_path = os.path.join(mask_dir, str(img_sz)+'_'+str(ratio)+'_mask.npy')
    img_path = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/0_'+str(img_sz)+'.npy')

    # load data
    header = util.get_header(data_dir, img_sz)
    gt = np.load(img_path).astype(np.float32) # [n,h,w,c]
    gt = gt.transpose((1,2,0))
    gt = np.expand_dims(gt, axis=0)
    _, w, h, c = np.shape(gt)

    masked_img, mask = util.mask_pixel(gt, recon_dir, 1 - ratio, mask_path)
    print(mask.shape, np.count_nonzero(mask[...,2]))

    print('GT max', np.round(np.max(gt, axis=(0,1,2)), 3) )
    print('Training on {}% pixls'.format(ratio))
    print('GT & mask shape', gt.shape, gt.dtype, mask.shape, mask.dtype,
          masked_img.shape, masked_img.dtype)

    # train
    tf.reset_default_graph()
    model = network.Punet.build_inpainting_unet(masked_img, mask, 1 - dropout_rate)

    saver = model['saver']
    avg_op = model['avg_op']
    summay = model['summary']
    non_zero = model['non_zero']
    #init_nonzero = model['init_nonzero']
    #drop_nonzero = model['drop_nonzero']
    loss = model['training_error']
    our_image = model['our_image']
    slice_avg = model['slice_avg']

    optimizer = tf.train.AdamOptimizer(learning_rt).minimize(loss)

    id = 0
    avg_loss = 0

    start = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(n_iters):
            cur_non_zero, _, _op, loss_value, merged, o_image = \
                    sess.run([non_zero, optimizer, avg_op, loss, summay, our_image])
            '''
            init_nonzero, drop_nonzero, cur_non_zero, _, _op, \
                loss_value, merged, o_image = \
                    sess.run([init_nonzero, drop_nonzero, non_zero,
                              optimizer, avg_op, loss, summay, our_image])
            '''

            #print('Iteration {}, non zero mask {}'.format(step, cur_non_zero))

            avg_loss += loss_value

            if step == 0 or (step + 1) % save_intvl == 0:
                print("[Iteration/Total]:[%d/%d], [avg_loss]:[%.4f]" %
                          (step+1, n_iters, avg_loss / save_intvl))
                print('Iteration {}, non zero mask {}'.format(step, cur_non_zero))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))

                for j in range(N_PREDICTION):
                    o_avg, o_image = sess.run([slice_avg, our_image])
                    sum += o_image

                recon = sum/N_PREDICTION
                recon = recon.reshape((img_sz,img_sz,-1)).transpose(2,0,1)
                recon_path = os.path.join(recon_dir, '0_'+str(img_sz) +
                                          '_' + str(id+1) + '_0')
                util.reconstruct(gt[0].transpose(2,0,1), recon,
                                 recon_path, loss_dir, header=header)

                '''
                if id == 3: # last model
                    recon = sum/N_PREDICTION
                    print(recon.shape)
                    recon = recon.reshape((img_sz,img_sz,-1)).transpose(2,0,1)
                    print(recon.shape)
                    recon_path = os.path.join(recon_dir, '0_'+str(img_sz)+'_'+str(id+1) + '_0')
                    util.reconstruct(gt[0].transpose(2,0,1), recon,
                                     recon_path, loss_dir, header=header)
                '''
                id += 1


    duration = time.time() - start
    print('Timing ', duration )

if __name__ == '__main__':
    train()
