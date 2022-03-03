
import os
import cv2
import util
import argparse
import numpy as np
import network.Punet
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from astropy.io import fits

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nfls', type=int, default=5)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=100)
    parser.add_argument('--niters', type=int, default=100)
    parser.add_argument('--spectral', action='store_true')

    args = parser.parse_args()

    TF_DATA_TYPE = tf.float32
    N_PREDICTION = 1#100

    batch_sz = 1
    mask_mode = 2
    learning_rt = 1e-4
    dropout_rate = 0.3
    mask_reverse = False

    test = args.test
    load = args.load
    nfls = args.nfls      # num bands
    ratio = args.ratio    # ratio %
    img_sz = args.imgsz
    n_iters = args.niters
    spectral = args.spectral
    save_intvl = n_iters // 4

    loss = 'l1_'
    dim = '2d_'+ str(nfls)
    data_dir = '../../data'
    mask_dir = os.path.join(data_dir, 'pdr3_output/sampled_id',
                            'spectral' if spectral else 'spatial')
    output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/S2S',
                              str(img_sz), loss + str(ratio))

    loss_dir = os.path.join(output_dir, 'losses')
    model_dir = os.path.join(output_dir, 'models')
    recon_dir = os.path.join(output_dir, 'reconstr_imgs')
    n = len(os.listdir(model_dir))
    model_path = os.path.join(model_dir, 'model' + str(n-1) + '.pth')

    mask_path = os.path.join(mask_dir, str(img_sz)+'_'+str(ratio)+'_mask.npy')
    img_path = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs/0_'+str(img_sz)+'_rfr.npy')


    # load data
    header = util.get_header(data_dir, img_sz)
    gt = np.load(img_path).astype(np.float32) # [n,h,w,c]
    gt = np.expand_dims(gt, axis=0)
    _, w, h, c = np.shape(gt)

    masked_img, mask = util.mask_pixel(gt, recon_dir, 1 - ratio, mask_path)

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
    loss = model['training_error']
    our_image = model['our_image']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(learning_rt).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(n_iters):
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image])
            avg_loss += loss_value

            if (step + 1) % save_intvl == 0:
                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / save_intvl))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))

                for j in range(N_PREDICTION):
                    o_avg, o_image = sess.run([slice_avg, our_image])
                    sum += o_image

                recon = sum/N_PREDICTION
                recon = recon.reshape((img_sz,img_sz,-1)).transpose(2,0,1)
                recon_path = os.path.join(recon_dir, '0_'+str(img_sz)+'_'+str(step) + '_0')
                util.reconstruct(gt, recon, recon_path, loss_dir, header=None)

                #recon = sum / N_PREDICTION
                #recon_fn = os.path.join(recon_dir, '0_'+str(img_sz)+'_'+str(step + 1) + '_0.npy')
                #np.save(recon_fn, recon)

                #o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                #cv2.imwrite(o_image_fn, o_image)
                #saver.save(sess, os.path.join(model_dir, 'model'+str(step)))

if __name__ == '__main__':
    train()
    '''
    path = './testsets/Set11/'
    file_list = os.listdir(path)
    mask_rate = 0.01
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(path + file_name, 0.3, mask_rate)
    '''
