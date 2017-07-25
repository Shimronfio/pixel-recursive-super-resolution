import sys
import os
import time
from PIL import Image 
import argparse
from net import *
#repair the error of sse4 of tensorflow: 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 

#let count time: 
t1 = time.time() 


#next line will help you choose the picture you want to edit: 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--low", required=True,
	help="path to the image")
ap.add_argument("-s", "--high", required=True,
	help="path to the image")
args = vars(ap.parse_args())

#image will load: 
lr_imgs = Image.open(args["low"])
hr_imgs = Image.open(args["high"])

#load back the models you did from training:
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./models/model.ckpt.meta')
	saver.restore(sess, "./models/model.ckpt")
	print('models loaded : ')
	#graph=tf.get_default_graph()   
	#output=graph.get_tensor_by_name('./gadot.jpg')
	#ph = graph.get_tensor_byname()
	#results = sess.run(output,feed_dict={ph:INPUT})

mu =1.1 

    c_logits = net.conditioning_logits
    p_logits = net.prior_logits
    np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
    gen_hr_imgs = np.zeros( 32, 32, 3), dtype=np.float32)
    np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, self.net.train:False})
    print('inside creator : ')
    
    for i in range(32):
      for j in range(32):
        for c in range(3):
          np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
          new_pixel = logits_2_pixel_value(np_c_logits[i, j, c*256:(c+1)*256] + np_p_logits[i, j, c*256:(c+1)*256], mu=mu)
          gen_hr_imgs[i, j, c] = new_pixel
    #

'''
    save_samples(np_lr_imgs, self.samples_dir + '/lr_' + str(mu*10) + '_' + str(step) + '.jpg')
    save_samples(np_hr_imgs, self.samples_dir + '/hr_' + str(mu*10) + '_' + str(step) + '.jpg')
    save_samples(gen_hr_imgs, self.samples_dir + '/generate_' + str(mu*10) + '_' + str(step) + '.jpg')
'''
	gen_hr_imgs.save('result.jpg',"JPEG")
sess.close()

# ending session: 
t2 = time.time()
print ('done , time run : %.3f seconds' % (t2-t1))




