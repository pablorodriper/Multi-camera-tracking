"""
Created on Wed Mar  7 18:50:06 2017

@author: joans
"""
import os
import time
import numpy as np

import tensorflow as tf


from siamese.vgg16 import Vgg16


""" Siamese or triplet in one tower architecture, depending on the chosen loss. Training
is done with Dataset.next_batch_pk() that depends on two parameters, P=number of sampled
classes and K=number of images per class. Both for contrastive and triplet loss. In the
first case, with P, K control the ratio of same/diffent pairs, in total PK(PK-1). In the
second, the number of valid tripets is printed, for P=5, K=3 we get 360 valid out of 2730. """
class One_tower():
    def __init__(self, dim_embedding, image_size):
        self.dim_embedding = dim_embedding
        self.image_size = image_size
        datestr = time.asctime().replace(' ','_').replace(':','_')        
        self.path_experiment = 'experiments/'+datestr
               
        tf.reset_default_graph()
        self.make_placeholders()
        self.make_model()
                   

    def make_placeholders(self):
        shape_x = [None, self.image_size, self.image_size, 3]
        self.x = tf.placeholder(tf.float32, shape_x)
        self.y = tf.placeholder(tf.int64, [None])
        self.keep_prob_fc = tf.placeholder(tf.float32)
        self.val_P = tf.placeholder(tf.int32)
        self.val_K = tf.placeholder(tf.int32)


    def make_model(self):
        with tf.variable_scope("tower"):
            self.out = tf.nn.l2_normalize(self.network_branch(self.x),dim=1)


    def network_branch(self, x):
        self.vgg16 = Vgg16(x, self.keep_prob_fc, width_fc=512, 
                           num_features=self.dim_embedding)
        return self.vgg16.out
                

    def make_loss(self):
        if self.type_loss=='contrastive':
            self.loss = self.make_contrastive_loss()
            
        elif self.type_loss=='triplet':
            self.loss = self.make_triplet_loss()
                
        else:
            assert False

                
                
    """ See test_contrastive.py """
    def make_pairs(self):
        print('making contrastive loss')
        """ converts self.out, self.label into pairs (out1, out2) and
        same/different class, y """
        r = tf.range(tf.shape(self.out)[0])
        idx_i, idx_j = tf.meshgrid(r, r)
        """ cartesian product """
        idx = tf.where(tf.greater(idx_i, idx_j))
        out1, out2 = tf.split(tf.gather(self.out, idx),axis=1, num_or_size_splits=2)
        lab1, lab2 = tf.split(tf.gather(self.y, idx), axis=1, num_or_size_splits=2)
        out1 = tf.squeeze(out1)
        out2 = tf.squeeze(out2)
        lab1 = tf.squeeze(lab1)
        lab2 = tf.squeeze(lab2)
        self.out1 = out1
        y_similar_diff = 1.0 - tf.to_float(tf.equal(lab1, lab2))
        """ 1 - tf.to_... because the convention in the contrastive loss is that
        1 = different and 0 = same, but float(True)==1 """
        print('done contrastive loss')
        return out1, out2, y_similar_diff



    """ contrastive loss, original or modified with margin for similar pairs """
    def contrastive_loss(self, out1, out2, y, two_margins=False):        
        eucd2 = tf.pow(out1 - out2, 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        """ ||CNN(p1i)-CNN(p2i)||_2^2 """
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        """ ||CNN(p1i)-CNN(p2i)||_2 """
        margin = tf.constant(self.margin, dtype=tf.float32, name="margin")
        """ self.y : 0 = same class, 1 = different class """

        if two_margins:
            """ with additional margin for same class, stops pulling
            closer similar samples if their distance <= second margin.
            we set it to 1/20th of the other margin which is 1.0 """
            margin_same = tf.constant(0.05, dtype=tf.float32, name="margin_same")
            same = tf.multiply(tf.subtract(1.0, y, name="1-yi"),
                               tf.pow(tf.maximum(0.0, eucd - margin_same), 2),
                               name="same_class")
        else:
            """ original version, keeps pulling closer samples from
            same class all the time """
            same = tf.multiply(tf.subtract(1.0, y, name="1-yi"), eucd2,
                               name="same_class")

        different = tf.multiply(y, tf.pow(tf.maximum(0.0, margin - eucd), 2),
                           name="different_class")
        pair_losses = tf.add(same, different, name='pair_losses')
        self.loss_same = tf.reduce_mean(same)
        self.loss_different = tf.reduce_mean(different)
        self.num_different = tf.reduce_sum(y)
        self.pair_losses = pair_losses
        return pair_losses



    def make_contrastive_loss(self):
        out1, out2, y = self.make_pairs()
        pair_losses = self.contrastive_loss(out1, out2, y)
        loss = tf.reduce_mean(pair_losses, name="loss")
        return loss
        


    """
    See test_triplet.py.
    Creates all possible valid triplets  (anchor, positive, negative) 
    from self.out taking into account self.label : anchor and positive
    are from the same class and negative from different class than anchor.
    This is 'Batch All' mining. Several mining algorithms are already
    implemented in tensorflow 1.6, see
    https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
    """
    def make_triplets(self):
        lab1, lab2, lab3 = tf.meshgrid(self.y, self.y, self.y)
        """cartesian product: all possible 3-tuples of labels"""
        idx = tf.where(tf.logical_and(tf.equal(lab1, lab2),
                                      tf.not_equal(lab1, lab3)))
        """ indices of positions where anchor and positive are
        of same class (label), negative different class """
        idx1, idx2, _ = tf.split(idx, axis=1,
                                 num_or_size_splits=3)
        idx1, idx2 = tf.squeeze(idx1), tf.squeeze(idx2)
        idx_dif = tf.squeeze(tf.gather(idx,
                                       tf.where(tf.not_equal(idx1, idx2))))
        """ anchor different from positive """
        selected_emb = tf.gather(self.out, idx_dif)
        s1, s2, s3 = tf.split(selected_emb, axis=1,
                              num_or_size_splits=3)
        """ the 3 elements of a triplet, for all triplets """
        anchors = tf.squeeze(s1)
        positives = tf.squeeze(s2)
        negatives = tf.squeeze(s3)
        return anchors, positives, negatives


    """ For each 3-tuple (a_i,p_i,n_i) computes 
    [d_ap]_i = ||CNN(a_i) - CNN(p_i)||_2 and 
    [d_an]_i = ||CNN(a_i) - CNN(n_i)||_2 """
    def compute_distances(self, anchors, positives, negatives):
        d2_ap = tf.reduce_sum( tf.pow(anchors - positives, 2), 1)
        d_ap = tf.sqrt(d2_ap + 1e-8, name="eucd_ap")
        d2_an = tf.reduce_sum( tf.pow(anchors - negatives, 2), 1)
        d_an = tf.sqrt(d2_an + 1e-8, name="eucd_an")
        return d_ap, d_an
       

    """ Basic version of triplet loss plus ratio of distances and non-zero mean variants """
    def triplet_loss(self, anchors, positives, negatives, ratio_distances=False, nonzero_mean=False):
        """
        Triplet loss based on Andrew Ng course.
        https://www.coursera.org/lecture/convolutional-neural-networks/triplet-loss-HuUtN

        :param anchors:
        :param positives:
        :param negatives:
        :param ratio_distances:
        :param nonzero_mean:
        :return:
        """
        # TODO

        d2_ap = tf.reduce_sum(tf.pow(anchors - positives, 2), 1, name='eucd_ap_2')
        d2_an = tf.reduce_sum(tf.pow(anchors - negatives, 2), 1, name='eucd_an_2')
        margin = tf.constant(self.margin, name="margin")
        loss = tf.maximum(0.0, d2_ap - d2_an + margin)

        return loss

        

    """ For each anchor, find the hardest positive and the hardest negative,
    and the compute their triplet loss. The way we do it (reshapes) assumes
    an equal number of triplets for a given anchor. If we have drawn PK images
    from P different classes, K images per class, this number is  (K-1)*(P-1)*K :
    foran anchor of some class, there are K-1 positives and (P-1)*K possible
    negatives. """
    def batch_hard_loss(self, anchors, positives, negatives):
        d_ap, d_an = self.compute_distances(anchors, positives, negatives) 
        margin = tf.constant(self.margin, dtype=tf.float32, name="margin")
        d_ap_by_a = tf.reshape(d_ap, [-1,(self.val_P-1)*self.val_K])
        d_an_by_a = tf.reshape(d_an, [-1,(self.val_P-1)*self.val_K])
        d_hard_pos = tf.reduce_max(d_ap_by_a, axis=1)
        d_hard_neg = tf.reduce_min(d_an_by_a, axis=1)
        batch_hard_losses = tf.maximum(0.0, margin + d_hard_pos - d_hard_neg)
        loss_bh = tf.reduce_mean(batch_hard_losses, name='batch_hard_loss')
        return loss_bh



    def make_triplet_loss(self):
        print('making triplet loss')
        self.anchors, self.positives, self.negatives = self.make_triplets()
        if self.type_mining=='batch all':
            loss = self.triplet_loss(self.anchors, self.positives, self.negatives,
                                          nonzero_mean=False)
                                          
        elif self.type_mining=='batch all nonzero mean':
            loss = self.triplet_loss(self.anchors, self.positives, self.negatives,
                                          nonzero_mean=True)
       
        elif self.type_mining=='batch hard loss':
            loss = self.batch_hard_loss(self.anchors, self.positives, self.negatives)
                                          
        else:
            assert False

        print('done triplet loss ')
            
        return loss



    def train(self, margin, type_loss, type_mining, dataset, learning_rate, max_steps,
              P, K, keep_prob_fc, show_loss_every_steps, save_weights_every_steps):
        self.margin = margin
        self.type_loss = type_loss
        self.type_mining = type_mining

        self.make_loss()
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print('begin training')
            for step in range(1,max_steps+1):
                batch = dataset.next_batch_pk(P,K)
                feed_dict = { self.x: batch.x,
                              self.y: batch.y,
                              self.keep_prob_fc: keep_prob_fc,
                              self.val_P: P,
                              self.val_K: K, # needed in batch_hard_loss()
                             }
                _, bloss = sess.run([train_op, self.loss], feed_dict = feed_dict)


                if step==1:
                    if self.type_loss=='triplet':
                        anchors = self.anchors.eval(feed_dict = feed_dict)
                        print('{} valid triplets per batch'.format(anchors.shape[0]))
                        
                    if self.type_loss=='contrastive':
                        pair_losses, num_different = \
                            sess.run([self.pair_losses, self.num_different],
                            feed_dict = feed_dict)
                        print('{} pairs per batch, {} pairs of different class'.\
                            format(pair_losses.shape, num_different))
 
                
                if step % show_loss_every_steps == 0:
                    if self.type_loss=='contrastive':
                        loss_same, loss_different = \
                            sess.run([self.loss_same, self.loss_different],
                                     feed_dict = feed_dict)
                        print('step {}, loss {}, loss same {}, loss different {}'.\
                            format(step, bloss, loss_same, loss_different))

                    if self.type_loss=='triplet':
                        print('step {}, loss {}'.format(step, bloss))
 

                if step % save_weights_every_steps == 0:
                    self.save_weights(sess, saver)
                    
                    

    def save_weights(self, sess, saver):
        if not os.path.isdir(self.path_experiment):
            os.makedirs(self.path_experiment)
            print('made output directory {}'.format(self.path_experiment))

        saver.save(sess, os.path.join(self.path_experiment, 'model.ckpt'))
        print('saved weights at {}'.format(self.path_experiment))

    def load_weights(self, path_experiment, saver, sess):
        checkpoint_dir = path_experiment
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('loaded weights from {}'.format(ckpt.model_checkpoint_path))
        else:
            print('ERROR: no checkpoint found for path {}'.format(path_experiment))
            assert False

    def inference_one_image(self, ima, path_experiment):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)

        emb =  self.out.eval({self.x: [ima], self.keep_prob_fc: 1.0})
        point = list(emb[0])
        return point

    def inference_detection(self, ima, path_experiment):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)

            emb =  self.out.eval({self.x: [ima], self.keep_prob_fc: 1.0})
        return emb

    def inference_detections(self, detections_to_embed, path_experiment):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)
            embeds = []
            for det in detections_to_embed:
                image = detections_to_embed[det]
                emb =  self.out.eval({self.x: [image], self.keep_prob_fc: 1.0})
                embeds.append(dict(detection=det, embedding=emb))
        return embeds

    def inference_dataset(self, ds, path_experiment): 
        labels = []
        points = []
        images = []
        
        saver = tf.train.Saver()
        i = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            self.load_weights(path_experiment, saver, sess)
            
            for (ima, lab) in ds.samples():
                emb = self.out.eval({self.x: [ima], self.keep_prob_fc: 1.0})
                points.append(list(emb[0]))
                labels.append(lab)
                images.append(ima)
                
                i += 1
                if i%100==0:
                    print(i)
                                
        return np.array(labels), np.array(points), np.array(images)

