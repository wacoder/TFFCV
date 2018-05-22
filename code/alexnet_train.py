import argparse
import os
import tensorflow as tf
from alexnet import alexnet_v2
from utils import inputs

def run_training(tf_records, batch_size, num_epoch):
    with tf.Graph().as_default():
        images, labels = inputs(tf_records, batch_size, num_epoch)

        train_mode = tf.placeholder(tf.bool)
        logits = alexnet_v2(images, is_training=train_mode)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        total_loss = cross_entropy_mean + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        
        # prediction accuracy
        acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

        # training operation 
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001, global_step=global_step, decay_steps=3000, decay_rate=0.9, 
        staircase=True)

        optimizer = tf.train.AdamOptimizer(lr)
        
        
        train_ops = optimizer.minimize(total_loss, global_step)
        init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # summary
        tf.summary.scalar("Accuracy", acc)
        tf.summary.scalar("learning rate", lr)

        # summary writer
        merged = tf.summary.merge_all()
        
        
        # checkpoint writer
        new_saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state('./models')

        with tf.Session() as sess:
            sess.run(init_ops)
            train_writer = tf.summary.FileWriter('./log_dir', sess.graph)
            if ckpt and ckpt.model_checkpoint_state('./models'):
                new_saver.restore(sess, ckpt.model_checkpoint_path)
                print('restore and continue training!')

            
            try:
                while True:    
                    step = sess.run(global_step)
                    _, summary = sess.run([train_ops, merged], feed_dict={train_mode:True})
                    # _ = sess.run([train_ops], feed_dict={train_mode:True})
                    train_writer.add_summary(summary, step)
                    print(step)
                    
                    if step %1000 == 0:
                        save_path = new_saver.save(sess, os.path.join('./models', 'model.ckpt'), global_step=global_step)
                        print("Model saved in file {}".format(save_path))
            except tf.errors.OutOfRangeError:
                print('Done training for {} epoches, {} steps'.format(epoches, step))
            finally:
                save_path = new_saver.save(sess, os.path.join('./models', 'model.ckpt'), global_step=global_step)
                print("Model saved in file {}".format(save_path))

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', type=str, default="../data/tfrecords/*.tfrecords")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=40)
    parser.add_argument('--cuda', action='store_true', default=False)

    
    args = parser.parse_args()

    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] =''
    run_training(args.tfrecord_path, args.batch_size, args.num_epoch)