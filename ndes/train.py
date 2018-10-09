import tensorflow as tf
import numpy as np
import numpy.random as rng
import os
import tqdm

class Trainer:
    """
    Training class for the standard MADEs/MAFs classes using a tensorflow optimizer.
    """
    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}):
        """
        Constructor that defines the training operation.
        :param model: made/maf instance to be trained.
        :param optimizer: tensorflow optimizer class to be used during training.
        :param optimizer_arguments: dictionary of arguments for optimizer intialization.
        """

        self.model = model

        # If the model has batch norm and it is activated, update operations on moving
        # mean and moving average have to be added to the training operation
        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)
        else:
            self.has_batch_norm = False
            self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)


    def train(self, sess, train_data, val_data=None, validation_split = 0.1, epochs=1000, batch_size=100,
              patience=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: train data to be used.
        :param val_data: validation data to be used for early stopping. If None, train_data is split
             into p_val percent for validation randomly.
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_epochs: maximum number of epochs for training.
        :param batch_size: batch size of each batch within an epoch.
        :param early_stopping: number of epochs for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """

        train_idx = np.arange(train_data.shape[0])

        # If no validation data was found, split training into training and
        # validation data using p_val percent of the data
        if val_data == None:
            rng.shuffle(train_idx)
            val_data = train_data[train_idx[-int(validation_split*train_data.shape[0]):]]
            train_data = train_data[train_idx[:-int(validation_split*train_data.shape[0])]]
            train_idx = np.arange(train_data.shape[0])

        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()

        # Main training loop
        epoch_range = tqdm.trange(epochs, desc = "Epochs")
        for epoch in epoch_range:
            # Shuffel training indices
            rng.shuffle(train_idx)
            for batch in range(len(train_idx)//batch_size):
                # Last batch will have maximum number of elements possible
                batch_idx = train_idx[batch*batch_size:np.min([(batch+1)*batch_size,len(train_idx)])]
                if self.has_batch_norm:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx],self.model.training:True})
                else:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx]})

            # Early stopping check
            if epoch%check_every_N == 0:
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data})
                if show_log:
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data})
                    epoch_range.set_postfix(train_loss = train_loss, val_loss = this_loss)
                    #print("Epoch {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(epoch,train_loss,this_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= patience:
                break

        if show_log:
            print("Training finished")
            print("Best epoch {:05d}, Val_loss: {:05.4f}".format(epoch-check_every_N,bst_loss))

        # Restore best model
        saver.restore(sess,"./"+saver_name)
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)


class ConditionalTrainer():

    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}):
        """
            Constructor that defines the training operation.
            :param model: made/maf instance to be trained.
            :param optimizer: tensorflow optimizer class to be used during training.
            :param optimizer_arguments: dictionary of arguments for optimizer intialization.
            """

        self.model = model

        # If the model has batch norm and it is activated, update operations on moving
        # mean and moving average have to be added to the training operation
        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)
        else:
            self.has_batch_norm = False
            self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)

    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """
    def train(self, sess, train_data, validation_split = 0.1, epochs=1000, batch_size=100,
              patience=20, check_every_N=1, saver_name='tmp_model'):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param validation_split: percentage of training data randomly selected to be used for validation
        :param epochs: maximum number of epochs for training.
        :param batch_size: batch size of each batch within an epoch.
        :param early_stopping: number of epochs for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """

        # Training data
        train_data_X, train_data_Y  = train_data
        train_idx = np.arange(train_data_X.shape[0])

        # validation data using p_val percent of the data
        rng.shuffle(train_idx)
        N = train_data_X.shape[0]
        val_data_X = train_data_X[train_idx[-int(validation_split*N):]]
        train_data_X = train_data_X[train_idx[:-int(validation_split*N)]]
        val_data_Y = train_data_Y[train_idx[-int(validation_split*N):]]
        train_data_Y = train_data_Y[train_idx[:-int(validation_split*N)]]
        train_idx = np.arange(train_data_X.shape[0])

        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()

        # Validation and training losses
        validation_losses = []
        training_losses = []

        # Main training loop
        epoch_range = tqdm.trange(epochs, desc = "Epochs")
        for epoch in epoch_range:
            # Shuffel training indices
            rng.shuffle(train_idx)
            for batch in range(len(train_idx)//batch_size):
                # Last batch will have maximum number of elements possible
                batch_idx = train_idx[batch*batch_size:np.min([(batch+1)*batch_size,len(train_idx)])]
                if self.has_batch_norm:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                      self.model.y:train_data_Y[batch_idx],
                                                      self.model.training:True})
                else:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                      self.model.y:train_data_Y[batch_idx]})
            # Early stopping check
            if epoch%check_every_N == 0:
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data_X,
                                                                    self.model.y:val_data_Y})
                train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data_X,
                                                                         self.model.y:train_data_Y})
                epoch_range.set_postfix(train_loss = train_loss, val_loss = this_loss)
                #print("Epoch {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(epoch,train_loss,this_loss))
                validation_losses.append(this_loss)
                training_losses.append(train_loss)

                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= patience:
                break
        print("Training finished")
        print("Best epoch {:05d}, Val_loss: {:05.4f}".format(epoch-check_every_N,bst_loss))

        # Restore best model
        saver.restore(sess,"./"+saver_name)

        return np.array(validation_losses), np.array(training_losses)
