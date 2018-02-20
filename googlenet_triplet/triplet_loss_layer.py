import caffe
import numpy as np
from numpy import *
import time


class TripletLossLayer(caffe.Layer):
    global no_residual_list, margin, aux_idx

    def setup(self, bottom, top):
        """Setup the TripletDataLayer."""

        assert shape(bottom[0].data) == shape(bottom[1].data)
        assert shape(bottom[1].data) == shape(bottom[2].data)

        params = eval(self.param_str)
        print params
        self.margin = params['margin']

        self.a = 1
        top[0].reshape(1)
        top[1].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        anchor_minibatch_db = []
        positive_minibatch_db = []
        negative_minibatch_db = []
        for i in range((bottom[0]).num):
            anchor_minibatch_db.append(bottom[0].data[i])
            positive_minibatch_db.append(bottom[1].data[i])
            negative_minibatch_db.append(bottom[2].data[i])

        eq = 0
        loss = float(0)
        self.no_residual_list = []
        correct_pairs = 0
        # print "-------------Start Batch --> min(a): " + str(min(np.array(anchor_minibatch_db[0]))) + " /  a: " + str(np.array(anchor_minibatch_db[0][0:5]))

        self.aux_idx = 0
        for i in range(((bottom[0]).num)):
            a = np.array(anchor_minibatch_db[i])
            p = np.array(positive_minibatch_db[i])
            n = np.array(negative_minibatch_db[i])
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p, a_p)
            an = np.dot(a_n, a_n)
            dist = (self.margin + ap - an)
            _loss = max(dist, 0.0)

            if _loss == 0: #
                correct_pairs += 1
                self.no_residual_list.append(i)
            elif sum(p) == 0 or sum(n) == 0: #or _loss > 1
                self.no_residual_list.append(i)
            elif ap == an or abs(ap-an) < 0.0001:
                self.no_residual_list.append(i)
                eq+=1

            loss += _loss

        loss = (loss / (2 * (bottom[0]).num))
        top[0].data[...] = loss
        top[1].data[...] = correct_pairs


    def backward(self, top, propagate_down, bottom):
        considered_instances = bottom[0].num - len(self.no_residual_list)
        if propagate_down[0]:
            for i in range((bottom[0]).num):

                if not i in self.no_residual_list:
                    x_a = bottom[0].data[i]
                    x_p = bottom[1].data[i]
                    x_n = bottom[2].data[i]

                    # Divided per batch size because Caffe doesn't average by default?
                    # bottom[0].diff[i] = self.a * ((x_n - x_p) / considered_instances)
                    bottom[0].diff[i] = self.a * ((x_n - x_p) / ((bottom[0]).num))
                    bottom[1].diff[i] = self.a * ((x_p - x_a) / ((bottom[1]).num))
                    bottom[2].diff[i] = self.a * ((x_a - x_n) / ((bottom[2]).num))

                else:
                    bottom[0].diff[i] = np.zeros(shape(bottom[0].data)[1])
                    bottom[1].diff[i] = np.zeros(shape(bottom[1].data)[1])
                    bottom[2].diff[i] = np.zeros(shape(bottom[2].data)[1])


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass