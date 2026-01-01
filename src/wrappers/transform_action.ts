import * as tf from '@tensorflow/tfjs';

import { ActionWrapper } from '../core';
import { Box } from '../spaces';

export class ClipAction<ObsType> extends ActionWrapper<
  tf.Tensor,
  ObsType,
  tf.Tensor
> {
  actionTransform(action: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      if (!(this.actionSpace instanceof Box)) {
        throw new Error('Clip action only works for Box space');
      }

      let newAction = action.clone();

      let low: tf.Tensor;
      let high: tf.Tensor;

      if (
        typeof this.actionSpace.low === 'number' &&
        typeof this.actionSpace.high === 'number'
      ) {
        low = tf.ones(this.actionSpace.shape).mul(this.actionSpace.low);
        high = tf.ones(this.actionSpace.shape).mul(this.actionSpace.high);
      } else if (
        this.actionSpace.low instanceof tf.Tensor &&
        this.actionSpace.high instanceof tf.Tensor
      ) {
        low = this.actionSpace.low;
        high = this.actionSpace.high;
      } else {
        throw new Error('Low and high must be of the same type');
      }

      const lower = action.less(low);
      const higher = high.less(action);

      newAction = tf.where(lower, low, newAction);
      newAction = tf.where(higher, high, newAction);

      return newAction;
    });
  }
}
