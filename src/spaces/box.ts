import * as tf from '@tensorflow/tfjs';
import { Space } from './space';

/**
 * Box is the representation of the Cartesian product of n closed intervals
 */
export class Box extends Space {
  /** The lower bound for the value */
  public low: tf.Tensor | number;
  /** The upper bound for the value */
  public high: tf.Tensor | number;

  constructor(
    low: tf.Tensor | number,
    high: tf.Tensor | number,
    shape: number[],
    dtype: tf.DataType
  ) {
    super(shape, dtype);

    if (typeof low !== typeof high) {
      throw new Error('Low and high should be of the same type!');
    }

    if (low instanceof tf.Tensor && high instanceof tf.Tensor) {
      if (
        JSON.stringify(low.shape) !== JSON.stringify(shape) ||
        JSON.stringify(high.shape) !== JSON.stringify(shape)
      ) {
        throw new Error('Low and high should have the same shape as Box!');
      }
    }

    this.low = low;
    this.high = high;
  }

  /**
   * Gets a sample of the box space.
   *
   * @returns a random tensor in the space range
   *
   * @override
   */
  sample(): tf.Tensor {
    if (typeof this.low === 'number' && typeof this.high === 'number') {
      return tf.randomUniform(this.shape, this.low, this.high, this.dtype);
    } else if (
      this.low instanceof tf.Tensor &&
      this.high instanceof tf.Tensor
    ) {
      return tf.tidy(() => {
        let low = this.low as tf.Tensor;
        let high = this.high as tf.Tensor;
        if (high.dtype === 'int32') {
          high = high.add(tf.scalar(1));
        }

        low = low.asType('float32');
        high = high.asType('float32');

        let sample = tf.zeros(this.shape, 'float32');

        let boundedBelow = tf.neg(low).less(tf.scalar(Infinity));
        let boundedAbove = high.less(tf.scalar(Infinity));

        let unbounded = boundedBelow
          .logicalNot()
          .logicalAnd(boundedAbove.logicalNot());
        let boundedBelowOnly = boundedBelow.logicalAnd(
          boundedAbove.logicalNot()
        );
        let boundedAboveOnly = boundedAbove.logicalAnd(
          boundedBelow.logicalNot()
        );
        let bounded = boundedBelow.logicalAnd(boundedAbove);

        sample = tf.where(
          unbounded,
          tf.randomNormal(this.shape, 0, 1, 'float32'),
          sample
        );

        let boundedTensor = tf.randomUniform(this.shape, 0, 1, 'float32');
        boundedTensor.print();
        boundedTensor = boundedTensor.mul(high.sub(low));
        boundedTensor = boundedTensor.add(low);
        sample = tf.where(bounded, boundedTensor, sample);

        let unboundedBelowTensor = randomExponential(this.shape);
        unboundedBelowTensor = unboundedBelowTensor.add(low);
        sample = tf.where(boundedBelowOnly, unboundedBelowTensor, sample);

        let unboundedAboveTensor = randomExponential(this.shape);
        unboundedAboveTensor = tf.neg(unboundedAboveTensor).add(high);
        sample = tf.where(boundedAboveOnly, unboundedAboveTensor, sample);

        return sample.asType(this.dtype);
      });
    } else {
      throw new Error('Low and high should be of the same type!');
    }
  }
}

function randomExponential(shape: number[]) {
  let randomTensor = tf.randomUniform(shape, 0, 1, 'float32');
  randomTensor = tf.neg(tf.log(randomTensor));
  return randomTensor;
}
