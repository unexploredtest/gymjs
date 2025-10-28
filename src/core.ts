import * as tf from '@tensorflow/tfjs';

import { Space } from './spaces/space';

export type InfoType<T> = Record<string, T>;

export abstract class Env<T> {
  protected renderMode: string | null;
  public actionSpace: Space;
  public observationSpace: Space;

  constructor(
    actionSpace: Space,
    observationSpace: Space,
    renderMode: string | null
  ) {
    this.actionSpace = actionSpace;
    this.observationSpace = observationSpace;
    this.renderMode = renderMode;
  }

  abstract reset(): [tf.Tensor, InfoType<T> | null];
  abstract step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, InfoType<T> | null]>; // Action is number for now
  abstract render(): Promise<void | tf.Tensor>;
  abstract close(): void;
}
