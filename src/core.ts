import * as tf from '@tensorflow/tfjs';

import { Space } from './spaces/space';

/**
 * An abstract class that represents the structure of an environment.
 */
export abstract class Env {
  /** The render mode of the environment */
  protected renderMode: string | null;
  /** The action space of the environment */
  public actionSpace: Space;
  /** The observation space of the environment */
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

  /**
   * Resets the environment.
   *
   * @returns An array of the observation of the initial state and info
   */
  abstract reset(): [tf.Tensor, Record<string, any> | null];
  /**
   * Takes one step in the environment
   *
   * @param action - action to take in the environment
   * @returns A tuple of the observation of the initial state, reward, termination, truncation and info
   */
  abstract step(
    action: tf.Tensor | number
  ): Promise<[tf.Tensor, number, boolean, boolean, Record<string, any> | null]>;
  /**
   * Renders the environment graphically.
   *
   * @returns Either no return or an array of the screen of the environment
   */
  abstract render(): Promise<void | tf.Tensor>;
  /**
   * Closes the environment.
   */
  abstract close(): void;

  get unwrapped(): Env {
    return this;
  }
}

export abstract class Wrapper {
  env: Env | Wrapper;
  protected _actionSpace: Space | null;
  protected _observationSpace: Space | null;

  constructor(env: Env | Wrapper) {
    this.env = env;
    this._actionSpace = null;
    this._observationSpace = null;
  }

  reset(): [tf.Tensor, Record<string, any> | null] {
    return this.env.reset();
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    return this.env.step(action);
  }

  async render(): Promise<void | tf.Tensor> {
    return this.env.render();
  }

  close(): void {
    this.env.close();
  }

  get actionSpace(): Space {
    if (this._actionSpace === null) {
      return this.env.actionSpace;
    } else {
      return this._actionSpace;
    }
  }

  get observationSpace(): Space {
    if (this._observationSpace === null) {
      return this.env.observationSpace;
    } else {
      return this._observationSpace;
    }
  }

  get unwrapped(): Env | Wrapper {
    return this.env.unwrapped;
  }
}

export abstract class ObservationWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  reset(): [tf.Tensor, Record<string, any> | null] {
    let [obs, info] = this.env.reset();
    return [this.observarionTransform(obs), info];
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [
      this.observarionTransform(obs),
      reward,
      terminated,
      truncated,
      info,
    ];
  }

  abstract observarionTransform(obs: tf.Tensor): tf.Tensor;
}

export abstract class RewardWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, this.rewardTransform(reward), terminated, truncated, info];
  }

  abstract rewardTransform(reward: number): number;
}

export abstract class ActionWrapper extends Wrapper {
  constructor(env: Env | Wrapper) {
    super(env);
  }

  async step(
    action: tf.Tensor | number
  ): Promise<
    [tf.Tensor, number, boolean, boolean, Record<string, any> | null]
  > {
    action = this.actionTransform(action);
    let [obs, reward, terminated, truncated, info] =
      await this.env.step(action);
    return [obs, reward, terminated, truncated, info];
  }

  abstract actionTransform(action: tf.Tensor | number): tf.Tensor | number;
}
