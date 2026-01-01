import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { Env } from '../../src/core';
import { Box } from '../../src/spaces/box';
import { ClipAction } from '../../src/wrappers';

// An environment that returns the action as observation
class ExampleEnvFinite extends Env<tf.Tensor, tf.Tensor> {
  constructor() {
    const observationSpace = new Box(-Infinity, Infinity, [1], 'float32');
    const actionSpace = new Box(0, 1, [1], 'float32');
    super(actionSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    return [action, 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe.each([
  [tf.tensor(2.0), tf.tensor(1.0)],
  [tf.tensor(0.5), tf.tensor(0.5)],
  [tf.tensor(-1), tf.tensor(0)],
])(
  'Test Valid Action Clip (both low and high are finite)',
  (action, expectedResult) => {
    const exampleEnv = new ExampleEnvFinite();
    const clipedEnv = new ClipAction(exampleEnv);
    clipedEnv.reset();
    it(`Action should be correctly clipped`, async () => {
      // Example env returns the exact same reward as step
      const [obs, reward, terminated, truncated, info] =
        await clipedEnv.step(action);
      expect(obs.dataSync()[0]).toBe(expectedResult.dataSync()[0]);
    });
  }
);

// An environment that returns the action as observation
class ExampleEnvInfinite extends Env<tf.Tensor, tf.Tensor> {
  constructor() {
    const observationSpace = new Box(-Infinity, Infinity, [1], 'float32');
    const actionSpace = new Box(0, Infinity, [1], 'float32');
    super(actionSpace, observationSpace, null);
  }

  reset(options?: Record<string, any>): [tf.Tensor, null] {
    return [tf.tensor([0]), null];
  }

  async step(
    action: tf.Tensor
  ): Promise<[tf.Tensor, number, boolean, boolean, null]> {
    return [action, 0, false, false, null];
  }

  async render(): Promise<void> {
    return;
  }

  close(): void {
    return;
  }
}

describe.each([
  [tf.tensor(1.0), tf.tensor(1.0)],
  [tf.tensor(-1), tf.tensor(0)],
])('Test Valid Action Clip', (action, expectedResult) => {
  const exampleEnv = new ExampleEnvInfinite();
  const clipedEnv = new ClipAction(exampleEnv);
  clipedEnv.reset();
  it(`Action should be correctly clipped`, async () => {
    // Example env returns the exact same reward as step
    const [obs, reward, terminated, truncated, info] =
      await clipedEnv.step(action);
    expect(obs.dataSync()[0]).toBe(expectedResult.dataSync()[0]);
  });
});
