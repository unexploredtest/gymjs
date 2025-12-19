import { test, expect, beforeEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';

import { CartPoleEnv, PendulumEnv } from '../../src/envs/classic_control';
import { Env } from '../../src/core';

describe.each([
  [CartPoleEnv, 'CartPoleEnv'],
  [PendulumEnv, 'PendulumEnv'],
])('Testing Environment $name', (env, name) => {
  it(`${name} Should construct successfully`, () => {
    expect(() => new env()).not.toThrowError();
  });
});
