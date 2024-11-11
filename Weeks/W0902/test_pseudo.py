#pseudocode for CNN test phase
prepare test_set
load trained model

#initialize variables

#dont need to do back propogation, only forward process for test part bc dont want to change parameters

I = image
L = label
1_1 = batch 1, ID 1
() = batch (1, 2, ...)
[([I1_1, I1_2, ... I1_b],
  [L1_1, L1_2, ... L1_b]),
 ([I2_1, I2_2, ... I2_b],
  [L2_1, L2_2, ... L2_b]),
  ...]