# Mountain-Car-DQN
This is a precise, clean, minimal and complete implementation of the DQN algorithm used in the Nature paper. It is demonstrated on Mountain Car.  

The training is shown in Mcar_train_curve, which shows that the trained DQN model has a high variance at testing time. 

I firmly believe something is wrong here. We can improve the high variance issue. This first paper will try to understand the data, regarding data distribution, and backbone samples. 

todo:
1. Get a poor model near the end of training, say this episode point is t0( e.g., 1490). Compare this with the final model(a good one) in some way like minima sharpness. Examine why the immature model performs poorly. 
2. Dump the samples in the buffer at t0 and in the end. Compare these two datasets. 
3. Compare the replayed samples at these two time points. This is related to Tom's question. 
4. 