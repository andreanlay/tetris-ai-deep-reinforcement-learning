# Tetris AI Deep Q-Learning
Basic applications of Deep Reinforcement Learning for playing Tetris

![](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/demo/Tetris-DQN-Demo.gif)

Full video [link](https://www.youtube.com/watch?v=Sppq_rtr9mg)

### Deep Q Network Architecture
![](https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/master/diagram.png)

### Hand Engineered Features
1. Cleared Lines
2. Bumpiness (Sum of height difference between each column)
3. Holes (Space with block on top of it)
4. Sum of heights

### Prerequisites

* Python 3.8
* Tensorflow
* Keras
* OpenCV
