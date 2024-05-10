import gymnasium as gym
import cv2
import numpy as np

def preprocess(img, image_hw=84):
    img = img[1:172, :] # MsPacman-specific cropping
    img = cv2.resize(img, dsize=(image_hw, image_hw))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        image_hw=84,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.image_hw = image_hw
    
    def reset(self):
        # Reset the original environment.
        state, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            state, reward, terminated, truncated, info = self.env.step(0)
        
        # Convert the frame `state` to Grayscale and resize it
        state = preprocess(state, image_hw=self.image_hw)

        # The initial observation is simply a copy of the frame `state`
        self.stacked_state = np.tile(state, (self.stack_frames, 1, 1))  # [4, 84, 84]
        
        return self.stacked_state, info
        
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        rewards = 0
        for _ in range(self.skip_frames):
            state, reward, terminated, truncated, info = self.env.step(action)
            rewards += reward
            if terminated or truncated:
                break

        # Convert the frame `state` to Grayscale and resize it
        state = preprocess(state, image_hw=self.image_hw)

        # Push the current frame `state` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], state[np.newaxis]), axis=0)
        
        return self.stacked_state, rewards, terminated, truncated, info