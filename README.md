# DQN_Cartpole
Deep Q Network Solution to the Cartpole Problem written in Jupyter Notebook. Thoroughly commented for anyone relatively new to machine learning.

# The Neural Network Model
self.model = Sequential()  
self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))  
self.model.add(Dense(24, activation="relu"))  
self.model.add(Dense(self.action_space, activation="linear"))  
self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

# Cartpole in action
![DQN_Cartpole_Screenshot_Environment](https://user-images.githubusercontent.com/88697660/218208279-93d4f6d8-df5e-4d56-a50c-3282b54ac4c4.png)

# Terminal output for first few episodes
![DQN_Cartpole_Screenshot_Output1](https://user-images.githubusercontent.com/88697660/218207359-406941db-5776-4824-aa11-52bcd6bdf55d.png)

# Cartpole solved!
![DQN_Cartpole_Screenshot_Output2](https://user-images.githubusercontent.com/88697660/218207369-f3620cfb-338e-4b45-a688-739b4519eeaa.png)
