from tensorflow import keras 
from keras.models import Model 
from keras import Input 
from keras.layers import Dense, Lambda 
from tensorflow.keras.utils import plot_model 
from keras import backend as K 
from utils import sampling
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
import graphviz 
import plotly
import plotly.express as px 

import sys
import os

main_dir=os.path.dirname(sys.path[0])

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(X_train[n])
        axs[i,j].set(title=y_train[n])
        n=n+1
plt.show() 

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

original_dim = 784 
latent_dim = 2 

visible = keras.Input(shape=(original_dim,), name='Encoder-Input-Layer')

h_enc1 = Dense(units=64, activation='relu', name='Encoder-Hidden-Layer-1')(visible)
h_enc2 = Dense(units=32, activation='relu', name='Encoder-Hidden-Layer-2')(h_enc1)
h_enc3 = Dense(units=16, activation='relu', name='Encoder-Hidden-Layer-3')(h_enc2)

z_mean = Dense(units=latent_dim, name='Z-Mean')(h_enc3) # Mean component
z_log_sigma = Dense(units=latent_dim, name='Z-Log-Sigma')(h_enc3) # Standard deviation component
z = Lambda(sampling, name='Z-Sampling-Layer')([z_mean, z_log_sigma])

encoder = Model(visible, [z_mean, z_log_sigma, z], name='Encoder-Model')

#plot_model(encoder, show_shapes=True, dpi=300)

latent_inputs = Input(shape=(latent_dim,), name='Input-Z-Sampling')

h_dec = Dense(units=16, activation='relu', name='Decoder-Hidden-Layer-1')(latent_inputs)
h_dec2 = Dense(units=32, activation='relu', name='Decoder-Hidden-Layer-2')(h_dec)
h_dec3 = Dense(units=64, activation='relu', name='Decoder-Hidden-Layer-3')(h_dec2)

outputs = Dense(original_dim, activation='sigmoid', name='Decoder-Output-Layer')(h_dec3)

decoder = Model(latent_inputs, outputs, name='Decoder-Model')

#plot_model(decoder, show_shapes=True, dpi=300)


outpt = decoder(encoder(visible)[2]) 
vae = Model(inputs=visible, outputs=outpt, name='VAE-MNIST')

r_loss = original_dim * keras.losses.mse(visible, outpt)

kl_loss =  -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)

vae_loss = K.mean(r_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

history = vae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.xticks(ticks=np.arange(len(history.history['loss']), step=1), labels=np.arange(1, len(history.history['loss'])+1, step=1))
plt.legend()
plt.show()

X_test_encoded = encoder.predict(X_test)

fig = px.scatter(None, x=X_test_encoded[2][:,0], y=X_test_encoded[2][:,1], 
                 opacity=1, color=y_test.astype(str))

fig.update_layout(dict(plot_bgcolor = 'white'))

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white',
                 title_font=dict(size=10), tickfont=dict(size=10))

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white',
                 title_font=dict(size=10), tickfont=dict(size=10))

fig.update_layout(title_text="MNIST digit representation in the 2D Latent Space")

fig.update_traces(marker=dict(size=2))

fig.show()

z_sample_digit=[[0,2.5]]

digit_decoded = decoder.predict(z_sample_digit)

plt.matshow(digit_decoded.reshape(28,28))
plt.show()

n = 30 
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

grid_x = np.linspace(1.5, -1.5, n)
grid_y = np.linspace(-1.5, 1.5, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)

        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(18, 16))
plt.imshow(figure)
plt.show()

z_sample_digit=[[0,2.5]]

digit_decoded = decoder.predict(z_sample_digit)

plt.matshow(digit_decoded.reshape(28,28))
plt.show()

z_sample_digit=[[0,0.4]]

digit_decoded = decoder.predict(z_sample_digit)

plt.matshow(digit_decoded.reshape(28,28))
plt.show()