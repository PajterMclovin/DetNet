""" @PETER HALLDESTAM, 2022

    Small example of how a typical fully connected neural network (FCN) is trained
    on GEANT4 generated data and then evaluated graphically.
"""
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils.models import FCN

from loss_function.loss import LossFunction

from utils.plot_methods import plot_predictions
from utils.plot_methods import plot_loss

from utils.data_preprocess import load_data
from utils.data_preprocess import get_eval_data

from utils.help_methods import save
from utils.help_methods import get_permutation_match
from utils.help_methods import cartesian_to_spherical

## ----------- Load and prepare data for training/evaluation -------------------

NPZ_DATAFILE = os.path.join(os.getcwd(), 'data', 'XB_mixed_data.npz')

TOTAL_PORTION = 1      #portion of file data to be used, (0,1]
EVAL_PORTION = 0.2      #portion of total data for final evalutation (0,1)
VALIDATION_SPLIT = 0.1  #portion of training data for epoch validation

#load simulation data. OBS. labels need to be ordered in decreasing energy!
data, labels = load_data(NPZ_DATAFILE, TOTAL_PORTION)

#detach subset for final evaluation, train and eval are inputs, train_ and eval_ are labels
train, train_, eval, eval_ = get_eval_data(data, labels, eval_portion=EVAL_PORTION)

## ---------------------- Build the neural network -----------------------------

# initiate the network structure
NO_LAYERS = 5
NO_NODES = 124
no_inputs = len(train[0])
no_outputs = len(train_[0])
model = FCN(no_inputs, no_outputs, NO_LAYERS, NO_NODES)

# select mean squared error as loss function
max_mult = int(no_outputs / 3)
loss = LossFunction(max_mult, regression_loss='squared')

#compile the network
LEARNING_RATE = 1e-4    # learning rate/step size
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=loss.get(), metrics=['accuracy'])

## ----------------- Train the neural network and plot results -----------------

#train the network with training data
NO_EPOCHS = 10          # no. times to go through training data
BATCH_SIZE = 2**8       # the training batch size
training = model.fit(train, train_,
                     epochs=NO_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=VALIDATION_SPLIT,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# plot the learning curve
learning_curve = plot_loss(training)

# get predictions on evaluation data
predictions = model.predict(eval)

# return the combination that minimized the loss function (out of max_mult! possible combinations)
predictions, eval_ = get_permutation_match(predictions, eval_, loss, max_mult)

# plot the "lasersvärd" in spherical coordinates (to compare with previous year)
predictions = cartesian_to_spherical(predictions, error=True)
eval_ = cartesian_to_spherical(eval_, error=True)
figure, rec_events = plot_predictions(predictions, eval_, show_detector_angles=True)

# save figures and trained parameters
save('example', figure, learning_curve, model)
