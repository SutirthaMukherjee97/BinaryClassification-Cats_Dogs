## Model

import urllib

pre_trained_model1 =  tf.keras.applications.resnet.ResNet50(include_top=False , weights='imagenet' , input_shape=(224,224,3))

for layers in pre_trained_model1.layers:
  layers.trainable = True

pre_trained_model2 =  tf.keras.applications.vgg19.VGG19(include_top=False , weights='imagenet' , input_shape=(224,224,3))

for layers in pre_trained_model2.layers:
  layers.trainable = True



pre_trained_model3 =  tf.keras.applications.inception_v3.InceptionV3(include_top=False , weights='imagenet' , input_shape=(224,224,3))

for layers in pre_trained_model3.layers:
  layers.trainable = True

pre_trained_model3.summary()

# conv5_block3_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block3_add[0][0]']   
# block5_pool (MaxPooling2D)  (None, 7, 7, 512)       0         
# mixed10 (Concatenate)          (None, 5, 5, 2048)   0

#plot_model(pre_trained_model3 , show_shapes= True , show_layer_names=True, to_file='base-model.png')

def base_Model():

  input = tkl.Input(shape=(224,224,3,));

  L1 = pre_trained_model1 
  L2 = pre_trained_model2
  L3 = pre_trained_model3
  
  L1_OUT = L1(input)
  L2_OUT = L2(input) 
  L3_OUT = L3(input)

  X1 = tkl.GlobalMaxPool2D()(L1_OUT) # 1 x 2048
  XX1 = Dense(256, activation='relu')(X1) # 1 x 256
  XX_BN_1 = tk.layers.BatchNormalization()(XX1)

  X2 = tkl.GlobalMaxPool2D()(L2_OUT) # 1 x 512
  XX2 = Dense(256, activation='relu')(X2) # 1 x 256
  XX_BN_2 = tk.layers.BatchNormalization()(XX2)

  X3 = tkl.GlobalMaxPool2D()(L3_OUT) # 1 x 2048
  XX3 = Dense(256, activation='relu')(X3) # 1 x 256
  XX_BN_3 = tk.layers.BatchNormalization()(XX3)

  # m x 256
  # 256 x m
  ENSEMBLE = tf.reduce_mean([XX_BN_1,XX_BN_2,XX_BN_3],axis=0)
  ENSEMBLE_DENSE_1 = Dense(128, activation='relu')(ENSEMBLE) # 1 x 128
  ENSEMBLE_BN_1 = tk.layers.BatchNormalization()(ENSEMBLE_DENSE_1)
  ENSEMBLE_DENSE_2 = Dense(2, activation='softmax')(ENSEMBLE) # 1 x 2

  model = tk.Model(input,ENSEMBLE_DENSE_2);

  return model;

model = base_Model()

plot_model(model , show_shapes= True , show_layer_names=True, to_file='base-model.png')

### Custom Train

def run_optimizer(model,optimizer,loss_object, x , y_true):

    with tf.GradientTape() as tape:
      logits = model(x)
      L = loss_object(y_true=y_true , y_pred=logits)

    grad = tape.gradient(L , model.trainable_weights)
    optimizer.apply_gradients(grads_and_vars = zip(grad , model.trainable_weights))
    del tape
    return logits , L

def train_data_for_one_epoch(model,optimizer,loss_object, train_data, t_steps_per_epoch=64):
  losses = []
  pbar = tqdm(total=t_steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

  for step in range(1,t_steps_per_epoch+1):
      (x_batch_train, y_batch_train) = next(train_data)
      y_batch_train_1d = tf.argmax(y_batch_train,axis=1);

      logits, loss_value = run_optimizer(model,optimizer,loss_object, x_batch_train, (y_batch_train_1d))
      losses.append(loss_value)
      train_acc_metric(y_batch_train,  (logits))

      pbar.set_description(f"Training loss: {loss_value:.6f} for step: {step}")
      pbar.update()
  return losses

def perform_validation(model,loss_object,test_data, v_steps_per_epoch=64):
  losses = []
  for step in range(v_steps_per_epoch):
      x_batch_test, y_batch_test = next(test_data);
      y_batch_test_1d = tf.argmax(y_batch_test,axis=1);
      val_logits = model(x_batch_test)
      val_loss = loss_object(y_true=y_batch_test_1d, y_pred=(val_logits))
      losses.append(val_loss)
      val_acc_metric(y_batch_test,(val_logits))
  return losses

def train(model, train, test, epochs = 50,t_steps_per_epoch = 64,v_steps_per_epoch = 10, minority_list = [],m = [0.0]):
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []

    history['train_acc'] = []
    history['val_acc'] = []

    val_epoch_loss   = []

    init_lr = 0.0001;
    for epoch in range(epochs):
        lr = init_lr*pow(0.96, epoch);
        print("Learning Rate Set To: "+str(lr))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)        ####################################################### Experimental
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        print('Start of epoch %d' % (epoch,))
        train_losses = train_data_for_one_epoch(model,optimizer, loss_object, train_data=train,t_steps_per_epoch=t_steps_per_epoch)
        train_acc    = train_acc_metric.result()
        history['train_acc'].append(train_acc.numpy())
        train_acc_metric.reset_states()
        val_losses   = perform_validation(model, loss_object , test,v_steps_per_epoch)
        val_acc      = val_acc_metric.result()
        history['val_acc'].append(val_acc.numpy())
        val_acc_metric.reset_states()

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))

        print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f,\
         Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(np.mean(train_losses)), float(np.mean(val_losses)),
                                                            float(train_acc), float(val_acc)))

    history['model'] = model
    return history
