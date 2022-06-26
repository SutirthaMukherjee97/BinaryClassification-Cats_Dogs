## Train

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

history = train(model, train_generator , validation_generator,t_steps_per_epoch=64, v_steps_per_epoch=16, epochs = 5)

def plot_metrics(metric_name, title, ylim=1):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history['train_' + metric_name],color='blue',label='train_' +metric_name)
    plt.plot(history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.show()

plot_metrics(metric_name='loss', title="Loss",ylim=4)
plot_metrics(metric_name='acc', title="Accuracy")

x_test, y_test = (next(validation_generator))

for i in range(20):
 x_test_temp, y_test_temp = (next(validation_generator));
 x_test = np.concatenate((x_test,x_test_temp));
 y_test = np.concatenate((y_test,y_test_temp));

y_test_1d = tf.argmax(y_test,axis=1)

# Prediction on test data 
y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 

#Print confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(2), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')

