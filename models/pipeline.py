from util.data_utils import crop_images

class Pipeline(object):
    """
    Class that defines a complete pipeline to use for the Kaggle fish competition
    """
    def __init__(self, localizer, classifier):
        """
        Sets the classifier and localizer. The only requirement for both models is that they should
        implement a fit/predict API. 
        """
        self.classifier = classifier
        self.localizer = localizer
        
        self.cls_data = {}
        self.loc_data = {}
        
    def set_classifier_train_data(self, X_train, y_train, X_val, y_val):
        self.cls_data['X_train'] = X_train
        self.cls_data['y_train'] = y_train
        self.cls_data['X_val'] = X_val
        self.cls_data['y_val'] = y_val
        
    def set_localizer_train_data(self, X_train, y_train, X_val, y_val):
        self.loc_data['X_train'] = X_train
        self.loc_data['y_train'] = y_train
        self.loc_data['X_val'] = X_val
        self.loc_data['y_val'] = y_val
    
    def fit(self, batch_size=10, nb_epoch=5, shuffle=True, verbose=1, call=None):
        """
        Trains both models with data stored in instance attributes
        """
        # Check if data has been set
        if len(self.loc_data) == 0 or len(self.cls_data) == 0:
            print "Please set the training data for the classifier and the localizer"
            raise
        
        # Fit localizer first
        self.localizer.fit(self.loc_data['X_train'], self.loc_data['y_train'],
                           batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, callbacks=call,
                           validation_data=(self.loc_data['X_val'], self.loc_data['y_val']))
        print "A"
        # Fit classifier
        self.classifier.fit(self.cls_data['X_train'], self.cls_data['y_train'],
                            batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, callbacks=call,
                            validation_data=(self.cls_data['X_val'], self.cls_data['y_val']))
        
    def predict(self, X_test, batch_size=10, verbose=1):
        # Predict first boxes for every image
        box_pred = self.localizer.predict(X_test.astype('float32'), batch_size, verbose)
        
        # Crop original images based on the bbox predictions
        W, H = self.cls_data['X_train'].shape[:2]
        cropped_pred = crop_images(X_test, box_pred, W, H)
        
        final_pred = self.classifier.predict(cropped_pred, batch_size, verbose)
        
        return final_pred
        
        
        
        