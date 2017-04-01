from util.data_utils import crop_images
from util.data_utils import visualize_image

class Pipeline(object):
    """
    Class that defines a complete pipeline to use for the Kaggle fish competition
    """
    def __init__(self, loc, cls, loc_weights=None, cls_weights=None):
        """
        Sets the classifier and localizer. The only requirement for both models is that they should
        implement a fit/predict API. 
        """
        self.classifier = cls
        if cls_weights != None:
            self.classifier.load_weights(cls_weights)
            self.cls_istrained = True
        else:
            self.cls_istrained = False
        
        self.localizer = loc
        if loc_weights != None:
            self.localizer.load_weights(loc_weights)
            self.loc_istrained = True
        else:
            self.loc_istrained = False
        
        self.cls_data = {}
        self.loc_data = {}
        self.box_predictions = {}
        
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
    
    def fit(self, loc_batch_size=10, loc_num_epoch=50, cls_batch_size=16, cls_num_epoch=30):
        """
        Trains both models with data stored in instance attributes
        """
        # Check if data has been set
        if len(self.loc_data) == 0 or len(self.cls_data) == 0:
            print "Please set the training data for the classifier and the localizer"
            raise
        
        # Fit localizer first
        if not self.loc_istrained:
            self.localizer.fit(self.loc_data['X_train'], self.loc_data['y_train'],
                               batch_size=loc_batch_size, nb_epoch=loc_num_epoch, 
                               validation_data=(self.loc_data['X_val'], self.loc_data['y_val']))
        
        # Fit classifier
        if not self.cls_istrained:
            self.classifier.fit(self.cls_data['X_train'], self.cls_data['y_train'],
                                batch_size=cls_batch_size, nb_epoch=cls_num_epoch, 
                                validation_data=(self.cls_data['X_val'], self.cls_data['y_val']))

    def predict(self, X_test, batch_size=10, verbose=1):
        # Predict first boxes for every image
        box_pred = self.localizer.predict(X_test.astype('float32'), batch_size, verbose)
        self.box_predictions['X_test'] = X_test
        self.box_predictions['pred'] = box_pred
        
        # Crop original images based on the bbox predictions
        W, H = self.cls_data['X_train'].shape[1:3]
        cropped_pred = crop_images(X_test, box_pred, W, H)
        
        final_pred = self.classifier.predict(cropped_pred, batch_size, verbose)
        
        return final_pred
    
    def fit_one(self, model="cls", batch_size=10, nb_epoch=5, shuffle=True, verbose=1, call=None):
        if model == "cls":
            self.classifier.fit(self.cls_data['X_train'], self.cls_data['y_train'],
                                batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, callbacks=call,
                                validation_data=(self.cls_data['X_val'], self.cls_data['y_val']))
        if model == "loc":
            self.localizer.fit(self.cls_data['X_train'], self.cls_data['y_train'],
                                batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, callbacks=call,
                                validation_data=(self.cls_data['X_val'], self.cls_data['y_val']))
        
    def save_models(self, loc_name=None, cls_name=None):
        if loc_name != None:
            self.localizer.save_weights('/a/data/fisheries_monitoring/data/models/localizers/%s.h5' % loc_name)
        if cls_name != None:
            self.classifier.save_weights('/a/data/fisheries_monitoring/data/models/%s.h5' % cls_name)
        
    def visualize_predictions(self):
        X_test = self.box_predictions['X_test']
        preds = self.box_predictions['pred']
    
        visualize_grid(X_test, preds)
            
            