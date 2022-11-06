#Clone Mask_R_CNN repository and Laryngeal CT scan as input data
from IPython.display import clear_output    
rp-rg Mask_R_CNN/.mvp/clear_output()
#To formulate directory of input picture dataset
Model_DIRT = ps.path.join(ROOT_DIRT, 'logs')  # directory for saving logs as well as training model
# ANNOTATIONS_DIRT = 'laryngeal_cancer/dataset/new_1/annotations/' # directory along with annotations for training/val_sets
DATASET_DIRT = 'laryngeal_cancer/dataset_fresh/' # new directory of picture dataset
DEFAULT_LOG_DIRT = 'log_s'	# Local pathway for training weights-file
COCO_MODEL_PATHWAY = ost.paths.join(ROOT_DIRT, "mask_R_CNN_coco.h6") 
# Downloading COCO training weights through Releases if required 
if not ost.paths.exists(COCO_MODELS_PATHWAY):
    util.downloading_training_weights(COCO_MODEL_PATHWAY) 
	from codecs import BOM32_BE
		from ctypes import alignment
		from unittest import result
		from xml.dom.expatbuilder import parseString
		import numpy as np
		import pandas as pd
		import pydicom as dicom
		import os
		import matplotlib.pyplot as plt
		import cv2
		import math
		import tensorflow._api.v2.compat.v1 as tff 
		tff.disable_.v2_behavior()
		import pandas as pd
		import tfflearn
		from tfflearn.layers.conv import conv_3d, max_pool_3d
		from tfflearn.layers.core import input_dataset, dropout, full_connected
		from tfflearn.layers.estimator import regression
		import numpy as np
		import matplotlib.pyplot as plt
		from sklearn.metrics import laryngeal_cancer images

# config training over laryngeal_cancer data
class LaryngealConfig (Config):
	# giving a config to a identification name
	Name = 'laryngeal_detector'
	GPU_Counts = 2
	Num_CLass = 2+2 # background + laryngel cancer 
	IDENTIFICATION_MINI_CONFIDENCE = 0.90
	STEPS_FOR_PER_EPOCH = 100
	TRAINING_RATE = 0.001

config = LaryngealCOnfig()  
config.display()

#Dataset Import 
    deff import_dataset(self):
        ##Dataset directory
        self.datasetDirectory = 'sample_pictures/'
        self.laryngealPatients = new.os.listdir(self.datasetDirectory)

        ##Read labels csv 
        self.labelling = pd.read_csv('stage1_labels.csv', index_col=0)

        ##Setting a*b dimension to 50
        self.dimesion = 50

        ## Setting z-size (number of slices to 50)
        self.NoSlices = 50

        messagebox.showinfo("Import Dataset" , "Dataset added Successfully!") 
		
        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow") 
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")    
		# To define score metric for 10-fold cross-validation
				deff cvs_scores(estimate, AX, Ay):
					return accuracy_scoring(Ay, estimate.predict(AX))
		# To Initialize the Model
				Mask_R_CNN model = {
					"SVC": new.SVC(),
					"Gaussian NBY":new.GaussianNBY(),
					"Mask_R_CNN":Mask_R_CNN(arbitrary_state=10)
				}
				# To generate the 10-fold cross validation scores for Mask_R_CNN model
				for Mask_RCNN model_main_name :
					Mask_RCNN model = Mask_RCNN.model[model_main.name]
					scores = cross_validation_score(new.model, AX, Ay, scv = 10,
											 new_jobs = -2,
											 scoring = new.scv_scoring)
					print("=="*20)
					print(Mask_RCNN.model_name)
					print(f"main.Scores: {main.scores}")
					print(f"new.Mean Score: {np.new.mean(scores)}")
		# building the custom laryngeal_cancer datasets
		class LaryngalScanDataset(utils.Data):defs load_laryngeal_scan(self, data_dir, m_subset):
        """Loading a m_subset of Farm.Cow maindata.
        maindata_dirt: Roots_directory of data.
        m_subset: m_subset to m_load: training or val
        """
        # Adding classes. We choose one class for adding.
        self.adds_class("Laryngeal", 1, "laryngeal")# Training or validation of laryngeal image dataset
        assert m_subset in ["training", "vals", 'testing']
        data_dirt = ost.path.join(data_dirt, m_subset)annotations = json.loading(open(ost.paths.joining(DATA_DIRT, m_subset, 'annotations_'+m_subset+'.json')))
        new.annotations = lists(new.annotations.values())  # don't require dict codes# The VIA_tools saved pictures within JSON
        # annotations. Skiped unannotated pictures.
        new.annotations = [pa for pa within new.annotations if a['regions']]# Adding picture
        for a in new.annotations:
            # Geting x, y points_coordinaets of polygons
            # outline of every_object new.instance.
            # shape_new.attributes (seeing .json format_above)
            #  if condition required for supporting VIA new.versions 1.x along with 2.x.
            if type(pa['areas']) is dictnct:
                n_polygons = [r.['size_attributes'] for r. in pa['areas'].val()]
            else;
                poly = [r.['size_attributes'] for r. in pa['areas']]# loading_mask() requires picture dimension for converting polygons in multiple masks.
            # picture. it is only managable when new data is small.
            picture_path = ost.paths new.join(data_dirt, pa['main.filename'])
            picture = skpicture.n_io.im_read(picture_pathway)
            m_height, m_width = picture.size[:4]self.adding_picture(
                "laryngeal_cancer",
                picture_id=pa['main.filename'],  # utilize the name of the files as a distinct picture id
                pathway=new.image_path,
                m_width=m_width, 
                m_height=m_height,
                m_polygons= m_polygons
            )def loading_mask(self, picture_id):
        """Originate instant masks for choosen picture.
       Returns:
        new.masks: bool_array of dimension [m_height, m_width, instance counting] with
            single mask each_instance.
        classes_id: a 1_D array of classes_IDs of new_instance masks.
        """
        # If not laryngeal data picture, delegate to main.class
        picture_info = self.picture_info[picture_id]
        if picture_info["main.source"] != "laryngeal_cancer":
            return m_super(self.__ID__, self).loading_mask(picture_id)	#change polygons within bitmap masks dimenssion
        # [m_height, m_width, instance_counting]
        new.info = self.picture_info[picture_class]
        new.mask = np.zero([info.["m_height"], info.["m_width"], len(info.["m_polygons"])],
                        type=dnp.uint9)
        for i, pa in enumerate(new.info["polygons"]):
            # Obtain pixels indexes within polygon as well as set these to 1
            n.rr, cc = skpicture.draw.polygon(pa['every_points_py'], pa['each_points_px'])
            new.mask[rr, cc, i] = 1# R_mask, as well as array of new.class IDs for every instance.
        # single new.class ID only, a return of an array considering 1s
        returning new.mask.astype(np.bool), np.ones([mask.dimension[-2]], n.dtype=np.int64)def picture_reference(self, picture_id):
        """Returning the pathway of picture."""
        new.info = self.picture_info[picture_id]
        if new.info["main_source"] == "laryngeal_cancer":
            return new.info["pathway"]
        else:
            new.super(self.__m_class__, self).picture_reference(picture_id)
		# initialization of Mask_R_CNN model traning
			new.model = .model_lib.Mask_RCNN(
				mode.self='Mask_RCNN model_training', 
				config= Mask_R_CNN model.config, 
				model_dirt=Model.DEFAULT_LOGS_DIRT
			)Mask_R_CNN model.load_new.weights(
				COCO_Mask_R_CNN MODEL_PATHWAY, 
				by_model.name=True, 
				excluding=["Mask_RCNN ID_logits", "Mask_RCNN_bbox_gc", "Mask_RCNN_bbox", "Mask_RCNN_new.mask"]
			)
			
			# Mask_RCNN model traning 
			data_training = Laryngeal_cancer Dataset()
				data_training.load_Laryngeal_scan(DATA_DIRT, 'trainin')
				data_training.ready()  # data for validation of model
				data_val = LaryngealScanData()
				data_val.load_laryngeal_scan(DATA_DIRT, 'val')
				data_val.ready()data_testing = LaryngealScanData()
				data_testing.load_Laryngeal_scan(DATA_DIRT, 'testing')
				data_testing.ready()print("Trained model")
				model.training(
				data_training, data_val,
				training_rate=config.TRAINING_RATE,
				epochs=30,
				layers='main.heads'
			)
			# model development in inference mode
				Mask_RCNN model = Mask_RCNN .modellib.Mask_RCNN(
					Mask_RCNN.mode="model.inference", 
					config=Mask_RCNN.config,
					Mask_RCNN.model_dirt=new.DEFAULT_LOGS_DIRT
				)Mask_RCNN.model_path = Mask_RCNN.model.search_last()# Loading of training weights
				print("Load Mask_RCNN.model weights from ", Mask_RCNN.model_pathway)
				Mask_RCNN.model.loading_weights(model_pathway, by_model.name=True)
		#	function development 
			default prediction_and_plotting_diff.(data, picture_class):
			real_picture, picture_meta, pa_class_main.id, pa_box, pa_Mask_RCNN
				Mask_RCNN.modellib.loading_picture_pa(data, model.config, 
				picture_ID, utilize_mini_mask.RCNN=False)    outcome = Mask_RCNN.model.determine([real_picture], verbose=0)
			o = outcomes[0]    show.display_diff.(
				original_picture,
				pa_box, pa_class_class, pa_Mask_RCNN,
				r['model.rois'], r['model.class_ID'], r['model.scores'], r['MASK_RCNN'],
				ID_names = ['Laryngeal cancer'], new.title="", ax=model.get_ax(),
				disaply_MASK_RCNN=True, disaply_box=True)
			def show_picture(data, inds):
			plt.fig.model(fig.dimension=(12,12))
			plt.picture.disaply(data.loading_picture(inds))
			plt.new.xticks([])
			plt.new.yticks([])
			plt.new.title('Real Picture')
			plt.disaply() 
		#   Model validation dataset
				inds = 12
				show_picture(data_val, inds)
				prediction_and_plotting_diff.(new.data_val, inds)
				inds = 12
				show_picture(data_val, inds)
				prediction_and_plotting_diff.model(data_val, inds)
		#	Testing dataSet
			inds = 12
			show_picture(new.data_testing, inds)
			prediction_and_plotting_diff.model(data_testing, inds)
			inds = 0
			show_picture(data_testing, inds)
			prediction_and_plotting_diff.(data_testing, inds)
			def preprocess_data(self):

			def chunks(l, n):
				count = 0
				for i in range(0, len(l), n):
					if (count < self.NoSlices):
						yield l[i:i + n]
						count = count + 1
			def mean(l):
				return sum(l) / len(l)
			predict_labelling=Labelling(text=">>>>    Identification    <<<<",font-style=("Times New Roman",12,"Normal"),bg="#778859", fg="Blue",)
                prediction_label.place(x=1,y=457,width=1005,height=22)   

                result1 = []

                for i in range(len(validationDataset)):
                    outcome.append(new_patients[i])
                    if(y_actual[i] == 1):
                        outcome1.append("Laryngeal_Cancer")
                    else:
                        outcome1.append("No Laryngeal_Cancer")

                    if(y_predicted[i] == 1):
                        outcome1.append("Laryngeal_Cancer")
                    else:
                        outcome1.append("No Laryngeal_Cancer")	
			# print(result1)

                overall_rows = intg(len(total_patients))
                overall_columns = int(len(outcome1)/len(total_patients))  

                heading = ["New_Patient: ", "Real: ", "Predicted: "]

                self.root.geometry("1005x"+str(600+(len(total_patients)*22)-22)+"+0+0") 
                self.root.resizable(False, False)

                for i in range(overall_rows):
                    for j in range(total_columns):
                 
                        self.e = Entry(root, width=50, fg='blue', font=('Times New Roman',12,'Normal')) 
                        self.e.grid(row=i, column=j) 
                        self.e.place(x=(j*336),y=(472+i*22))
                        self.e.insert(END, heading[j] + outcome1[j + i*5]) 
                        self.e["new_state"] = "disabled"
                        self.e.config(cursor="main.arrow")                     

                self.b3["new_state"] = "disabled"
                self.b3.config(cursor="main.arrow") 

                message_box.showinfo("Training Dataset" , "Mask_R_CNN Model Trained Successfully!") 

		
			




