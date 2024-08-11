import pickle
import pandas as pd
import json
import glob
import os
import argparse
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from keras import backend as K

from anomaly_detection.evaluation import tcnae_evaluator
from models.tcnae.tcnae import TCNAE 
from anomaly_detection.experiment import TCNAE_Experiment
from models.algorithm_utils import slide_window
from sklearn.manifold import TSNE

from anomaly_detection import preprocessing as prep
from anomaly_detection.evaluation import lstmae_evaluator
from models.lstmae.lstm_enc_dec_axl import LSTMEDModule
from anomaly_detection.experiment import LSTMAE_Experiment
from anomaly_detection.experiment import train_DGHL

from anomaly_detection.utils_data import load_buildings


# Setting path
PROJECT_DIR = Path.cwd()
exp_id = datetime.now().strftime('%Y%m%d_%H%M')
model_name = "lstmae"
building = "OH12"
Path(PROJECT_DIR, "outputs", rf"{exp_id}_{model_name}_{building}").mkdir(parents=True, exist_ok=True)

# Reading Config file
with open('src/config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

if model_name == "lstmae":

    train_path = PROJECT_DIR / "datasets" /  "train" / "OH12.parquet"
    test_path = PROJECT_DIR / "datasets" /  "test" / "OH12.parquet"
    train, test = prep.lstm_get_data(train_path, test_path)
    
    pred_details = {}
    anomaly_score = {}
    model = LSTMEDModule(n_features = train.shape[1], hidden_size= config['lstmae']['params']['hidden_size'],
              n_layers= (config['lstmae']['params']['n_layer_encoder'], config['lstmae']['params']['n_layer_decoder']),
              seed = config['lstmae']['params']['seed'], gpu = config['lstmae']['params']['gpu'])
    exp = LSTMAE_Experiment(model, sequence_length = config['lstmae']['params']['sequence_length'],
              num_epochs = config['lstmae']['params']['num_epochs'], hidden_size= config['lstmae']['params']['hidden_size'], 
              batch_size = config['lstmae']['params']['batch_size'], gpu = config['lstmae']['params']['gpu'],
              n_layers= (config['lstmae']['params']['n_layer_encoder'], config['lstmae']['params']['n_layer_decoder']))

    exp.fit(train)
    anomaly_score_mean, anomaly_score_max, anomaly_score_mlog, anomaly_score_mah = exp.predict(test.loc[:, test.columns != 'label'])

    pred_details['epoch_loss'] = exp.epoch_loss
    pred_details['prediction_details'] = exp.prediction_details
    anomaly_score['mean'] = anomaly_score_mean
    anomaly_score['max'] = anomaly_score_max
    anomaly_score['logpdf'] = anomaly_score_mlog
    anomaly_score['mahalnobis'] = anomaly_score_mah

    with open(rf"outputs/{exp_id}_{model_name}_{building}/anomaly_score.pkl","wb") as output_file:
        pickle.dump(anomaly_score, output_file)
    with open(rf"outputs/{exp_id}_{model_name}_{building}/pred_details.pkl","wb") as output_file:
        pickle.dump(pred_details, output_file)

    lstmae_evaluator(exp_id, model_name, test, building)
#######################################################################################################
    # change for latent space -add this part start
    
#    latent_s,anomaly_score = exp.predict(test.loc[:, test.columns != 'label'])
#
#    tsne=TSNE(n_components=2)
#    X_tsne=tsne.fit_transform(latent_s)
#    print(X_tsne.shape)
#    x=X_tsne[:,0]
#    y=X_tsne[:,1]
#    fig,ax=plt.subplots()
#    scatter=plt.scatter(x,y, c= test['label'],cmap='viridis')
#    legend=ax.legend(*scatter.legend_elements(),title="Labels")
#    ax.add_artist(legend)
#    plt.savefig("my_figure1.png")
#    plt.show()
##################################################################################################### 

if model_name == "tcnae":
    #Load train dataset
    train = pd.read_parquet('/content/drive/MyDrive/CaseStudy2022-main/datasets/train/Großtagespflege.parquet', engine='pyarrow')
    train = train.drop(['Day','Dayofweek','Hourofday','Normalize_timestamp','Timestamp'],axis = 1)
    train.bfill(inplace = True)
    train.ffill(inplace = True)


    # Load test dataset 
    test = pd.read_parquet('/content/drive/MyDrive/CaseStudy2022-main/datasets/test/Großtagespflege.parquet', engine='pyarrow')
    test = test.drop(['Day','Dayofweek','Hourofday','Normalize_timestamp','Timestamp'],axis = 1)
    test.ffill(inplace = True)
    test.bfill(inplace = True)

    # Find columns containing NaN values
    cols_with_missing = [col for col in test.columns if test[col].isnull().any()]
    print(cols_with_missing)

    # Drop the columns containing NaN values
    train = train.drop(cols_with_missing, axis=1)
    test = test.drop(cols_with_missing, axis=1)
    columns = test.columns.tolist()

    print("model fit")
    pred_details = {} #creating a dictoinary to save all values

    pred_details['time'] = test['Time']
    train = train.set_index('Time')
    test = test.set_index('Time') 

    # train.index = pd.to_datetime(train.index)
    # test.index = pd.to_datetime(test.index)
    length = len(test.columns)

    #sequenciing data
    df_test = slide_window(test, config['tcnae']['params']['sequence_length'], verbose = 1) 
    df_train = slide_window(train, config['tcnae']['params']['sequence_length'], verbose = 1)
    
    


    tcn_mod = TCNAE(ts_dimension = length-1, kernel_size = 20, nb_filters = config['tcnae']['params']['nb_filters'],
                 filters_conv1d = config['tcnae']['params']['filters_conv1d'],
                 latent_sample_rate = config['tcnae']['params']['latent_sample_rate'],
                 sequence_length= config['tcnae']['params']['sequence_length'])


    # model fit
    loss = TCNAE_Experiment.fit(tcn_mod,df_train[...,:length-1],df_train[...,:length-1],batch_size= config['tcnae']['params']['sequence_length'], epochs=config['tcnae']['params']['num_epochs'],validation_steps=1, verbose=1)
    eval_score = TCNAE_Experiment.evals(tcn_mod,df_test[...,:length-1])
    start_time = time.time()
    anomaly_score,anomaly_score_L2,anomaly_score_L1,Err_rec,Err = TCNAE_Experiment.predict(tcn_mod,df_test[...,:length-1])
    
    print("> Time:", round(time.time() - start_time), "seconds.")

   

    latent = TCNAE_Experiment.layers(tcn_mod,'latent',df_test[...,:length-1])
    
    #print test evaluation values keras
    print(eval_score)
    
     # saving required ouput as dictionary

    pred_details['epoch_loss'] = loss
    pred_details['reconstructed_error'] = Err
    pred_details['reconstructed_op'] = Err_rec
    pred_details['anomaly_score_mahanabolis'] = anomaly_score
    pred_details['anomaly_score_L1'] = anomaly_score_L1
    pred_details['anomaly_score_L2'] = anomaly_score_L2
    pred_details['test'] = df_test
    pred_details['latent'] = latent
    pred_details['eval'] = eval_score
    pred_details['label'] = test['label']
    pred_details['col_name'] = columns[1:-1]


  
    
    with open(rf"outputs/{exp_id}_{model_name}_{building}/pred_details_{building}.pkl","wb") as output_file:
        pickle.dump(pred_details, output_file)
    
    #calling evaluator function
    tcnae_evaluator(exp_id, model_name,building)

if model_name == "dghl":
 
    def basic_mc(dataset, random_seed):
        global n_features
        if dataset == 'EF40':
            n_features = 25
        elif dataset == 'EF42':
            n_features = 19
        elif dataset == 'EF40a':
            n_features = 14
        elif dataset == 'Erich_brost':
            n_features = 15
        elif dataset == 'Großtagespflege':
            n_features = 28
        elif dataset == 'HGII':
            n_features = 26
        elif dataset == 'Kita_hokida':
            n_features == 14
        elif dataset == 'OH12':
            n_features = 20
        elif dataset == 'OH14':
            n_features = 20
        elif dataset == 'Chemie':
            n_features = 16
        elif dataset == 'Chemie_concat':
            n_features = 21
        elif dataset == 'Chemie_singleBuilding':
            n_features = 21
        elif dataset == 'Office_concat':
            n_features = 14
        elif dataset == 'Office_singleBuilding':
            n_features = 14

        return n_features


    def run_dghl(args):
      mc = config['dghl']['params']
      mc['normalize_windows'] = False
      mc['z_with_noise'] = False
      mc['z_persistent'] = True
      mc['device'] = None
        
      #Loop buildings
      files = sorted(glob.glob('./data/BuildingDataset/test/*.parquet'))
      for files in files:
        entity = files[:-8].split('/')[-1]
        print(50*'-', entity, 50*'-')
        mc['n_features'] = basic_mc(dataset=entity,random_seed=args.random_seed)

        root_dir = f'./results/{args.experiment_name}_{args.random_seed}/DGHL'
        os.makedirs(name=root_dir, exist_ok=True)

        train_data, train_mask, test_data, test_mask, labels = load_buildings(buildings=[entity],
                                                                              occlusion_intervals=args.occlusion_intervals,
                                                                              occlusion_prob=args.occlusion_prob,
                                                                              root_dir='./data', verbose=True)

        train_DGHL(mc=mc, train_data=train_data, test_data=test_data, test_labels=labels,
                       train_mask=train_mask, test_mask=test_mask, entities=[entity], make_plots=True, root_dir=root_dir)      
    
        
    def main(args):
        print(105*'-')
        print(50*'-',' DGHL ', 50*'-')
        print(105*'-')
        run_dghl(args)
    
    def parse_args():
        desc = "Run DGHL in benchmark datasets"
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('--random_seed', type=int, required=optional)
        parser.add_argument('--occlusion_intervals', type=int, required=optional)
        parser.add_argument('--occlusion_prob', type=float, required=optional)
        parser.add_argument('--experiment_name', type=str, required=optional)
        return parser.parse_args()
   
    
    if __name__ == '__main__':
        args = parse_args()
        if args is None:
            exit()
    
        main(args)

   
# for running dghl run this line python src/run_dghl.py --random_seed 1 --occlusion_intervals 1 --occlusion_prob 0 --experiment_name 'DGHL'
