import numpy as np
import torch
import os
import h5py

from skfeature.function.sparse_learning_based.ls_l21 import proximal_gradient_descent
from get_our_dataloader import get_our_datasets
from models import FFLSTMEncoder1,FFLSTMClassifier
from train_fs import train_single_domain_source,train_single_domain_target
import params


#init_random_seed(1)

def one_hot(labels,num_class):
    num_samp = len(labels)
    labels_onehot = np.zeros((num_samp,num_class))
    for samp_id in range(num_samp):
        labels_onehot[samp_id,labels[samp_id]] = 1
    return labels_onehot

def main_env2env_fs(tgt_num_samp_per_class,src_type,sim_type,
                    k,target_env, action_list , lambda_l21=1e-2):

    TGT_NUM_SAMP_PER_CLASS = int(tgt_num_samp_per_class)

    # Dataset information
    SRC_NUM_CLASSES = 5
    TGT_NUM_CLASSES = len(action_list)
    NUM_DIM = 128
    # shots = 1


    # Initialize random seed
    #init_random_seed(1)

    # Load data
    print("=== Data loading ===")
    src_loader,tgt_loader_train,tgt_loader_test = get_our_datasets(target_env, tgt_num_samp_per_class, action_list )

    for data_src,labels_src in src_loader:

        for data_train_tgt,labels_train_tgt in tgt_loader_train:

            for data_test_tgt,labels_test_tgt in tgt_loader_test:

                # model Construction
                print("=== model Construction ===")
                # Initialize encoder_init
                print(">>> Encoder Initialization <<<")
                model_path = os.path.join("model_init", "ours")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                if not os.path.exists( os.path.join(model_path, "encoder_init.pkl")):
                    # Construct encoder_init
                    encoder_init = \
                        FFLSTMEncoder1(lstm_input_size=NUM_DIM,
                                       lstm_hidden_size=params.lstm_hidden_size_ours,
                                       lstm_num_layers=params.lstm_num_layers_ours,
                                       fc2_size=params.fc2_size_ours)
                    # Save encoder_init
                    torch.save(encoder_init, os.path.join(model_path, "encoder_init.pkl"))
                # Initialize classifier_tgt_init
                print(">>> Target Classifier Initialization <<<")
                classifier_tgt_path = os.path.join(model_path,"classifier_tgt_init.pkl")
                if not os.path.exists(classifier_tgt_path):
                    # Construct classifier_tgt_init
                    classifier_tgt_init = \
                        FFLSTMClassifier(fc2_size=params.fc2_size_ours,
                                         num_classes=TGT_NUM_CLASSES)
                    # Save classifier_tgt_init
                    torch.save(classifier_tgt_init,classifier_tgt_path)
                # Initialize classifier_src_init
                print(">>> Source Classifier Initialization <<<")
                if not os.path.exists(os.path.join(model_path, "classifier_src_init.pkl")):
                    # Construct classifier_src_init
                    classifier_src_init = \
                        FFLSTMClassifier(fc2_size=params.fc2_size_ours,
                                         num_classes=SRC_NUM_CLASSES)
                    # Save classifier_src_init
                    torch.save(classifier_src_init,os.path.join(model_path, "classifier_src_init.pkl"))


                # Source Only
                print("=== Source Only ===")
                if (not os.path.exists(os.path.join(model_path,
                                                    "encoder_src_"+src_type+".pkl"))) \
                        or (not os.path.exists(os.path.join(model_path,
                                                            "classifier_src_"+src_type+".pkl"))):
                    # Train encoder_src and classifier_src
                    encoder_init = torch.load(os.path.join(model_path,"encoder_init.pkl") )
                    classifier_src_init = torch.load(os.path.join(model_path,"classifier_src_init.pkl"))
                    if torch.cuda.is_available():
                        encoder_init.cuda()
                        classifier_src_init.cuda()
                        data_src = data_src.cuda()
                        labels_src = labels_src.cuda()
                    encoder_src,classifier_src = \
                        train_single_domain_source(encoder_init,classifier_src_init,
                                                   data_src,labels_src,1e-3)
                    # Save encoder_src and classifier_src
                    torch.save(encoder_src.cpu(),os.path.join(model_path, "encoder_src_"+src_type+".pkl") )
                    torch.save(classifier_src.cpu(), os.path.join(model_path,"classifier_src_"+src_type+".pkl"))


                # Parameter Transfer
                print("=== Parameter Transfer ===")
                accuracy_list = np.zeros((3))
                confusemat_list = np.zeros((TGT_NUM_CLASSES,TGT_NUM_CLASSES,3))
                # Target Only
                print("=== Target Only ===")
                # Train encoder_tgt and classifier_tgt
                encoder_init = torch.load(os.path.join(model_path,"encoder_init.pkl" ))
                classifier_tgt_init = torch.load(os.path.join(model_path,"classifier_tgt_init.pkl") )
                if torch.cuda.is_available():
                    encoder_init.cuda()
                    classifier_tgt_init.cuda()
                    data_train_tgt =  data_train_tgt.cuda()
                    labels_train_tgt =  labels_train_tgt.cuda()
                    data_test_tgt = data_test_tgt.cuda()
                    labels_test_tgt = labels_test_tgt.cuda()

                encoder_tgt,classifier_tgt,accuracy,confusemat = \
                    train_single_domain_target(encoder_init,classifier_tgt_init,
                                               data_train_tgt,labels_train_tgt,
                                               data_test_tgt,labels_test_tgt,
                                               1e-3)
                accuracy_list[0] = accuracy
                confusemat_list[:,:,0] = confusemat


                # Sample Selection
                print("=== Sample Selection ===")
                if sim_type == "l21":
                    our_path = os.path.join("results_"+sim_type + "_" + str(int(np.log10(lambda_l21))),
                                            "ours",
                                            src_type+"num_samp_"+str(tgt_num_samp_per_class) )
         
                else:
                    our_path = os.path.join("results_"+sim_type,
                                            "ours",
                                            src_type+"num_samp_"+str(tgt_num_samp_per_class) )
               
                if not os.path.exists(our_path):
                    os.makedirs(our_path)
                # Source feature extraction
                encoder_src = torch.load( os.path.join(model_path, "encoder_src_"+src_type+".pkl") )
                feat_src = encoder_src(data_src.cpu()).detach().numpy()
                print("Source features are with size "+str(feat_src.shape))
                # Target feature extraction
                feat_train_tgt = encoder_src(data_train_tgt.cpu()).detach().numpy()
                print("Target features are with size "+str(feat_train_tgt.shape))
                if sim_type == "SR":
                    A,_,_ = \
                        proximal_gradient_descent(np.transpose(feat_src),
                                                  np.transpose(feat_train_tgt),
                                                  1e-2)
                    APos = abs(A)
                    NUM_SAMP_SRC = A.shape[0]
                    SRC_NUM_SAMP_PER_CLASS = int(NUM_SAMP_SRC/SRC_NUM_CLASSES)
                    AProbPrime = np.zeros((SRC_NUM_CLASSES,TGT_NUM_CLASSES))
                    for i in range(SRC_NUM_CLASSES):
                        for j in range(TGT_NUM_CLASSES):
                            APosij = APos[i*SRC_NUM_SAMP_PER_CLASS:(i+1)*SRC_NUM_SAMP_PER_CLASS,j*TGT_NUM_SAMP_PER_CLASS:(j+1)*TGT_NUM_SAMP_PER_CLASS]
                            AProbPrime[i,j] = sum(sum(APosij))
                elif sim_type == "NGD":
                    f_ngd = h5py.File('sim_ngd_opp.h5')
                    AProbPrime = f_ngd.get('sim_ngd')[()]
                    print(AProbPrime.shape)
                    print("NGD aprobprime", AProbPrime)
                elif sim_type == "Cos":
                    feat_src_norm = np.zeros(feat_src.shape)
                    for i in range(feat_src.shape[0]):
                        x = np.squeeze(feat_src[i,:])
                        feat_src_norm[i,:] = x/np.linalg.norm(x)
                    feat_train_tgt_norm = np.zeros(feat_train_tgt.shape)
                    for i in range(feat_train_tgt.shape[0]):
                        x = np.squeeze(feat_train_tgt[i,:])
                        feat_train_tgt_norm[i,:] = x/np.linalg.norm(x)
                    A = np.exp(np.dot(feat_src_norm,np.transpose(feat_train_tgt_norm)))
                    NUM_SAMP_SRC = A.shape[0]
                    SRC_NUM_SAMP_PER_CLASS = int(NUM_SAMP_SRC/SRC_NUM_CLASSES)
                    AProbPrime = np.zeros((SRC_NUM_CLASSES,TGT_NUM_CLASSES))
                    for i in range(SRC_NUM_CLASSES):
                        for j in range(TGT_NUM_CLASSES):
                            Aij = A[i*SRC_NUM_SAMP_PER_CLASS:(i+1)*SRC_NUM_SAMP_PER_CLASS,j*TGT_NUM_SAMP_PER_CLASS:(j+1)*TGT_NUM_SAMP_PER_CLASS]
                            AProbPrime[i,j] = sum(sum(Aij))


                # Source Classifier Combination
                # Initialize classifier_tgt_init
                classifier_src = torch.load(os.path.join(model_path,"classifier_src_"+src_type+".pkl") )
                classifier_src_weight = classifier_src.fc.weight.detach().numpy()
                # Combination1
                AProbSoft = np.zeros((SRC_NUM_CLASSES,TGT_NUM_CLASSES))
                for j in range(TGT_NUM_CLASSES):
                    AProbPrimej = np.squeeze(AProbPrime[:,j])
                    AProbSoft[:,j] = AProbPrimej/sum(AProbPrimej)
                if (k+1) % 20 == 0:
                    # Save statistics as file
                    f = h5py.File(our_path+"AProbSoft_"+str(k)+".h5")
                    f.create_dataset("AProbSoft", data=AProbSoft)
                    f.close()
                print(">>> Combined Classifier 1 Initialization <<<")
                classifier_comb1_weight = np.matmul(np.transpose(AProbSoft),
                                                    classifier_src_weight)
                classifier_comb1 = \
                    FFLSTMClassifier(fc2_size=params.fc2_size_ours,
                                     num_classes=TGT_NUM_CLASSES)
                classifier_comb1.fc.weight.data.copy_(torch.tensor(classifier_comb1_weight))
                # Combination2
                AProbHard = np.zeros((SRC_NUM_CLASSES,TGT_NUM_CLASSES))
                for j in range(TGT_NUM_CLASSES):
                    AProbPrimej = np.squeeze(AProbPrime[:,j])
                    AProbHard[np.argmax(AProbPrimej).astype(int),j] = 1.0
                if (k+1) % 20 == 0:
                    # Save statistics as file
                    f = h5py.File(our_path+"AProbHard_"+str(k)+".h5")
                    f.create_dataset("AProbHard", data=AProbHard)
                    f.close()
                print(">>> Combined Classifier 2 Initialization <<<")
                classifier_comb2_weight = np.matmul(np.transpose(AProbHard),
                                                    classifier_src_weight)
                classifier_comb2 = \
                    FFLSTMClassifier(fc2_size=params.fc2_size_ours,
                                     num_classes=TGT_NUM_CLASSES)
                classifier_comb2.fc.weight.data.copy_(torch.tensor(classifier_comb2_weight))


                # Fine Tuning

                # Initialize with encoder_src and classifier_comb1
                print("=== Fine Tuning (Combined Classifier 1) ===")
                encoder_src = torch.load(os.path.join(model_path, "encoder_src_"+src_type+".pkl"))
                print("The size of source classifier is "+str(classifier_comb1.fc.weight.shape))
                if torch.cuda.is_available():
                    encoder_src.cuda()
                    classifier_comb1.cuda()
                encoder_tgt_comb1,classifier_tgt_comb1,accuracy_comb1,confusemat_comb1 = \
                    train_single_domain_target(encoder_src,classifier_comb1,
                                               data_train_tgt,labels_train_tgt,
                                               data_test_tgt,labels_test_tgt,
                                               5e-4)
                accuracy_list[1] = accuracy_comb1
                confusemat_list[:,:,1] = confusemat_comb1
                # Initialize with encoder_src and classifier_comb2
                print("=== Fine Tuning (Combined Classifier 2) ===")
                encoder_src = torch.load(os.path.join(model_path,"encoder_src_"+src_type+".pkl"))
                print("The size of source classifier is "+str(classifier_comb2.fc.weight.shape))
                if torch.cuda.is_available():
                    encoder_src.cuda()
                    classifier_comb2.cuda()
                encoder_tgt_comb2,classifier_tgt_comb2,accuracy_comb2,confusemat_comb2 = \
                    train_single_domain_target(encoder_src,classifier_comb2,
                                               data_train_tgt,labels_train_tgt,
                                               data_test_tgt,labels_test_tgt,
                                               5e-4)
                accuracy_list[2] = accuracy_comb2
                confusemat_list[:,:,2] = confusemat_comb2

                return accuracy_list, confusemat_list