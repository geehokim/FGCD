import time

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import umap.umap_ as umap
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from torch.utils.data import DataLoader

from utils import *
from utils.loss import MetricLoss

#from .cka import CKACalculator


__all__ = ['imshow', 'log_acc', 'log_ConfusionMatrix_Umap', 'get_activation', 'calculate_delta_cv','calculate_delta_variance',
           'calculate_divergence_from_optimal','calculate_divergence_from_center','calculate_cosinesimilarity_from_optimal',
           'calculate_cosinesimilarity_from_center','log_models_Umap','cka_visualize','AverageMeter','__log_test_metric__',
           '__log_local_sim__','metriceval','cka_allmodels','umap_allmodels', 'log_fisher_diag']
#filedir = './log_imgs'
filedir = './log_imgs/'

def log_fisher_diag(args, model, trainset, testloader, device, this_lr, wandb_dict):
    start = time.time()
    test_model = copy.deepcopy(model)
    optimizer = optim.SGD(test_model.parameters(), lr=this_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    fisher = compute_fisher_matrix_diag(test_model, optimizer, testloader, device=device, sampling_type='true')
    for key in fisher.keys():
        wandb_dict["Test_Fisher_" + key + '_mean'] = fisher[key].mean()
        wandb_dict["Test_Fisher_" + key + '_std'] = fisher[key].std()
    global_train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_model = copy.deepcopy(model)
    optimizer = optim.SGD(test_model.parameters(), lr=this_lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    fisher = compute_fisher_matrix_diag(test_model, optimizer, global_train_loader, device=device, sampling_type='true')
    for key in fisher.keys():
        wandb_dict["G_Train_Fisher_" + key + '_mean'] = fisher[key].mean()
        wandb_dict["G_Train_Fisher_" + key + '_std'] = fisher[key].std()
    print("Time for Logging Fisher scores : ", time.time() - start)

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg)
    print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(np.transpose(npimg))#, (1, 2, 0)))
    plt.show()


def log_acc(model, testloader, args, wandb_dict, name):
    model.eval()
    device = next(model.parameters()).device
    first = True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4', activation))
            images, labels = data[0].to(device), data[1].to(device)
            if 'byol' in args.method:
                _ ,outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features = activation['layer4'].view(len(images), -1)
                saved_labels = labels
                saved_pred = predicted
                first = False
            else:
                features = torch.cat((features, activation['layer4'].view(len(images), -1)))
                saved_labels = torch.cat((saved_labels, labels))
                saved_pred = torch.cat((saved_pred, predicted))

        saved_labels = saved_labels.cpu()
        saved_pred = saved_pred.cpu()

        f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
        acc = metrics.accuracy_score(saved_labels, saved_pred)
        wandb_dict[name + " f1"] = f1
        wandb_dict[name + " acc"] = acc

    model.train()
    return acc


def log_ConfusionMatrix_Umap(model, testloader, args, wandb_dict, name):
    if args.set == 'CIFAR10':
        classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.set == 'MNIST':
        classes=['0','1','2','3','4','5','6','7','8','9']
    elif args.set == 'CIFAR100':
        classes= testloader.dataset.classes
    else:
        raise Exception("Not valid args.set")   
    
    
    
    
    model.eval()
    device = next(model.parameters()).device
    first = True
    with torch.no_grad():
        for data in testloader:
            activation = {}
            model.layer4.register_forward_hook(get_activation('layer4', activation))
            images, labels = data[0].to(device), data[1].to(device)
            if 'byol' in args.method or 'simsiam' in args.method:
                _, outputs = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if first:
                features = activation['layer4'].view(len(images), -1)
                saved_labels = labels
                saved_pred = predicted
                first = False
            else:
                features = torch.cat((features, activation['layer4'].view(len(images), -1)))
                saved_labels = torch.cat((saved_labels, labels))
                saved_pred = torch.cat((saved_pred, predicted))

        saved_labels = saved_labels.cpu()
        saved_pred = saved_pred.cpu()

        # plt.figure()
        f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
        acc = metrics.accuracy_score(saved_labels, saved_pred)
        cm = metrics.confusion_matrix(saved_labels, saved_pred)
        wandb_dict[name + " f1"] = f1
        wandb_dict[name + " acc"] = acc
        plt.figure(figsize=(20, 20))
        # wandb_dict[args.mode+name+" f1"]=f1
        # wandb_dict[args.mode+name+" acc"]=acc
        fig, ax = plot_confusion_matrix(cm, class_names=classes,
                                        colorbar=True,
                                        show_absolute=False,
                                        show_normed=True,
                                        figsize=(16, 16)
                                        )
        ax.margins(2, 2)

        wandb_dict[name + " confusion_matrix"] = wandb.Image(fig)
        plt.close()
        y_test = np.asarray(saved_labels.cpu())

        reducer = umap.UMAP(random_state=0, n_components=args.umap_dim, min_dist=0.5, n_neighbors=3)
        embedding = reducer.fit_transform(features.cpu())
        
        
        ##################### plot ground truth #######################
        plt.figure(figsize=(20, 20))

        if args.umap_dim == 3:
            ax = plt.axes(projection=('3d'))
        else:
            ax = plt.axes()

        for i in range(len(classes)):
            y_i = (y_test == i)
            scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
            ax.scatter(*scatter_input, label=classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()

        wandb_dict[name + " umap"] = wandb.Image(plt)
        plt.close()
        
        
        
        ############### plot model predicted class ###########################
        plt.figure(figsize=(20, 20))

        if args.umap_dim == 3:
            ax = plt.axes(projection=('3d'))
        else:
            ax = plt.axes()

        for i in range(len(classes)):
            y_i =(np.asarray(saved_pred.cpu()) == i)
            scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
            ax.scatter(*scatter_input, label=classes[i])
        plt.legend(loc=4)
        plt.gca().invert_yaxis()

        wandb_dict[name + " umap_model predicted class"] = wandb.Image(plt)
        plt.close()        
        
        
    model.train()
    return acc


def get_feature_specifiedclass(features, labels,  classidx):
    result = features[labels ==classidx]
    return result


def concat_all(x):    
    for idx, el in enumerate(x):
        if idx==0:
            result = el
        else:
            result = torch.cat((result, el))
    return result
        

def divide_features_classwise(features, labels, num_of_sample_per_class, draw_classes = 10):
    sorted_label_class = []
    feature_class = []
    
    for i in range(draw_classes):
        
        this_class_feature = get_feature_specifiedclass(features,labels.cpu(),i)
        max_num = max(len(this_class_feature),num_of_sample_per_class)
        this_class_feature =this_class_feature[:max_num]
        feature_class.append(this_class_feature)
        sorted_label_class.append(torch.ones([max_num])*i)
        #breakpoint()
        
    return feature_class, sorted_label_class



def log_models_Umap(model, models_dict_list, testloader, args, names_list = None, num_of_sample_per_class = 100, draw_classes = None, drawing_options = None, feat_lev = 4):
    if args.set == 'CIFAR10':
        classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.set == 'MNIST':
        classes=['0','1','2','3','4','5','6','7','8','9']
    elif args.set == 'CIFAR100':
        classes= testloader.dataset.classes
    else:
        raise Exception("Not valid args.set")      
    
    assert(num_of_sample_per_class <= float(len(testloader.dataset)/len(classes)) )
    #color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']* 100
    # NUM_COLORS = min(len(classes),draw_classes)#100
    # cm = plt.get_cmap('gist_rainbow')
    # color_cycle =[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    color_cycle  = plt.cm.get_cmap('tab20').colors[:100]
    marker_list = ['o','P','X','^']* 100
    #marker_list = ['o','o','o','o']
    opacity_max = 1
    opacity_min = 0.2
    #opacity_list = [1, 0.8, 0.6, 0.4]
    wandb_dict = {}

    if draw_classes == None:
        draw_classes = len(classes)
    else:
        draw_classes = min(draw_classes,len(classes))
    

    if drawing_options == None:
        drawing_options = [[True for model in models_dict_list]]

    saved_features = []
    saved_preds =[]
    for model_dict,name in zip(models_dict_list,names_list):
        model.load_state_dict(model_dict)
        model.eval()
        device = next(model.parameters()).device
        first = True
        with torch.no_grad():
            for data in testloader:
                # activation = {}
                # model.layer4.register_forward_hook(get_activation('layer4', activation))
                images, labels = data[0].to(device), data[1].to(device)
                all_out = model(images, return_feature = True)
                outputs = all_out[-1]

                _, predicted = torch.max(outputs.data, 1)
                if feat_lev != 4:
                    this_feat = F.adaptive_avg_pool2d(all_out[feat_lev],1)
                else:
                    this_feat = all_out[feat_lev]
                if first:
                    features = this_feat.view(len(images),-1)
                    saved_labels = labels
                    saved_pred = predicted
                    first = False
                else:
                    features = torch.cat((features, this_feat.view(len(images),-1)   ))
                    saved_labels = torch.cat((saved_labels, labels))
                    saved_pred = torch.cat((saved_pred, predicted))

            saved_labels = saved_labels.cpu()
            saved_pred = saved_pred.cpu()

            #breakpoint()
            f1 = metrics.f1_score(saved_labels, saved_pred, average='weighted')
            acc = metrics.accuracy_score(saved_labels, saved_pred)
            #print(len(labels), len(saved_labels))
            #cm = metrics.confusion_matrix(saved_labels, saved_pred)
            wandb_dict[name + " f1"] = f1
            wandb_dict[name + " acc"] = 100 * acc
            #plt.figure(figsize=(20, 20))
            # wandb_dict[args.mode+name+" f1"]=f1
            # wandb_dict[args.mode+name+" acc"]=acc
            # fig, ax = plot_confusion_matrix(cm, class_names=classes,
            #                                 colorbar=True,
            #                                 show_absolute=False,
            #                                 show_normed=True,
            #                                 figsize=(32, 32)
            #                                 )
            #ax.margins(2, 2)
            
            # wandb_dict[name + "_confusion_matrix"] = wandb.plot.confusion_matrix(probs=None,
            #             y_true=[l.item() for l in saved_labels], preds=[p.item() for p in saved_pred],
            #             class_names=classes)
#wandb.Image(fig)
            #fig.savefig(name + "_confusion_matrix")
            #fig.close()
            #breakpoint()
            #plt.savefig(name + " confusion_matrix")
            
            draw_critic = (saved_labels< draw_classes)
            saved_labels = saved_labels[draw_critic]
            saved_pred = saved_pred[draw_critic]
            saved_preds.append(saved_pred)
            features = features[draw_critic]
            sorted_feature, sorted_label = divide_features_classwise(features, saved_labels.cpu(), num_of_sample_per_class = num_of_sample_per_class, draw_classes = draw_classes)
            sorted_feature = concat_all(sorted_feature)
            saved_features.append(sorted_feature)

            sorted_label = concat_all(sorted_label)

    
    y_test = np.asarray(sorted_label)
    all_feature = concat_all(saved_features)
    # for idx,a in enumerate(saved_features):
    #     print(idx, len(a))
    reducer = umap.UMAP(random_state=0, n_components=args.umap_dim, metric='cosine')
    embedding = reducer.fit_transform(all_feature.cpu())
    embedding_seperate_model = [embedding[len(sorted_label)*j:len(sorted_label) *(j+1)] for j in range(len(models_dict_list))]
        




        

    ##################### plot ground truth #######################

    for drawing_option in drawing_options:
        all_names = "umap"
        for model_option, name in zip(drawing_option,names_list):
            if model_option:
                all_names += "_" + str(name)

        plt.figure(figsize=(10, 10))

        # if args.umap_dim == 3:
        #     ax = plt.axes(projection=('3d'))
        # else:
        #     ax = plt.axes()
        this_draw_num = float(sum(drawing_option))
        this_opacity_gap = (opacity_max-opacity_min)/max((this_draw_num - 1),1)
        
        for i in range(draw_classes):
            first = True
            y_i = (y_test == i)
            count = -1
            for j in range(len(drawing_option)):
                #breakpoint()
                if drawing_option[j]:
                    try:
                        count += 1
                        this_embedding = embedding_seperate_model[j]
                        #scatter_input = [this_embedding[y_i, k] for k in range(args.umap_dim)]
                        plt.scatter(this_embedding[y_i, 0],this_embedding[y_i, 1], color = color_cycle[i], marker =marker_list[count], alpha = opacity_max - this_opacity_gap*count) #, label=classes[i] if first else None
                        plt.xticks([])  # Remove x-axis ticks
                        plt.yticks([])  # Remove y-axis ticks 
                        first = False
                    except:
                        breakpoint()
        #plt.legend(loc=4)
        plt.legend(loc=4)
        plt.gca().invert_yaxis()
        #plt.show()
        this_name = all_names + "_truelabels_class" + str(draw_classes) + "feat" + str(feat_lev)
        #plt.savefig(filedir + args.set + args.mode+args.additional_experiment_name+this_name)
        #breakpoint()
        wandb_dict[this_name] = wandb.Image(plt)#filedir +args.set + args.mode + args.additional_experiment_name+this_name+'.png')
        plt.close()
  
        
        
        
        ############### plot model predicted class ###########################
        # plt.figure(figsize=(20, 20))

        # if args.umap_dim == 3:
        #     ax = plt.axes(projection=('3d'))
        # else:
        #     ax = plt.axes()

        # for i in range(draw_classes):
        #     y_i =(np.asarray(saved_pred.cpu()) == i)
        #     scatter_input = [embedding[y_i, k] for k in range(args.umap_dim)]
        #     ax.scatter(*scatter_input, label=classes[i])
        # plt.legend(loc=4)
        # plt.gca().invert_yaxis()

        # wandb_dict[all_names + "model predicted class"] = wandb.Image(plt)
        # plt.close()        
        
    # for model in models_dict_list:  
    #     model.train()
    return wandb_dict


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def calculate_delta_cv(args, model, local_delta, num_of_data_clients):
    total_num_of_data_clients = sum(num_of_data_clients)
    global_delta = copy.deepcopy(local_delta[0])
    variance = 0
    total_parameters = sum(p.numel() for p in model.parameters())
    for key in global_delta.keys():
        for i in range(len(local_delta)):
            if i == 0:
                global_delta[key] *= num_of_data_clients[i]
            else:
                global_delta[key] += local_delta[i][key] * num_of_data_clients[i]

        global_delta[key] = global_delta[key] /  (total_num_of_data_clients)
        for i in range(len(local_delta)):
            if i==0:
                this_variance = (((local_delta[i][key] - global_delta[key])**2) / (global_delta[key]*total_parameters  + 1e-10)**2)
            #variance += ((((local_delta[i][key] - global_delta[key])**2) / global_delta[key]**2) ** 0.5).sum()
            else:
                this_variance += (((local_delta[i][key] - global_delta[key])**2) / (global_delta[key]*total_parameters + 1e-10)**2)
        variance += (this_variance**0.5).sum()
    return variance #/ total_num_of_data_clients



def calculate_delta_variance(args, local_delta, num_of_data_clients):
    total_num_of_data_clients = sum(num_of_data_clients)
    global_delta = copy.deepcopy(local_delta[0])
    variance = 0
    for key in global_delta.keys():
        for i in range(len(local_delta)):
            if i == 0:
                global_delta[key] *= num_of_data_clients[i]
            else:
                global_delta[key] += local_delta[i][key] * num_of_data_clients[i]

        global_delta[key] = global_delta[key] /  (total_num_of_data_clients)
        for i in range(len(local_delta)):
            if i==0:
                this_variance = (((local_delta[i][key] - global_delta[key])**2))
            #variance += ((((local_delta[i][key] - global_delta[key])**2) / global_delta[key]**2) ** 0.5).sum()
            else:
                this_variance += (((local_delta[i][key] - global_delta[key])**2))
        variance += (this_variance**0.5).sum()
    return variance #/ total_num_of_data_clients


def calculate_divergence_from_optimal(args, checkpoint_path, agg_model_weight):

    optimal_model_weight = torch.load(checkpoint_path)['model_state_dict']

    divergence = 0
    denom = 0
    for key in agg_model_weight.keys():
        divergence += ((optimal_model_weight[key] - agg_model_weight[key])**2).sum()
        denom += ((optimal_model_weight[key]) ** 2).sum()

    divergence = divergence / denom
    return divergence


def calculate_divergence_from_center(args, optimal_model_weight, agg_model_weight):


    divergence = 0
    denom = 0
    for key in agg_model_weight.keys():
        divergence += ((optimal_model_weight[key] - agg_model_weight[key])**2).sum()
        denom += ((optimal_model_weight[key]) ** 2).sum()

    divergence = divergence / denom
    return divergence


def calculate_cosinesimilarity_from_optimal(args, checkpoint_path, current_model_weight, prev_model_weight):

    optimal_model_weight = torch.load(checkpoint_path)['model_state_dict']

    a_dot_b = 0
    a_norm = 0
    b_norm = 0
    for key in optimal_model_weight.keys():
        a= (optimal_model_weight[key] - prev_model_weight[key])
        b= (current_model_weight[key] - prev_model_weight[key])
        a_dot_b += (a*b).sum()
        a_norm += (a*a).sum()
        b_norm += (b*b).sum()

    cosinesimilarity = a_dot_b / (((a_norm)**0.5)*((b_norm)**0.5))
    return cosinesimilarity


def calculate_cosinesimilarity_from_center(args, optimal_model_weight, current_model_weight, prev_model_weight):


    a_dot_b = 0
    a_norm = 0
    b_norm = 0
    for key in optimal_model_weight.keys():
        a= (optimal_model_weight[key] - prev_model_weight[key])
        b= (current_model_weight[key] - prev_model_weight[key])
        a_dot_b += (a*b).sum()
        a_norm += (a*a).sum()
        b_norm += (b*b).sum()

    cosinesimilarity = a_dot_b / (((a_norm)**0.5)*((b_norm)**0.5))
    return cosinesimilarity


def select_2(num:int = 0):
    assert(num>=2)
    result = []
    used_only_one = []
    combi = int(num*(num-1)/2)
    one_el_true_num = num - 1
    used_never = list(range(combi))
    for i in range(combi):
        result.append([False] * num)
        
    for j in range(num):
        
        for n in used_only_one:
            use_second = n.pop(0)

            result[use_second][j] = True
        this_use_first = []
        for o in range(one_el_true_num - len(used_only_one)):
            use_first = used_never.pop(0)
            this_use_first.append(use_first)
            result[use_first][j] = True
        used_only_one.append(this_use_first)

        
        
    return result

def cka_visualize(model, models_dict_list, testloader, args,names_list = None, epoch = None):
    wandb_dict = {}
    from utils.cka import CKACalculator
    testloader_shuffle = torch.utils.data.DataLoader(testloader.dataset, batch_size=testloader.batch_size,shuffle=True, num_workers=args.workers)
    combi_2_cases = select_2(len(names_list))
    

    for idx in range(len(combi_2_cases) + len(models_dict_list)):
        if idx < len(combi_2_cases):
            this_case = combi_2_cases[idx]
            this_name = []
            this_model = []
            for idx,el in enumerate(this_case):
                if el:
                    this_name.append(names_list[idx])
                    this_model.append(models_dict_list[idx])
            model1 = copy.deepcopy(model)
            model1.load_state_dict(this_model[0])
            model2 = copy.deepcopy(model)
            model2.load_state_dict(this_model[1])
            draw_CKA_diagonal = True
        else:
            model_index = idx - len(combi_2_cases)
            this_name[0] = names_list[model_index]
            this_name[1] = names_list[model_index]
            model1 = copy.deepcopy(model)
            model1.load_state_dict(models_dict_list[model_index])
            model2 = copy.deepcopy(model)
            model2.load_state_dict(models_dict_list[model_index])       
            draw_CKA_diagonal = False   
        calculator = CKACalculator(model1=model1, model2=model2, dataloader=testloader_shuffle, num_epochs = args.CKA_epochs)
        cka_output = calculator.calculate_cka_matrix()
        #draw CKA diagonal
        if draw_CKA_diagonal:
            
            diag_fig_name = this_name[0] + "_" + this_name[1] + "_CKA_diag"
            print(diag_fig_name)
            
            diag = (torch.diagonal(cka_output)) 
            x, y = np.arange(len(diag)),diag.cpu().numpy()


            # fig = plt.figure(figsize=(12, 12))
            # plt.plot(x,y)
            # for xx,yy in zip(x,y):
            #     plt.text(xx,yy,str(yy),fontsize = 5)
            # wandb_dict[diag_fig_name] = fig
            #breakpoint()


            data = [[xx,yy] for (xx,yy) in zip(x,y)]
            table = wandb.Table(data = data, columns = ["layer_idx","CKA_diagonal"],)
            wandb_dict[diag_fig_name] =  wandb.plot.line(table,"layer_idx","CKA_diagonal")

            #plt.savefig(filedir +args.set + args.mode + args.additional_experiment_name + diag_fig_name)
            #wandb_dict[diag_fig_name] = wandb.Image(filedir +args.set + args.mode + args.additional_experiment_name+diag_fig_name+'.png')
            #plt.close()

        #draw CKA matrix
        fig_name = this_name[0] + "_" + this_name[1] + "_CKA"
        plt.figure(figsize=(7, 7))
        axes = plt.imshow(cka_output.cpu().numpy(), cmap='inferno')
        axes.axes.invert_yaxis()
        plt.colorbar()
        wandb_dict[fig_name] = wandb.Image(axes)
        #wandb_dict[fig_name] = wandb.plot(axes)
        #wandb_dict[fig_name] = wandb.Image(axes)    #wandb.Image(cka_output.cpu().numpy())
        #plt.savefig(filedir +args.set + args.mode + args.additional_experiment_name+fig_name)
        #breakpoint()
        #wandb_dict[fig_name] = wandb.Image(filedir +args.set + args.mode + args.additional_experiment_name+fig_name+'.png')
        plt.close()

    return wandb_dict



class AverageMeter():
    """Computes and stores the average and current value."""

    def __init__(self,
                 name: str,
                 fmt: str = ":f") -> None:
        """Construct an AverageMeter module.

        :param name: Name of the metric to be tracked.
        :param fmt: Output format string.
        """
        self.name = name
        self.fmt = fmt
        self.reset()
        self.first_val = 0
        self.first_update = True

    def reset(self):
        """Reset internal states."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.first_val = 0
        self.first_update = True


    def update(self, val: float, n: int = 1) -> None:
        """Update internal states given new values.

        :param val: New metric value.
        :param n: Step size for update.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
        if self.first_update:
            self.first_val = val
            self.first_update = False

    def __str__(self):
        """Get string name of the object."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize,
#                                          shuffle=False, num_workers=args.workers)

def change_n_to_model1(input_string, model1_name):
    result = input_string.replace('n',model1_name)
    return result

def change_o_to_model2(input_string, model2_name):
    result = input_string.replace('o',model2_name)
    return result

def change_no_to_model12(input_string, model1_name, model2_name):
    ntomodel1 = change_o_to_model2(input_string, model2_name)
    result = change_n_to_model1(ntomodel1, model1_name)
    return result


def __log_test_metric__(model1, model2 ,model1_name, model2_name, args, testloader,device):

    wandb_dict = {}

    num_of_branch = 5
    
    #log_criterion = MetricLoss(topk_pos=int(testloader.batch_size/2), topk_neg=testloader.batch_size, temp=1, pairs=None, ignore_self=args.ignore_self)
    #log_criterion = MetricLoss(topk_pos=1, topk_neg=10, temp=1, pairs=None, ignore_self=args.ignore_self)
    log_criterion = MetricLoss(topk_pos=5, topk_neg=400, temp=0.05, pairs=None, ignore_self=args.ignore_self)
    loss_func=nn.CrossEntropyLoss() 
    #
    nn_nn_losses = [AverageMeter(change_no_to_model12("nn_nn", model1_name, model2_name) + "/{}".format(name), ":.3f") for name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] ]
    no_no_losses = [AverageMeter(change_no_to_model12("no_no", model1_name, model2_name) + "/{}".format(name), ":.3f") for name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] ]
    # on_on_losses = [AverageMeter(change_no_to_model12("on_on", model1_name, model2_name) + "/{}".format(name), ":.3f") for name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] ]
    # no_nonn_losses = [AverageMeter(change_no_to_model12("no_nonn", model1_name, model2_name) + "/{}".format(name), ":.3f") for name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] ]
    # nn_nonn_losses = [AverageMeter(change_no_to_model12("nn_nonn", model1_name, model2_name) + "/{}".format(name), ":.3f") for name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] ]
    ce_losses1 = AverageMeter( model1_name + "_ce")
    ce_losses2 = AverageMeter( model2_name + "_ce")

    
    model1.eval()
    model2.eval()
    for p1,p2 in zip(model1.parameters(),model2.parameters()):
        p1.requires_grad = False
        p2.requires_grad = False
        
    total = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            model1_outputs = model1(images, return_feature=True, return_feature_norelu = args.return_feature_norelu)
            supervised_celoss1 = loss_func(model1_outputs[-1], labels)
            ce_losses1.update(supervised_celoss1.item(),1)


            _, predicted1 = torch.max(model1_outputs[-1].data, 1)
            correct1 += (predicted1 == labels).sum().item()


            model2_outputs = model2(images, return_feature=True, return_feature_norelu = args.return_feature_norelu)
            supervised_celoss2 = loss_func(model2_outputs[-1], labels)
            ce_losses2.update(supervised_celoss2.item(),1)


            _, predicted2 = torch.max(model2_outputs[-1].data, 1)
            correct2 += (predicted2 == labels).sum().item()


            for it in range(num_of_branch):


                this_local_feature = model1_outputs[(it)]                  
                this_global_feature = model2_outputs[(it)]

                if it != num_of_branch -1:
                    this_local_feature = F.adaptive_avg_pool2d(this_local_feature, 1)
                    with torch.no_grad():
                        this_global_feature = F.adaptive_avg_pool2d(this_global_feature, 1) 

                # For logging
                nn_pair = {'pos': ['nn'], 'neg': ['nn']}
                nn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=nn_pair).detach()
                nn_nn_losses[it].update(nn_loss.item(), 1)
                # nn_nn_losses_all[it].update(nn_loss.item(), 1)



                no_pair = {'pos': ['no'], 'neg': ['no']}
                no_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=no_pair).detach()
                no_no_losses[it].update(no_loss.item(), 1)
                # no_no_losses_all[it].update(no_loss.item(), 1)

                # on_pair = {'pos': ['on'], 'neg': ['on']}
                # on_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=on_pair).detach()
                # on_on_losses[it].update(on_loss.item(), 1)
                # # on_on_losses_all[it].update(on_loss.item(), 1)

                # no_nonn_pair = {'pos': ['no'], 'neg': ['no', 'nn']}
                # no_nonn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=no_nonn_pair).detach()
                # no_nonn_losses[it].update(no_nonn_loss.item(), 1)
                # # no_nonn_losses_all[it].update(no_nonn_loss.item(), 1)

                # nn_nonn_pair = {'pos': ['nn'], 'neg': ['no' ,'nn']}
                # nn_nonn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=nn_nonn_pair).detach()
                # nn_nonn_losses[it].update(nn_nonn_loss.item(), 1)
                # # nn_nonn_losses_all[it].update(nn_nonn_loss.item(), 1)   




    model1.train()
    model2.train()
    for p1,p2 in zip(model1.parameters(),model2.parameters()):
        p1.requires_grad = True
        p2.requires_grad = True

    wandb_dict[ce_losses1.name] = ce_losses1.avg
    wandb_dict[ce_losses2.name] = ce_losses2.avg
    wandb_dict["testacc_"+ model1_name] = 100 * correct1 / float(total)
    wandb_dict["testacc_"+ model2_name] = 100 * correct2 / float(total)

    wandb_dict.update({"client_metric_" + meter.name: meter.avg for meter in nn_nn_losses})
    wandb_dict["client_metric_" + change_no_to_model12("nn_nn", model1_name, model2_name) +'/avg'] = sum([meter.avg for meter in nn_nn_losses])/len(nn_nn_losses)
    wandb_dict.update({"client_metric_" + meter.name: meter.avg for meter in no_no_losses})
    wandb_dict["client_metric_" +change_no_to_model12("no_no", model1_name, model2_name) +'/avg'] = sum([meter.avg for meter in no_no_losses])/len(no_no_losses)
    #wandb_dict.update({meter.name: meter.avg for meter in nn_nonn_losses})
    # wandb_dict[change_no_to_model12("nn_nonn", model1_name, model2_name) +'/avg'] = sum([meter.avg for meter in nn_nonn_losses])/len(nn_nonn_losses)
    # wandb_dict.update({meter.name: meter.avg for meter in no_nonn_losses})
    # wandb_dict[change_no_to_model12("no_nonn", model1_name, model2_name) +'/avg'] = sum([meter.avg for meter in no_nonn_losses])/len(no_nonn_losses)
    # wandb_dict.update({meter.name: meter.avg for meter in on_on_losses})
    # wandb_dict[change_no_to_model12("on_on", model1_name, model2_name) +'/avg'] = sum([meter.avg for meter in on_on_losses])/len(on_on_losses)      
    return wandb_dict

def __log_local_sim__(local_delta, local_delta_name = ""):
    wandb_dict = {}
    length = len(local_delta)
    sim_all = []
    for key in local_delta[0].keys():
        global_delta = 0
        local_delta_layer = []
        for i in local_delta:
            global_delta += i[key]
            local_delta_layer.append(i[key].view(-1))
        global_delta /= length
        local_delta_layer = torch.stack(local_delta_layer, dim=0)

        global_delta = F.normalize(global_delta.view(-1).unsqueeze(0), 2, dim=1)
        local_delta_layer = F.normalize(local_delta_layer, 2, dim=1)
        sim = (local_delta_layer @ global_delta.T).squeeze().mean()
        sim_all.append(sim)
        wandb_dict[local_delta_name + key + "_cosine_similarity"] = sim.item()
    sim_all = torch.stack(sim_all).mean()
    wandb_dict[local_delta_name + "all_layers_mean" + "_cosine_similarity"] = sim_all.item()  
    return wandb_dict    


def metriceval(model, global_model,testloader, device, args, desc='', print_metric_loss=True, num_of_branch = 5):
        if testloader is None:
            print("No testloader")
            return 0.
        pair1 = {'pos': args.pair_pos, 'neg': args.pair_neg}
        log_criterion = MetricLoss(topk_pos=1, topk_neg=10, temp=1, pairs=pair1, ignore_self=args.ignore_self)
        model.eval()
        
        correct = 0
        total = 0

        loss_meters = {}
        for meter_name in ["nn_nn", "no_no", "on_on", "no_nonn", "nn_nonn", "nonn_nonn"]:
            loss_meters[meter_name] = AverageMeter(f"{meter_name}", ":.3f") 


        with torch.no_grad():
            for data in testloader:

                images, labels = data[0].to(device), data[1].to(device)
                # outputs = model(images)
                all_features = model(images, return_feature=True, return_feature_norelu=args.return_feature_norelu)
                outputs = all_features[-1]
                local_features = all_features[:-1]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                with torch.no_grad():
                    global_outputs = global_model(images, return_feature=True)
                global_features = global_outputs[:-1]


                for it in range(num_of_branch):

                    this_local_feature = local_features[(it)]                  
                    this_global_feature = global_features[(it)]

                    if it != num_of_branch -1:
                        this_local_feature = F.adaptive_avg_pool2d(this_local_feature, 1)
                        with torch.no_grad():
                            this_global_feature = F.adaptive_avg_pool2d(this_global_feature, 1) 

                    ## feature
                    nn_pair = {'pos': ['nn'], 'neg': ['nn']}
                    nn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=nn_pair).detach()
                    loss_meters["nn_nn"].update(nn_loss.item(), 1)

                    no_pair = {'pos': ['no'], 'neg': ['no']}
                    no_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=no_pair).detach()
                    loss_meters["no_no"].update(no_loss.item(), 1)

                    on_pair = {'pos': ['on'], 'neg': ['on']}
                    on_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=on_pair).detach()
                    loss_meters["on_on"].update(on_loss.item(), 1)

                    no_nonn_pair = {'pos': ['no'], 'neg': ['no', 'nn']}
                    no_nonn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=no_nonn_pair).detach()
                    loss_meters["no_nonn"].update(no_nonn_loss.item(), 1)

                    nn_nonn_pair = {'pos': ['nn'], 'neg': ['no' ,'nn']}
                    nn_nonn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=nn_nonn_pair).detach()
                    loss_meters["nn_nonn"].update(nn_nonn_loss.item(), 1)

                    nonn_nonn_pair = {'pos': ['no', 'nn'], 'neg': ['no' ,'nn']}
                    nonn_nonn_loss = log_criterion(old_feat = this_global_feature, new_feat = this_local_feature, target = labels, pairs=nonn_nonn_pair).detach()
                    loss_meters["nonn_nonn"].update(nonn_nonn_loss.item(), 1)


        acc = 100 * correct / float(total)
        print(f'- Local Model ({desc}) on the test images: {acc} %%')
        model.train()

        # results = {
        #     "acc": acc,
        #     "metric_losses": loss_meters,
        # }

        return loss_meters



def cka_allmodels(prev_model_weight,FedAvg_weight,local_weight,model,wandb_dict,testloader,args,epoch):
    models_state_dict_list = []
    models_state_dict_list.append(prev_model_weight)
    models_state_dict_list.append(FedAvg_weight)
    models_state_dict_list.append(local_weight[0])
    models_state_dict_list.append(local_weight[1])
    names_list = ["pastglobal","global", "local0", "local1"]
    wandb_dict.update(cka_visualize(model = copy.deepcopy(model), models_dict_list = models_state_dict_list, testloader = testloader, args = args, names_list = names_list, epoch = epoch))
    return wandb_dict


def umap_allmodels(prev_model_weight,FedAvg_weight,local_weight,wandb_dict,model,testloader,args):
    models_state_dict_list = []
    models_state_dict_list.append(prev_model_weight)
    models_state_dict_list.append(FedAvg_weight)
    models_state_dict_list.append(local_weight[0])
    models_state_dict_list.append(local_weight[1])
    
    
    names_list = ["pastglobal","global", "local0", "local1"]
    drawing_option = [
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
        [False, True, False, True],
        [False, False, True, True],
        [True, False, True, True],
        [False, True, True, True],
        [True, True, True, True],
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]
    wandb_dict.update(log_models_Umap(model = copy.deepcopy(model), models_dict_list = models_state_dict_list, testloader = testloader, args = args, names_list = names_list, num_of_sample_per_class = 100, draw_classes = 10, drawing_options = drawing_option
    ))


    wandb_dict.update(log_models_Umap(model = copy.deepcopy(model), models_dict_list = models_state_dict_list, testloader = testloader, args = args, names_list = names_list, num_of_sample_per_class = 100, draw_classes = 4, drawing_options = drawing_option
    ))
    return wandb_dict