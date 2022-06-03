################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-02-2022                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
Starting template for the "object classification - instances" track

Mostly based on Avalanche's "getting_started.py" example.

The template is organized as follows:
- The template is split in sections (CONFIG, TRANSFORMATIONS, ...) that can be
    freely modified.
- Don't remove the mandatory plugin (in charge of storing the test output).
- You will write most of the logic as a Strategy or as a Plugin. By default,
    the Naive (plain fine tuning) strategy is used.
- The train/eval loop should be left as it is.
- The Naive strategy already has a default logger + the accuracy metric. You
    are free to add more metrics or change the logger.
- The use of Avalanche training and logging code is not mandatory. However,
    you are required to use the given benchmark generation procedure. If not
    using Avalanche, make sure you are following the same train/eval loop and
    please make sure you are able to export the output in the expected format.
"""

import argparse
import datetime
from pathlib import Path
from typing import List
import sys
import os
sys.path.append('./avalanche')
#from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
from avalanche.training.plugins import *
import torchvision.models as models

from avalanche.benchmarks.utils import Composecls
from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger
# from avalanche.training.plugins import EvaluationPlugin,LRSchedulerPlugin
from avalanche.training.supervised import Naive,ICaRL
from devkit_tools.benchmarks import challenge_classification_benchmark
from devkit_tools.metrics.classification_output_exporter import \
    ClassificationOutputExporter
from avalanche.training.storage_policy import *
from devkit_tools.plugins.model_checkpoint import *

import warnings
warnings.filterwarnings("ignore")
import pdb
torch.backends.cudnn.enabled = False
# TODO: change this to the path where you downloaded (and extracted) the dataset
DATASET_PATH = '/home/xp1/cvworkshop/data/'

from torch import nn
import torch

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.flatten(start_dim=1)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class Vit_cls(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, n_classes=1000):
        super(Vit_cls, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.head = LinearClassifier(dim = self.model.embed_dim, num_labels = n_classes)


    def forward(self, x):
        x = self.model(x)
        return self.head(x)

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        
        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



def main(args):
    # --- CONFIG
    device=torch.device('cuda',3)
    #device = torch.device(
    #    f"cuda:{args.cuda}"
    #    if args.cuda >= 0 and torch.cuda.is_available()
    #    else "cpu"
    #)
    # ---------

    # --- TRANSFORMATIONS
    # This is the normalization used in torchvision models
    # https://pytorch.org/vision/stable/models.html
    torchvision_normalization = transforms.Normalize(
        mean=[0.485,  0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Add additional transformations here
    train_transform = Composecls(
        [transforms.Resize((224,224)),
         #transforms.RandomGrayscale(),
         #transforms.ColorJitter(0.5,0.5,0.5,0.5),
         #transforms.RandomAdjustSharpness(sharpness_factor=2),
         #transforms.RandomRotation(degrees=(0, 180)),
         #transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 4)),
         transforms.RandomAffine(10),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         ToTensor(),
         torchvision_normalization]
    )

    # Don't add augmentation transforms to the eval transformations!
    eval_transform = Composecls(
        [transforms.Resize((224,224)),
        ToTensor(),
        torchvision_normalization]
    )
    # ---------

    # --- BENCHMARK CREATION
    benchmark = challenge_classification_benchmark(
        dataset_path=DATASET_PATH,
        train_transform=train_transform,
        eval_transform=eval_transform,
        n_validation_videos=0,
        train_json_name='ego_objects_challenge_train.json',
        test_json_name='ego_objects_challenge_test.json',   #split_ego_eval.json
    )
    # ---------

    # --- MODEL CREATION
    

    #model = models.resnet50(pretrained=True)
    #model.fc = nn.Linear(2048, benchmark.n_classes)
    
    #model = models.efficientnet_v2_m(pretrained=True)    #efficientnet_b3
    #model.avgpool=nn.AdaptiveAvgPool2d((1,1))
    #for i,layer in enumerate(list(model.classifier)):
    #    lastlayer=layer
    #feature = lastlayer.in_features
    #model.classifier = nn.Sequential(
    #    nn.BatchNorm1d(feature),
    #    nn.Linear(feature, benchmark.n_classes),
    #)

    model = models.regnet_x_16gf(pretrained=True)   #regnet_y_3_2gf
    feature = model.fc.in_features
    model.avgpool=nn.AdaptiveAvgPool2d((1,1))
    model.fc = nn.Sequential(
        nn.BatchNorm1d(feature),
        nn.Linear(feature, benchmark.n_classes),
    )
    
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=1e-5)
    mb_size=8

    warmup_factor = 1.0 / 1000
    warmup_iters = \
        min(1000, len(benchmark.train_stream[0].dataset) // mb_size - 1)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,  T_max=4 #start_factor=warmup_factor, total_iters=warmup_iters
    )

    # ---------
    # TODO: Naive == plain fine tuning without replay, regularization, etc.
    # For the challenge, you'll have to implement your own strategy (or a
    # strategy plugin that changes the behaviour of the SupervisedTemplate)

    # --- PLUGINS CREATION
    # Avalanche already has a lot of plugins you can use!
    # Many mainstream continual learning approaches are available as plugins:
    # https://avalanche-api.continualai.org/en/latest/training.html#training-plugins
    
    #replay = ReplayPlugin(mem_size=3500, task_balanced_dataloader=True, storage_policy=ClassBalancedBuffer(max_size=3500))
    ewc = EWCPlugin(ewc_lambda=0.001)
    save_checkpoint = ModelCheckpoint(out_folder='./instance_classification_results_vit', file_prefix='track2_output')
    mandatory_plugins = [ewc,save_checkpoint,
        ClassificationOutputExporter(
            benchmark, save_folder='./instance_classification_results_vit')
    ]
    plugins: List[SupervisedPlugin] = [
        LRSchedulerPlugin(
            lr_scheduler, step_granularity='iteration',
            first_exp_only=True, first_epoch_only=True),
        # ...
    ] + mandatory_plugins
    # ---------

    # --- METRICS AND LOGGING
    evaluator = EvaluationPlugin(
        accuracy_metrics(
            epoch=True,
            stream=True
        ),
        loss_metrics(
            minibatch=False,
            epoch_running=True
        ),
        # May be useful if using a validation stream
        # confusion_matrix_metrics(stream=True),
        timing_metrics(
            experience=True, stream=True
        ),
        loggers=[InteractiveLogger(),
                #  WandBLogger(
                #  	run_name='vit',
                #      dir='./log/track_inst_cls/exp_' +
                #                 datetime.datetime.now().isoformat())
                 ],
    )
    # ---------

    # --- CREATE THE STRATEGY INSTANCE
    # In Avalanche, you can customize the training loop in 3 ways:
    #   1. Adapt the make_train_dataloader, make_optimizer, forward,
    #   criterion, backward, optimizer_step (and other) functions. This is the
    #   clean way to do things!
    #   2. Change the loop itself by reimplementing training_epoch or even
    #   _train_exp (not recommended).
    #   3. Create a Plugin that, by implementing the proper callbacks,
    #   can modify the behavior of the strategy.
    #  -------------
    #  Consider that popular strategies (EWC, LwF, Replay) are implemented
    #  as plugins. However, writing a plugin from scratch may be a tad
    #  tedious. For the challenge, we recommend going with the 1st option.
    #  In particular, you can create a subclass of the SupervisedTemplate
    #  (Naive is mostly an alias for the SupervisedTemplate) and override only
    #  the methods required to implement your solution.
    cl_strategy = ICaRL(
        feature_extractor= nn.Sequential( model.stem,
                                          model.trunk_output,
                                          model.avgpool,
                                          Flatten(),
        ),
        classifier= model.fc,
        optimizer=optimizer,
        memory_size= 3500,
        buffer_transform= eval_transform,
        fixed_memory= True,
        
        train_mb_size=mb_size,
        train_epochs=3,
        eval_mb_size=mb_size,
        device=device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=0 if 'valid' in benchmark.streams else -1,
        
    )
    # ---------

    # TRAINING LOOP
    print("Starting experiment...")
    for experience in benchmark.train_stream:
        current_experience_id = experience.current_experience
        print("Start of experience: ", current_experience_id)
        print("Current Classes: ", experience.classes_in_this_experience)

        data_loader_arguments = dict(
            num_workers=10,
            persistent_workers=True
        )
        
        cl_strategy.train(
            experience,
            **data_loader_arguments)
        print("Training completed")

        print("Computing accuracy on the complete test set")
        cl_strategy.eval(benchmark.test_stream, num_workers=10,
                         persistent_workers=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
