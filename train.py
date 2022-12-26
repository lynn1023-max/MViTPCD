# coding=utf-8
import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import numpy as np
import random
from trainpcd_options import parser
import itertools
from PCDdatasets import trainLoader, testLoader, valLoader
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# set seeds
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(2022)


def validate(model, testLoader):
    correctPrediction = 0
    wrongPrediction = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    # print(device)
    prob_all = []
    label_all = []
    for batchStep, batchData in enumerate(testLoader):
        data1 = batchData['I1'].float().to(device)
        data2 = batchData['I2'].float().to(device)
        label = batchData['label'].float().to(device)

        changeIndicator = model(data1, data2)
        changeIndicator = torch.sigmoid(changeIndicator)
        prob = changeIndicator.cpu().detach().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label_all.extend(label.cpu().detach().numpy())

        # print(changeIndicator)
        prediction_list = torch.topk(changeIndicator, k=1)
        # prediction = changeIndicator > 0.2860
        predic = prediction_list
        # print(predic)
        batchsize = len(predic.indices)
        # print(batchsize)
        for i in range(batchsize):
            prediction = predic.indices[i]
            if label[i] == 1:
                if prediction == 1:
                    correctPrediction = correctPrediction + 1
                    truePositive = truePositive + 1
                if prediction == 0:
                    wrongPrediction = wrongPrediction + 1
                    falseNegative = falseNegative + 1
            if label[i] == 0:
                if prediction == 1:
                    wrongPrediction = wrongPrediction + 1
                    falsePositive = falsePositive + 1
                if prediction == 0:
                    correctPrediction = correctPrediction + 1
                    trueNegative = trueNegative + 1

    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    print('Validation:--------------------------')
    print('Correct prediction: ' + str(correctPrediction))
    print('Wrong prediction: ' + str(wrongPrediction))
    print('falsePositive: ' + str(falsePositive) + ' ' + 'falseNegative: ' + str(falseNegative))
    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))

    FPRvalue = falsePositive / (trueNegative + falsePositive)
    f1score = f1_score(label_all, prob_all)
    auc = roc_auc_score(label_all, prob_all)
    print("FPR:{:.4f}".format(FPRvalue))
    print("F1-Score:{:.4f}".format(f1score))
    print("AUC:{:.4f}".format(auc))

    return f1_score, FPRvalue, accuracy, auc, correctPrediction, wrongPrediction


def main():
    from model.mvitpcd import PcdmViT
    best_acc = 0

    # load data

    # define model
    model = PcdmViT()

    # Resume = False
    Resume = False
    if Resume:
        path_checkpoint = args.resume_dir
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    CDNet = model.to(device, dtype=torch.float)
    print(device)

    # set optimization
    optimizer = optim.Adam(itertools.chain(CDNet.parameters()), lr=args.lr, betas=(0.937, 0.999))
    CDcriterionCD = nn.CrossEntropyLoss().to(device, dtype=torch.float)
    prob_all = []
    label_all = []
    correctPrediction = 0
    wrongPrediction = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    # training
    for epoch in range(1, args.num_epochs + 1):
        # train_bar = tqdm(epoch)
        running_results = {'batch_sizes': 0, 'CD_loss': 0, 'loss': 0}  # 'SR_loss':0,

        CDNet.train()
        for i, batch in enumerate(trainLoader):
            hr_img1 = batch['I1']
            hr_img2 = batch['I2']
            label = batch['label']
            running_results['batch_sizes'] += args.batchsize

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            result1 = CDNet(hr_img1, hr_img2)

            CD_loss = CDcriterionCD(result1, label)  # +CDcriterionCD(result2, label)+CDcriterionCD(result3, label)

            CDNet.zero_grad()
            CD_loss.backward()
            optimizer.step()

            running_results['CD_loss'] += CD_loss.item() * args.batchsize

            print("Training epoch: {}/{} iter: {}/{} loss: {:.4f}".format(epoch, args.num_epochs, i, len(trainLoader),
                                                                          running_results['CD_loss'] / running_results[
                                                                              'batch_sizes'], ))

            changeIndicator = torch.sigmoid(result1)
            prob = changeIndicator.cpu().detach().numpy()
            prob_all.extend(np.argmax(prob, axis=1))
            label_all.extend(label.cpu().detach().numpy())

            # print(changeIndicator)
            prediction_list = torch.topk(changeIndicator, k=1)
            # prediction = changeIndicator > 0.2860
            predic = prediction_list
            # print(predic)
            batchsize = len(predic.indices)
            # print(batchsize)
            for i in range(batchsize):
                prediction = predic.indices[i]
                if label[i] == 1:
                    if prediction == 1:
                        correctPrediction = correctPrediction + 1
                        truePositive = truePositive + 1
                    if prediction == 0:
                        wrongPrediction = wrongPrediction + 1
                        falseNegative = falseNegative + 1
                if label[i] == 0:
                    if prediction == 1:
                        wrongPrediction = wrongPrediction + 1
                        falsePositive = falsePositive + 1
                    if prediction == 0:
                        correctPrediction = correctPrediction + 1
                        trueNegative = trueNegative + 1

        accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
        FPRvalue = falsePositive / (trueNegative + falsePositive)
        f1score = f1_score(label_all, prob_all)
        auc = roc_auc_score(label_all, prob_all)
        print("TrainDataset accuracy:{:.4f} FPR:{:.4f} AUC:{:.4f} F1-Score:{:.4f}".format(accuracy, FPRvalue, auc,
                                                                                          f1score))
        # eval
        CDNet.eval()
        f1, fpr, acc, auc, _, _ = validate(CDNet, valLoader)

        if acc > best_acc or epoch == 1:
            best_acc = acc
            torch.save(CDNet.state_dict(), args.model_dir + 'netCD_epoch_%d.pth' % (epoch))
            print("Best Already saved: acc=", best_acc)
        else:
            print("Model not saved: history best_acc=", best_acc)


if __name__ == '__main__':
    main()
    TEST = False
    if TEST:
        from model.pcdbase import PcdmViT
        best_acc = 0
        model = PcdmViT()
        path_checkpoint = ''
        checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        CDNet = model.to(device, dtype=torch.float)
        print(device)
        CDcriterionCD = nn.CrossEntropyLoss().to(device, dtype=torch.float)
        prob_all = []
        label_all = []
        correctPrediction = 0
        wrongPrediction = 0
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        CDNet.eval()
        Tf1, Tfpr, Tacc, Tauc, _, _ = validate(CDNet, testLoader)
