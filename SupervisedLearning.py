import time

import matplotlib.pyplot as plt
import torch
from numpy import Inf


class SupervisedLearning:
    def __init__(self, model, epochs, train_loader, val_loader, learning_rate, weight_decay):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

    def calc_weights(self, labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0, labels.size(0)):
            pos_weight = torch.sum(labels[label_idx] == 1)
            neg_weight = torch.sum(labels[label_idx] == 0)
            ratio = float(neg_weight.item() / pos_weight.item())
            pos_tensor[label_idx] = ratio * pos_tensor[label_idx]

        return pos_tensor

    def intersection_over_union(self, tensor, labels):
        iou = 0
        foreground_acc = 0
        labels_tens = labels.type(torch.BoolTensor)
        ones_tens = torch.ones_like(tensor, device=self.device)
        zeros_tens = torch.zeros_like(tensor, device=self.device)
        if tensor.shape[0] > 1:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum((1, 2))

            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum((1, 2))
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens
        else:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum()
            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum()
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens

        del temp_tens
        del labels_tens
        del ones_tens
        del zeros_tens
        torch.cuda.empty_cache()
        total_iou = iou
        return total_iou, foreground_acc

    def train(self):
        self.model.to(self.device)

        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.learning_rate, weight_decay=self.weight_decay)

        train_loss = 0
        train_iou = 0
        train_acc = 0
        val_loss = 0
        val_iou = 0
        val_acc = 0
        total_foreground_train = 0
        total_foreground_val = 0
        val_loss_min = Inf
        train_loss_epoch = []
        val_loss_epoch = []

        start = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            print(epoch)

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                fig = plt.figure()
                fig.add_subplot(3, 1, 1)
                img = images.numpy()[0][0]
                plt.imshow(img)
                fig.add_subplot(3, 1, 2)
                val = labels.numpy()[0][0]
                plt.imshow(val)

                output = self.model(images)

                fig.add_subplot(3, 1, 3)
                out = output.detach().numpy()[0][0]
                plt.imshow(out)
                print("output done")

                loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output,
                                                                                        labels)
                print("loss done")

                iou_temp, acc_temp = self.intersection_over_union(output, labels)

                loss.backward()
                print("backward done")
                optimizer.step()
                print("step done")
                train_loss += loss.item() * images.size(0)
                train_iou += iou_temp
                train_acc += torch.sum(acc_temp).item()
                total_foreground_train += torch.sum(labels == 1).item()

            print("Time per epoch: ", (time.time() - start) / 60)

            torch.save(self.model.state_dict(), 'state_dict.pt')

            train_loss = train_loss / len(self.train_loader.dataset)
            train_iou = train_iou.item() / len(self.train_loader.dataset)
            train_acc = train_acc / total_foreground_train

            train_loss_epoch.append(train_loss)

        return train_loss_epoch
