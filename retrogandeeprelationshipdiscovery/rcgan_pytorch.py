import datetime
import os
import random
from random import shuffle
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import wandb
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import tools

import torch.nn as nn


class MaxMargin_Loss(torch.nn.Module):

    def __init__(self, sim_neg=25, batch_size=32, cuda=False, sim_margin=1):
        super(MaxMargin_Loss, self).__init__()
        self.sim_neg = sim_neg
        self.batch_size = batch_size
        self.cuda = cuda
        self.sim_margin = sim_margin

    def forward(self, y_pred, y_true):
        cost = 0.
        for i in range(0, self.sim_neg):
            new_true = torch.randperm(self.batch_size).to(y_pred.device)
            new_true = y_true[new_true]
            mg = self.sim_margin - torch.cosine_similarity(y_true, y_pred) + torch.cosine_similarity(new_true,
                                                                                                     y_pred)
            cost += torch.clamp(mg, min=0)

        return cost.mean() / self.sim_neg





class RetroCycleGAN(nn.Module):
    def forward(self,x):
        return self.g_AB(x)

    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0001, generator_lr=0.0001,
                 one_way_mm=True,cycle_mm=True,cycle_dis=True,id_loss=True,
                 device="cpu",name="default",fp16=False):

        if fp16:
            self.fp16=True

        super().__init__()
        self.save_folder = save_folder
        self.device = device
        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size
        self.name = name
        d_lr = discriminator_lr
        self.d_lr = d_lr
        g_lr = generator_lr
        self.g_lr = g_lr
        # Configuration
        self.one_way_mm = one_way_mm
        self.cycle_mm = cycle_mm
        self.cycle_dis = cycle_dis
        self.id_loss = id_loss

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_ABBA = self.build_discriminator()
        self.d_BAAB = self.build_discriminator()
        # return Adam(lr,amsgrad=True,decay=1e-8)

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

    def compile_all(self, optimizer="sgd"):

        self.dA_optimizer = Adam(self.d_A.parameters(), lr=self.d_lr)
        self.dB_optimizer = Adam(self.d_B.parameters(), lr=self.d_lr)

        self.dABBA_optimizer = Adam(self.d_ABBA.parameters(), lr=self.d_lr)
        self.dBAAB_optimizer = Adam(self.d_BAAB.parameters(), lr=self.d_lr)

        self.g_AB_optimizer = Adam(self.g_AB.parameters(), lr=self.g_lr)
        self.g_BA_optimizer = Adam(self.g_BA.parameters(), lr=self.g_lr)
        self.combined_optimizer = Adam([x for x in self.g_BA.parameters()]+
                                       [x for x in self.g_AB.parameters()], lr=self.g_lr)
        if self.fp16:
            self.dA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dABBA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dBAAB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_AB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_BA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.combined_optimizerscaler = torch.cuda.amp.GradScaler()



    def build_generator(self, hidden_dim=2048):
        return nn.Sequential(
            nn.Linear(self.word_vector_dimensions, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.word_vector_dimensions),
        )

    def build_discriminator(self, hidden_dim=2048):
        return nn.Sequential(
            nn.Linear(self.word_vector_dimensions, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()
        )

    def train(self, epochs, dataset, save_folder, batch_size=1, cache=False, epochs_per_checkpoint=5,
              dis_train_amount=3,iters=None):
        writer = SummaryWriter()
        wandb.init(project="retrogan",dir=save_folder)
        wandb.run.name = self.name
        wandb.watch(self,criterion="simlex")
        wandb.run.save()

        start_time = datetime.datetime.now()
        res = []

        class RetroPairsDataset(Dataset):
            """Face Landmarks dataset."""

            def __init__(self, original_dataset, retrofitted_dataset, save_folder,cache):
                """
                Args:
                    csv_file (string): Path to the csv file with annotations.
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
                """
                X_train, Y_train = tools.load_all_words_dataset_final(original_dataset,retrofitted_dataset,
                                                                      save_folder=save_folder,cache=cache)
                print("Shapes of training data:",
                      X_train.shape,
                      Y_train.shape)
                print(X_train)
                print(Y_train)
                print("*" * 100)
                self.x = X_train
                self.y = Y_train

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                # a = self.x.index[idx]
                # b  = self.y.index[idx]
                imgs_A = np.array(self.x.iloc[idx])
                imgs_B = np.array(self.y.iloc[idx])
                return torch.from_numpy(imgs_A),torch.from_numpy(imgs_B)

        ds = RetroPairsDataset(dataset["original"], dataset["retrofitted"],
                                                                      save_folder=save_folder, cache=cache)

        dataloader = DataLoader(ds, batch_size=batch_size,
                                shuffle=True, num_workers=0)

        dis_train_amount = dis_train_amount

        self.compile_all("adam")
        def run_batch(batch_i,imgs_A,imgs_B,epoch,count,training_epochs):
            with torch.cuda.amp.autocast():
                if imgs_A.shape[0] == 1:
                    print("Batch is equal to 1 in training.")
                    return
                a = datetime.datetime.now()
                imgs_A = imgs_A.to(self.device)
                imgs_B = imgs_B.to(self.device)

                imgs_A = imgs_A.half() if self.fp16 else imgs_A
                imgs_B = imgs_B.half() if self.fp16 else imgs_B

                fake_B = self.g_AB(imgs_A)
                fake_A = self.g_BA(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss = None
                dB_loss = None
                valid = torch.ones((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                fake = torch.zeros((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                accs = []
                b = datetime.datetime.now()
                # print("Data prep time",b-a)
                # TRAIN THE DISCRIMINATORS
                a = datetime.datetime.now()

                for _ in range(int(dis_train_amount)):
                    # TRAIN ON BATCH VALID
                    self.dA_optimizer.zero_grad()
                    dA = self.d_A(imgs_A)
                    dA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                    if self.fp16:
                        self.dA_optimizerscaler.scale(dA_loss_real).backward(retain_graph=True)
                        self.dA_optimizerscaler.step(self.dA_optimizer)
                        self.dA_optimizerscaler.update()
                    else:
                        dA_loss_real.backward(retain_graph=True)
                        self.dA_optimizer.step()
                    # TRAIN ON BATCH FAKE
                    self.dA_optimizer.zero_grad()
                    dA_f = self.d_A(fake_A)
                    dA_loss_fake = nn.BCEWithLogitsLoss()(dA_f, fake)

                    if self.fp16:
                        self.dA_optimizerscaler.scale(dA_loss_fake).backward(retain_graph=True)
                        self.dA_optimizerscaler.step(self.dA_optimizer)
                        self.dA_optimizerscaler.update()
                    else:
                        dA_loss_fake.backward(retain_graph=True)
                        self.dA_optimizer.step()

                    if dA_loss is None:
                        dA_loss = 0.5 * (float(dA_loss_real) + float(dA_loss_fake))
                    else:
                        dA_loss += 0.5 * (float(dA_loss_real) + float(dA_loss_fake))

                    # TRAIN ON BATCH VALID
                    self.dB_optimizer.zero_grad()
                    dB = self.d_B(imgs_B)
                    dB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)
                    if self.fp16:
                        self.dB_optimizerscaler.scale(dB_loss_real).backward(retain_graph=True)
                        self.dB_optimizerscaler.step(self.dB_optimizer)
                        self.dB_optimizerscaler.update()
                    else:
                        dB_loss_real.backward(retain_graph=True)
                        self.dB_optimizer.step()

                    # TRAIN ON BATCH FAKE
                    self.dB_optimizer.zero_grad()
                    dB_f = self.d_B(fake_B)
                    dB_loss_fake = nn.BCEWithLogitsLoss()(dB_f, fake)

                    if self.fp16:
                        self.dB_optimizerscaler.scale(dB_loss_fake).backward(retain_graph=True)
                        self.dB_optimizerscaler.step(self.dB_optimizer)
                        self.dB_optimizerscaler.update()
                    else:
                        dB_loss_fake.backward(retain_graph=True)
                        self.dB_optimizer.step()

                    # dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                    # dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                    if dB_loss is None:
                        dB_loss = 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                    else:
                        dB_loss += 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                    accs.append(0.25 * (accuracy_score([[1] if x > 0.5 else [0] for x in dB_f.detach().cpu().numpy()],
                                                       fake.cpu()) + accuracy_score(
                        [[1] if x > 0.5 else [0] for x in dB.detach().cpu().numpy()], valid.cpu()) +
                                        accuracy_score([[1] if x > 0.5 else [0] for x in dA_f.detach().cpu().numpy()],
                                                       fake.cpu()) + accuracy_score(
                                [[1] if x > 0.5 else [0] for x in dA.detach().cpu().numpy()], valid.cpu())))
                # ABBA
                b = datetime.datetime.now()
                # print("Dis train time", b - a)
                # TRAIN THE CYCLE DISCRIMINATORS
                if self.cycle_dis:
                    a = datetime.datetime.now()

                    fake_ABBA = self.g_BA(fake_B)
                    fake_BAAB = self.g_AB(fake_A)
                    self.dABBA_optimizer.zero_grad()
                    dA = self.d_ABBA(imgs_A)
                    dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)
                    if self.fp16:
                        self.dABBA_optimizerscaler.scale(dABBA_loss_real).backward()
                        self.dABBA_optimizerscaler.step(self.dABBA_optimizer)
                        self.dABBA_optimizerscaler.update()
                    else:
                        dABBA_loss_real.backward()
                        self.dABBA_optimizer.step()

                    self.dABBA_optimizer.zero_grad()
                    dA = self.d_ABBA(fake_ABBA)
                    dABBA_loss_fake = nn.BCEWithLogitsLoss()(dA, fake)

                    if self.fp16:
                        self.dABBA_optimizerscaler.scale(dABBA_loss_fake).backward()
                        self.dABBA_optimizerscaler.step(self.dABBA_optimizer)
                        self.dABBA_optimizerscaler.update()
                    else:
                        dABBA_loss_fake.backward()
                        self.dABBA_optimizer.step()

                    # BAAB
                    self.dABBA_optimizer.zero_grad()
                    dB = self.d_BAAB(imgs_B)
                    dBAAB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)

                    if self.fp16:
                        self.dBAAB_optimizerscaler.scale(dBAAB_loss_real).backward()
                        self.dBAAB_optimizerscaler.step(self.dBAAB_optimizer)
                        self.dBAAB_optimizerscaler.update()
                    else:
                        dBAAB_loss_real.backward()
                        self.dBAAB_optimizer.step()

                    self.dBAAB_optimizer.zero_grad()
                    dB = self.d_BAAB(fake_BAAB)
                    dBAAB_loss_fake = nn.BCEWithLogitsLoss()(dB, fake)

                    if self.fp16:
                        self.dBAAB_optimizerscaler.scale(dBAAB_loss_fake).backward()
                        self.dBAAB_optimizerscaler.step(self.dBAAB_optimizer)
                        self.dBAAB_optimizerscaler.update()
                    else:
                        dBAAB_loss_fake.backward()
                        self.dBAAB_optimizer.step()

                    d_cycle_loss = 0.25 * (dBAAB_loss_fake.item() + dBAAB_loss_real.item() +
                                           dABBA_loss_fake.item() + dABBA_loss_real.item())
                    b = datetime.datetime.now()
                    # print("Cycle discriminator train time", b - a)

                else:
                    d_cycle_loss = 0
                d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)
                # Calculate the max margin loss for A->B, B->A
                self.g_AB_optimizer.zero_grad()
                ## Max margin AB and BA
                if self.one_way_mm:
                    a = datetime.datetime.now()
                    mm_a = self.g_AB(imgs_A)
                    mm_a_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_a, imgs_B)

                    # Calling the step function on an Optimizer makes an update to its
                    # parameters
                    if self.fp16:
                        self.g_AB_optimizerscaler.scale(mm_a_loss).backward()
                        self.g_AB_optimizerscaler.step(self.g_AB_optimizer)
                        self.g_AB_optimizerscaler.update()
                    else:
                        mm_a_loss.backward(retain_graph=True)
                        self.g_AB_optimizer.step()
                    mm_a_loss = mm_a_loss.item()

                    self.g_BA_optimizer.zero_grad()
                    mm_b = self.g_BA(imgs_B)
                    mm_b_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_b, imgs_A)
                    if self.fp16:
                        self.g_BA_optimizerscaler.scale(mm_b_loss).backward()
                        self.g_BA_optimizerscaler.step(self.g_BA_optimizer)
                        self.g_BA_optimizerscaler.update()
                    else:
                        mm_b_loss.backward()
                        self.g_BA_optimizer.step()
                    mm_b_loss = mm_b_loss.item()
                    b = datetime.datetime.now()
                    # print("MM one way discriminator train time", b - a)


                else:
                    mm_a_loss = mm_b_loss = 0
                # Calculate the cycle A->B->A, B->A->B with max margin, and mae
                a = datetime.datetime.now()
                self.combined_optimizer.zero_grad()
                fake_B = self.g_AB(imgs_A)
                fake_A = self.g_BA(imgs_B)
                id_a = self.g_AB(imgs_B)
                id_b = self.g_BA(imgs_A)
                fake_ABBA = self.g_BA(fake_B)
                fake_BAAB = self.g_AB(fake_A)
                if self.cycle_mm:
                    mm_abba = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_ABBA, imgs_A)
                    mm_baab = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_BAAB, imgs_B)
                else:
                    mm_abba = mm_baab = 0

                mae_abba = torch.nn.L1Loss()(fake_ABBA, imgs_A)
                mae_baab = torch.nn.L1Loss()(fake_BAAB, imgs_B)
                if self.id_loss:
                    mae_id_abba = torch.nn.L1Loss()(id_a, imgs_A)
                    mae_id_baab = torch.nn.L1Loss()(id_b, imgs_B)
                else:
                    mae_id_abba = mae_id_baab = 0
                g_loss = mm_abba + mm_baab + mae_abba + mae_baab + mae_id_abba + mae_id_baab
                if self.fp16:
                    self.combined_optimizerscaler.scale(g_loss).backward()
                    self.combined_optimizerscaler.step(self.combined_optimizer)
                    self.combined_optimizerscaler.update()
                else:
                    g_loss.backward()
                    self.combined_optimizer.step()
                b = datetime.datetime.now()
                # print("Combined gen train time", b - a)

                if batch_i % 50 == 0:
                    print(
                        "Epoch", epoch, "/", training_epochs,
                        "Batch:", batch_i, len(dataloader),
                        "Global Step",count,
                        "Discriminator loss:", d_loss,
                        "Discriminator acc:", "{:.2f}".format(100 * np.mean(accs)),
                        "Combined loss:", "{:.2f}".format(g_loss.item()),
                        "MM_ABBA_CYCLE:", "{:.2f}".format(mm_abba.item() if self.cycle_mm else 0),
                        "MM_BAAB_CYCLE:", "{:.2f}".format(mm_baab.item() if self.cycle_mm else 0),
                        "abba acc:", "{:.2f}".format(mae_abba.item()),
                        "baab acc:", "{:.2f}".format(mae_baab.item()),
                        "idloss ab:", "{:.2f}".format(mae_id_abba.item() if self.id_loss else 0),
                        "idloss ba:", "{:.2f}".format(mae_id_baab.item() if self.id_loss else 0),
                        "mm ab loss:", "{:.2f}".format(mm_a_loss if self.one_way_mm else 0),
                        "mm ba loss:", "{:.2f}".format(mm_b_loss if self.one_way_mm else 0),
                        "discriminator cycle loss:", "{:.2f}".format(d_cycle_loss),
                    )
                    scalars = {
                        "epoch": epoch,
                        # "batch": batch_i,
                        "global_step":count,
                        "discriminator_loss": d_loss,
                        "discriminator_acc": np.mean(accs),
                        "combined_loss": g_loss.item(),
                        "loss":g_loss.item()+d_loss,
                        "MM_ABBA_CYCLE": mm_abba.item() if self.cycle_mm else 0,
                        "MM_BAAB_CYCLE": mm_baab.item() if self.cycle_mm else 0,
                        "abba_mae": mae_abba.item(),
                        "baab_mae": mae_baab.item(),
                        "idloss_ab": mae_id_abba.item() if self.id_loss else 0,
                        "idloss_ba": mae_id_baab.item() if self.id_loss else 0,
                        "mm_ab_loss": mm_a_loss if self.one_way_mm else 0,
                        "mm_ba_loss": mm_b_loss if self.one_way_mm else 0,
                        "discriminator_cycle_loss": d_cycle_loss
                    }
                    wandb.log(scalars)
                    writer.add_scalars("run-AAAI2020", tag_scalar_dict=scalars, global_step=count)
                    writer.flush()
        def train_(training_epochs, iters=None):
            count = 0
            if iters is None:
                for epoch in range(training_epochs):
                    # noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
                    for batch_i, (imgs_A, imgs_B) in enumerate(dataloader):
                        # a = datetime.datetime.now()
                        run_batch(batch_i,imgs_A,imgs_B,epoch,count,training_epochs)
                        count+=1
                        # b = datetime.datetime.now()
                        # print("batch time", b - a)
                    print("\n")
                    sl, sv = self.test(dataset)
                    writer.add_scalar("simlex",sl,global_step=count)
                    writer.add_scalar("simverb", sv,global_step=count)
                    wandb.log({"simlex":sl,"simverb":sv})
                    writer.flush()

                    if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                        self.save_model(name="checkpoint")

                    res.append((sl, sv))

                    # self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                    # wandbcb.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})

                    print(res)
                    print("\n")
            else:
                epoch = 0
                running = True
                while running:
                    for batch_i, (imgs_A, imgs_B) in enumerate(dataloader):
                        run_batch(batch_i,imgs_A,imgs_B,epoch,count,iters%len(dataloader))
                        count += 1
                    epoch+=1
                    print("\n")
                    sl, sv = self.test(dataset)
                    writer.add_scalar("simlex", sl, global_step=count)
                    writer.add_scalar("simverb", sv, global_step=count)
                    writer.flush()

                    if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                        self.save_model(name="checkpoint")

                    res.append((sl, sv))
                    print(res)
                    print("\n")
        print("Actual training")
        train_(epochs,iters=iters)
        print("Final performance")
        sl, sv = self.test(dataset)
        res.append((sl, sv))

        self.save_model(name="final")
        return res

    def test(self, dataset, simlex="testing/SimLex-999.txt", simverb="testing/SimVerb-3500.txt",
             fasttext="fasttext_model/cc.en.300.bin",
             prefix="en_"):
        self.to("cpu")
        sl = tools.test_sem(self.g_AB, dataset, dataset_location=simlex,
                            fast_text_location=fasttext, prefix=prefix,pt=True)[0]
        sv = tools.test_sem(self.g_AB, dataset, dataset_location=simverb,
                            fast_text_location=fasttext, prefix=prefix,pt=True)[0]
        self.to(self.device)
        return sl, sv
    def to_device(self,device):
        self.device = device
        self.to(device)

    def save_model(self, name=""):
        try:
            print("Trying to save model...")
            os.makedirs(self.save_folder,exist_ok=True)
            torch.save(self,os.path.join(self.save_folder,name+"complete.bin"))
            print("Succeeded!")
        except Exception as e:
            print(e)

