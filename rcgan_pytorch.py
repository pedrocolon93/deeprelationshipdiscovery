import datetime
import os
import random
from random import shuffle
from typing import Any

import numpy as np
import torch
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
            new_true = torch.randperm(self.batch_size)
            new_true = new_true.cuda() if self.cuda else new_true
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
                 device="cpu"):
        super().__init__()
        self.save_folder = save_folder
        self.device = device
        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.save_index = save_index

        # Number of filters in the first layer of G and D
        self.gf = generator_size
        self.df = discriminator_size

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

    def train(self, epochs, dataset, save_folder, batch_size=1, cache=False, epochs_per_checkpoint=4,
              dis_train_amount=3,iters=None):
        writer = SummaryWriter()
        start_time = datetime.datetime.now()
        res = []
        X_train, Y_train = tools.load_all_words_dataset_final(dataset["original"], dataset["retrofitted"],
                                                              save_folder=save_folder, cache=cache)
        print("Shapes of training data:",
              X_train.shape,
              Y_train.shape)
        print(X_train)
        print(Y_train)
        print("*" * 100)
        def load_batch(batch_size=32, always_random=False):
            def _int_load():
                iterable = list(Y_train.index)
                shuffle(iterable)
                batches = []
                print("Prefetching batches")
                for ndx in tqdm(range(0, len(iterable), batch_size)):
                    # if count == 50:
                    #     break
                    try:
                        ixs = iterable[ndx:min(ndx + batch_size, len(iterable))]
                        if always_random:
                            ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
                        imgs_A = X_train.loc[ixs]
                        imgs_B = Y_train.loc[ixs]
                        if np.isnan(imgs_A).any().any() or np.isnan(imgs_B).any().any():  # np.isnan(imgs_B).any():
                            # print(ixs)
                            continue

                        batches.append((imgs_A, imgs_B))
                    except Exception as e:
                        print("Skipping batch")
                        # print(e)
                return batches

            batches = _int_load()

            print("Beginning iteration")
            for i in tqdm(range(0, len(batches)), ncols=30):
                imgs_A, imgs_B = batches[i]

                yield torch.tensor(imgs_A.values, dtype=torch.float32).to(self.device), \
                      torch.tensor(imgs_B.values, dtype=torch.float32).to(self.device)

        dis_train_amount = dis_train_amount

        self.compile_all("adam")

        def train_(training_epochs, always_random=False):
            count = 0
            if iters is not None:
                for epoch in range(training_epochs):
                    # noise = np.random.normal(size=(batch_size, dimensionality), scale=0.01)
                    for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, always_random=always_random)):
                        fake_B = self.g_AB(imgs_A)
                        fake_A = self.g_BA(imgs_B)
                        # Train the discriminators (original images = real / translated = Fake)
                        dA_loss = None
                        dB_loss = None
                        valid = torch.ones((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                        fake = torch.zeros((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                        accs = []

                        # TRAIN THE DISCRIMINATORS
                        for _ in range(int(dis_train_amount)):
                            # TRAIN ON BATCH VALID
                            self.dA_optimizer.zero_grad()
                            dA = self.d_A(imgs_A)
                            dA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dA_loss_real.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dA_optimizer.step()
                            # TRAIN ON BATCH FAKE
                            self.dA_optimizer.zero_grad()
                            dA_f = self.d_A(fake_A)
                            dA_loss_fake = nn.BCEWithLogitsLoss()(dA_f, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dA_loss_fake.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dA_optimizer.step()

                            if dA_loss is None:
                                dA_loss = 0.5 * (dA_loss_real.item() + dA_loss_fake.item())
                            else:
                                dA_loss += 0.5 * (dA_loss_real.item() + dA_loss_fake.item())

                            # TRAIN ON BATCH VALID
                            self.dB_optimizer.zero_grad()
                            dB = self.d_B(imgs_B)
                            dB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dB_loss_real.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dB_optimizer.step()
                            # TRAIN ON BATCH FAKE
                            self.dB_optimizer.zero_grad()
                            dB_f = self.d_B(fake_B)
                            dB_loss_fake = nn.BCEWithLogitsLoss()(dB_f, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dB_loss_fake.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dB_optimizer.step()

                            # dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                            # dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                            if dB_loss is None:
                                dB_loss = 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                            else:
                                dB_loss += 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                            accs.append(0.25*(accuracy_score([[1] if x>0.5 else [0] for x in dB_f.detach().cpu().numpy()],fake.cpu())+accuracy_score([[1] if x>0.5 else [0] for x in dB.detach().cpu().numpy()],valid.cpu())+
                                              accuracy_score([[1] if x>0.5 else [0] for x in dA_f.detach().cpu().numpy()],fake.cpu())+accuracy_score([[1] if x>0.5 else [0] for x in dA.detach().cpu().numpy()],valid.cpu())))
                        # ABBA

                        # TRAIN THE CYCLE DISCRIMINATORS
                        if self.cycle_dis:
                            fake_ABBA = self.g_BA(fake_B)
                            fake_BAAB = self.g_AB(fake_A)
                            self.dABBA_optimizer.zero_grad()
                            dA = self.d_ABBA(imgs_A)
                            dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dABBA_loss_real.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dABBA_optimizer.step()
                            self.dABBA_optimizer.zero_grad()
                            dA = self.d_ABBA(fake_ABBA)
                            dABBA_loss_fake = nn.BCEWithLogitsLoss()(dA, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dABBA_loss_fake.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dABBA_optimizer.step()

                            # BAAB
                            self.dABBA_optimizer.zero_grad()
                            dB = self.d_BAAB(imgs_B)
                            dBAAB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dBAAB_loss_real.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dBAAB_optimizer.step()
                            self.dBAAB_optimizer.zero_grad()
                            dB = self.d_BAAB(fake_BAAB)
                            dBAAB_loss_fake = nn.BCEWithLogitsLoss()(dB, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dBAAB_loss_fake.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dBAAB_optimizer.step()
                            d_cycle_loss = 0.25 * (dBAAB_loss_fake.item() + dBAAB_loss_real.item() +
                                                   dABBA_loss_fake.item() + dABBA_loss_real.item())
                        else:
                            d_cycle_loss = 0
                        d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)
                        # Calculate the max margin loss for A->B, B->A
                        self.g_AB_optimizer.zero_grad()
                        ## Max margin AB and BA
                        if self.one_way_mm:
                            mm_a = self.g_AB(imgs_A)
                            mm_a_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_a, imgs_B)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            mm_a_loss.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.g_AB_optimizer.step()
                            mm_a_loss = mm_a_loss.item()
                            self.g_BA_optimizer.zero_grad()
                            mm_b = self.g_BA(imgs_B)
                            mm_b_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_b, imgs_A)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            mm_b_loss.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.g_BA_optimizer.step()
                            mm_b_loss = mm_b_loss.item()
                        else:
                            mm_a_loss = mm_b_loss = 0
                        # Calculate the cycle A->B->A, B->A->B with max margin, and mae
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

                        mae_abba = torch.nn.L1Loss()(fake_ABBA,imgs_A)
                        mae_baab = torch.nn.L1Loss()(fake_BAAB, imgs_B)
                        if self.id_loss:
                            mae_id_abba = torch.nn.L1Loss()(id_a,imgs_A)
                            mae_id_baab = torch.nn.L1Loss()(id_b, imgs_B)
                        else:
                            mae_id_abba = mae_id_baab = 0
                        g_loss = mm_abba+mm_baab+mae_abba+mae_baab+mae_id_abba+mae_id_baab
                        g_loss.backward()
                        self.combined_optimizer.step()

                        count+=1
                        if batch_i % 50 == 0:
                            print(
                                "Epoch",epoch,"/",training_epochs,
                                "Batch:",batch_i,
                                "Discriminator loss:",d_loss,
                                "Discriminator acc:",100*np.mean(accs),
                                "Combined loss:",g_loss.item(),
                                "MM_ABBA_CYCLE:",mm_abba.item() if self.cycle_mm else 0,
                                "MM_BAAB_CYCLE:",mm_baab.item() if self.cycle_mm else 0,
                                "abba acc:",mae_abba.item(),
                                "baab acc:",mae_baab.item(),
                                "idloss ab:",mae_id_abba.item() if self.id_loss else 0,
                                "idloss ba:",mae_id_baab.item() if self.id_loss else 0,
                                "mm ab loss:",mm_a_loss if self.one_way_mm else 0,
                                "mm ba loss:",mm_b_loss if self.one_way_mm else 0,
                                "discriminator cycle loss:",d_cycle_loss,
                            )
                            scalars={
                                "epoch": epoch,
                                # "batch": batch_i,
                                "discriminator_loss": d_loss,
                                "discriminator_acc": np.mean(accs),
                                "combined_loss": g_loss.item(),
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
                            writer.add_scalars("run",tag_scalar_dict=scalars,global_step=count)
                            writer.flush()

                        print("\n")
                        sl, sv = self.test(dataset)
                        writer.add_scalar("simlex",sl,global_step=count)
                        writer.add_scalar("simverb", sv,global_step=count)
                        writer.flush()

                        if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                            self.save_model(name="checkpoint_" + str(epoch))

                        res.append((sl, sv))

                        # self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                        # wandbcb.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})

                        print(res)
                        print("\n")
            else:
                epoch = 0
                running = True
                while running:
                    for batch_i, (imgs_A, imgs_B) in enumerate(load_batch(batch_size, always_random=always_random)):
                        if count == iters:
                            running = False
                            break
                        fake_B = self.g_AB(imgs_A)
                        fake_A = self.g_BA(imgs_B)
                        # Train the discriminators (original images = real / translated = Fake)
                        dA_loss = None
                        dB_loss = None
                        valid = torch.ones((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                        fake = torch.zeros((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                        accs = []

                        # TRAIN THE DISCRIMINATORS
                        for _ in range(int(dis_train_amount)):
                            # TRAIN ON BATCH VALID
                            self.dA_optimizer.zero_grad()
                            dA = self.d_A(imgs_A)
                            dA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dA_loss_real.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dA_optimizer.step()
                            # TRAIN ON BATCH FAKE
                            self.dA_optimizer.zero_grad()
                            dA_f = self.d_A(fake_A)
                            dA_loss_fake = nn.BCEWithLogitsLoss()(dA_f, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dA_loss_fake.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dA_optimizer.step()

                            if dA_loss is None:
                                dA_loss = 0.5 * (dA_loss_real.item() + dA_loss_fake.item())
                            else:
                                dA_loss += 0.5 * (dA_loss_real.item() + dA_loss_fake.item())

                            # TRAIN ON BATCH VALID
                            self.dB_optimizer.zero_grad()
                            dB = self.d_B(imgs_B)
                            dB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dB_loss_real.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dB_optimizer.step()
                            # TRAIN ON BATCH FAKE
                            self.dB_optimizer.zero_grad()
                            dB_f = self.d_B(fake_B)
                            dB_loss_fake = nn.BCEWithLogitsLoss()(dB_f, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dB_loss_fake.backward(retain_graph=True)

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dB_optimizer.step()

                            # dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                            # dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                            if dB_loss is None:
                                dB_loss = 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                            else:
                                dB_loss += 0.5 * (dB_loss_real.item() + dB_loss_fake.item())
                            accs.append(0.25 * (
                                        accuracy_score([[1] if x > 0.5 else [0] for x in dB_f.detach().cpu().numpy()],
                                                       fake.cpu()) + accuracy_score(
                                    [[1] if x > 0.5 else [0] for x in dB.detach().cpu().numpy()], valid.cpu()) +
                                        accuracy_score([[1] if x > 0.5 else [0] for x in dA_f.detach().cpu().numpy()],
                                                       fake.cpu()) + accuracy_score(
                                    [[1] if x > 0.5 else [0] for x in dA.detach().cpu().numpy()], valid.cpu())))
                        # ABBA

                        # TRAIN THE CYCLE DISCRIMINATORS
                        if self.cycle_dis:
                            fake_ABBA = self.g_BA(fake_B)
                            fake_BAAB = self.g_AB(fake_A)
                            self.dABBA_optimizer.zero_grad()
                            dA = self.d_ABBA(imgs_A)
                            dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dABBA_loss_real.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dABBA_optimizer.step()
                            self.dABBA_optimizer.zero_grad()
                            dA = self.d_ABBA(fake_ABBA)
                            dABBA_loss_fake = nn.BCEWithLogitsLoss()(dA, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dABBA_loss_fake.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dABBA_optimizer.step()

                            # BAAB
                            self.dABBA_optimizer.zero_grad()
                            dB = self.d_BAAB(imgs_B)
                            dBAAB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dBAAB_loss_real.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dBAAB_optimizer.step()
                            self.dBAAB_optimizer.zero_grad()
                            dB = self.d_BAAB(fake_BAAB)
                            dBAAB_loss_fake = nn.BCEWithLogitsLoss()(dB, fake)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            dBAAB_loss_fake.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.dBAAB_optimizer.step()
                            d_cycle_loss = 0.25 * (dBAAB_loss_fake.item() + dBAAB_loss_real.item() +
                                                   dABBA_loss_fake.item() + dABBA_loss_real.item())
                        else:
                            d_cycle_loss = 0
                        d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)
                        # Calculate the max margin loss for A->B, B->A
                        self.g_AB_optimizer.zero_grad()
                        ## Max margin AB and BA
                        if self.one_way_mm:
                            mm_a = self.g_AB(imgs_A)
                            mm_a_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_a, imgs_B)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            mm_a_loss.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.g_AB_optimizer.step()
                            mm_a_loss = mm_a_loss.item()
                            self.g_BA_optimizer.zero_grad()
                            mm_b = self.g_BA(imgs_B)
                            mm_b_loss = MaxMargin_Loss(batch_size=imgs_A.shape[0])(mm_b, imgs_A)

                            # Backward pass: compute gradient of the loss with respect to model
                            # parameters
                            mm_b_loss.backward()

                            # Calling the step function on an Optimizer makes an update to its
                            # parameters
                            self.g_BA_optimizer.step()
                            mm_b_loss = mm_b_loss.item()
                        else:
                            mm_a_loss = mm_b_loss = 0
                        # Calculate the cycle A->B->A, B->A->B with max margin, and mae
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
                        g_loss.backward()
                        self.combined_optimizer.step()

                        count += 1
                        if batch_i % 50 == 0:
                            print(
                                "Epoch", epoch, "/", training_epochs,
                                "Batch:", batch_i,
                                "Discriminator loss:", d_loss,
                                "Discriminator acc:", 100 * np.mean(accs),
                                "Combined loss:", g_loss.item(),
                                "MM_ABBA_CYCLE:", mm_abba.item() if self.cycle_mm else 0,
                                "MM_BAAB_CYCLE:", mm_baab.item() if self.cycle_mm else 0,
                                "abba acc:", mae_abba.item(),
                                "baab acc:", mae_baab.item(),
                                "idloss ab:", mae_id_abba.item() if self.id_loss else 0,
                                "idloss ba:", mae_id_baab.item() if self.id_loss else 0,
                                "mm ab loss:", mm_a_loss if self.one_way_mm else 0,
                                "mm ba loss:", mm_b_loss if self.one_way_mm else 0,
                                "discriminator cycle loss:", d_cycle_loss,
                            )
                            scalars = {
                                "epoch": epoch,
                                # "batch": batch_i,
                                "discriminator_loss": d_loss,
                                "discriminator_acc": np.mean(accs),
                                "combined_loss": g_loss.item(),
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
                            writer.add_scalars("run", tag_scalar_dict=scalars, global_step=count)
                            writer.flush()

                        print("\n")
                        sl, sv = self.test(dataset)
                        writer.add_scalar("simlex", sl, global_step=count)
                        writer.add_scalar("simverb", sv, global_step=count)
                        writer.flush()

                        if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                            self.save_model(name="checkpoint_" + str(epoch))

                        res.append((sl, sv))
                        print(res)
                        print("\n")
        print("Actual training")
        train_(epochs)
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
        os.makedirs(self.save_folder,exist_ok=True)
        torch.save(self,os.path.join(self.save_folder,name+"complete.bin"))

