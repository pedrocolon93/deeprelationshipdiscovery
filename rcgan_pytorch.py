import datetime
import os
import random

import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# random.seed(10)
# np.random.seed(10)
# tf.random.set_seed(10)
# torch.manual_seed(10)
import tools
import wandb

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
            # mg = self.sim_margin - torch.cosine_similarity(y_true, y_pred) + torch.cosine_similarity(new_true,
            #                                                                                          y_pred)

            normalize_a = self.l2_norm(y_true)
            normalize_b =  self.l2_norm(y_pred)
            normalize_c =  self.l2_norm(new_true)
            minimize = torch.sum(torch.multiply(normalize_a, normalize_b))
            maximize = torch.sum(torch.multiply(normalize_a, normalize_c))
            mg = self.sim_margin - minimize + maximize
            cost += torch.clamp(mg, min=0)

        return cost / self.sim_neg

    def l2_norm(self, x):
        sq = torch.square(x)
        square_sum = torch.sum(torch.sum(sq, dim=1))
        epsilon = 1e-12
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon,device=x.device)))
        normalize_a_t = x * x_inv_norm
        return normalize_a_t


class CycleCondLoss(torch.nn.Module):

    def __init__(self):
        super(CycleCondLoss, self).__init__()

    def forward(self, d_ground, d_approx):
        cost = torch.log(d_ground)+torch.log(1-d_approx)
        return -1*cost.mean()




class RetroCycleGAN(nn.Module):
    def forward(self,x):
        return self.g_AB(x)

    def __init__(self, save_index="0", save_folder="./", generator_size=32,
                 discriminator_size=64, word_vector_dimensions=300,
                 discriminator_lr=0.0001, generator_lr=0.0001,
                 one_way_mm=True,cycle_mm=True,cycle_dis=True,id_loss=True, cycle_loss=True,
                 device="cpu",name="default",fp16=False):
        super().__init__()
        self.fp16=fp16

        self.save_folder = save_folder
        self.device = device
        # Input shape
        self.word_vector_dimensions = word_vector_dimensions
        self.save_index = save_index
        self.cycle_loss = cycle_loss
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
        self.cycle_mm_weight = 2
        self.id_loss_weight = 0.01
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.initializer = tf.keras.initializers.GlorotUniform()

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_ABBA = self.build_c_discriminator()
        self.d_BAAB = self.build_c_discriminator()
        # return Adam(lr,amsgrad=True,decay=1e-8)

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
    def put_train(self):
        self.d_A.train()
        self.d_B.train()
        self.d_ABBA.train()
        self.d_BAAB.train()
        self.g_AB.train()
        self.g_BA.train()

    def put_eval(self):
        self.d_A.eval()
        self.d_B.eval()
        self.d_ABBA.eval()
        self.d_BAAB.eval()
        self.g_AB.eval()
        self.g_BA.eval()


    def compile_all(self, optimizer="sgd"):

        self.dA_optimizer = Adam(self.d_A.parameters(), lr=self.d_lr,eps=1e-10)
        self.dB_optimizer = Adam(self.d_B.parameters(), lr=self.d_lr,eps=1e-10)

        self.dABBA_optimizer = Adam(self.d_ABBA.parameters(), lr=self.d_lr,eps=1e-10)
        self.dBAAB_optimizer = Adam(self.d_BAAB.parameters(), lr=self.d_lr,eps=1e-10)

        self.g_AB_optimizer = Adam(self.g_AB.parameters(), lr=self.g_lr,eps=1e-10)
        self.g_BA_optimizer = Adam(self.g_BA.parameters(), lr=self.g_lr,eps=1e-10)
        self.combined_optimizer = Adam([x for x in self.g_BA.parameters()]+
                                       [x for x in self.g_AB.parameters()], lr=self.g_lr,eps=1e-10)
        pp=0
        for p in list(self.g_BA.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print(pp)
        pp = 0
        for n,p in list(self.d_A.named_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            print(n,nn)
            pp += nn
        print(pp)
        if self.fp16:
            self.dA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dABBA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.dBAAB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_AB_optimizerscaler = torch.cuda.amp.GradScaler()
            self.g_BA_optimizerscaler = torch.cuda.amp.GradScaler()
            self.combined_optimizerscaler = torch.cuda.amp.GradScaler()



    def build_generator(self, hidden_dim=2048):
        inpt = nn.Linear(self.word_vector_dimensions, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)
        hid = nn.Linear(hidden_dim, hidden_dim)
        s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
        hid.weight = nn.parameter.Parameter(s)
        hid.bias.data.fill_(0)
        out = nn.Linear(hidden_dim, self.word_vector_dimensions)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)
        return nn.Sequential(
            inpt,
            nn.ReLU(),
            nn.Dropout(0.2),
            hid,
            nn.ReLU(),
            nn.Dropout(0.2),
            out,
        )

    def build_discriminator(self, hidden_dim=2048):
        inpt = nn.Linear(self.word_vector_dimensions, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)
        hid = nn.Linear(hidden_dim, hidden_dim)
        s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
        hid.weight = nn.parameter.Parameter(s)
        hid.bias.data.fill_(0)
        out = nn.Linear(hidden_dim, 1)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)
        bn = nn.BatchNorm1d(hidden_dim,momentum=0.99,eps=0.001)
        return nn.Sequential(
            inpt,
            nn.ReLU(),
            nn.Dropout(0.3),
            hid,
            nn.ReLU(),
            bn,
            nn.Dropout(0.3),
            out,
            # nn.Sigmoid()
        )
    def build_c_discriminator(self, hidden_dim=2048):
        inpt = nn.Linear(self.word_vector_dimensions*2, hidden_dim)
        s = torch.tensor(self.initializer(shape=inpt.weight.shape).numpy())
        inpt.weight = nn.parameter.Parameter(s)
        inpt.bias.data.fill_(0)
        hid = nn.Linear(hidden_dim, hidden_dim)
        s = torch.tensor(self.initializer(shape=hid.weight.shape).numpy())
        hid.weight = nn.parameter.Parameter(s)
        hid.bias.data.fill_(0)
        out = nn.Linear(hidden_dim, 1)
        s = torch.tensor(self.initializer(shape=out.weight.shape).numpy())
        out.weight = nn.parameter.Parameter(s)
        out.bias.data.fill_(0)
        return nn.Sequential(
            inpt,
            nn.ReLU(),
            nn.Dropout(0.3),
            hid,
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim,momentum=0.99,eps=0.001),
            nn.Dropout(0.3),
            out,
            nn.Sigmoid()
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
                imgs_A = np.array(self.x.iloc[idx],dtype=np.float)
                imgs_B = np.array(self.y.iloc[idx],dtype=np.float)
                imgs_A/=np.linalg.norm(imgs_A)
                imgs_B/=np.linalg.norm(imgs_B)
                return torch.from_numpy(imgs_A),torch.from_numpy(imgs_B)

        ds = RetroPairsDataset(dataset["original"], dataset["retrofitted"],
                                                                      save_folder=save_folder, cache=cache)
        #
        dataloader = DataLoader(ds, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        X_train, Y_train = tools.load_all_words_dataset_final(dataset["original"], dataset["retrofitted"],
                                                              save_folder=save_folder,cache=cache)

        dis_train_amount = dis_train_amount


        def load_batch(batch_size=32, always_random=False):
            def _int_load():
                iterable = list(Y_train.index)
                random.shuffle(iterable)
                batches = []
                print("Prefetching batches")
                for ndx in tqdm(range(0, len(iterable), batch_size)):
                    try:
                        ixs = iterable[ndx:min(ndx + batch_size, len(iterable))]
                        if always_random:
                            ixs = list(np.array(iterable)[random.sample(range(0, len(iterable)), batch_size)])
                        imgs_A = X_train.loc[ixs]
                        imgs_B = Y_train.loc[ixs]
                        if np.isnan(imgs_A).any().any() or np.isnan(imgs_B).any().any():  # np.isnan(imgs_B).any():
                            # print(ixs)
                            continue

                        batches.append((torch.tensor(imgs_A.values), torch.tensor(imgs_B.values)))
                    except Exception as e:
                        print("Skipping batch")
                        print(e)
                return batches

            batches = _int_load()

            print("Beginning iteration")
            for i in tqdm(range(0, len(batches)), ncols=30):
                imgs_A, imgs_B = batches[i]
                yield imgs_A, imgs_B
        self.compile_all("adam")
        def run_batch(batch_i,imgs_A,imgs_B,epoch,count,training_epochs):
            with torch.cuda.amp.autocast():
                if imgs_A.shape[0] == 1:
                    print("Batch is equal to 1 in training.")
                    return
                a = datetime.datetime.now()
                imgs_A = imgs_A.to(self.device)
                imgs_B = imgs_B.to(self.device)

                imgs_A = imgs_A.half() if self.fp16 else imgs_A.float()
                imgs_B = imgs_B.half() if self.fp16 else imgs_B.float()

                fake_B = self.g_AB(imgs_A)
                fake_A = self.g_BA(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss = None
                dB_loss = None
                valid = torch.ones((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                fake = torch.zeros((imgs_A.shape[0], 1)).to(self.device)  # *noisy_entries_num,) )
                # accs = []
                b = datetime.datetime.now()
                # print("Data prep time",b-a)
                # TRAIN THE DISCRIMINATORS
                a = datetime.datetime.now()
                if False:
                    for _ in range(int(dis_train_amount)):
                        if _ % 2 == 0:
                            # print("Adding noise")
                            i_A = imgs_A+torch.tensor(np.random.uniform(low =-1,size=(imgs_A.shape[0], self.word_vector_dimensions)), device=imgs_A.device).half()
                            i_B = imgs_B+torch.tensor(np.random.uniform(low =-1,size=(imgs_A.shape[0], self.word_vector_dimensions)), device=imgs_B.device).half()
                            f_A = fake_A+torch.tensor(np.random.uniform(low =-1,size=(imgs_A.shape[0], self.word_vector_dimensions)), device=fake_A.device).half()
                            f_B = fake_B+torch.tensor(np.random.uniform(low =-1,size=(imgs_A.shape[0], self.word_vector_dimensions)), device=fake_B.device).half()
                        else:
                            i_A = imgs_A
                            i_B = imgs_B
                            f_B = fake_B
                            f_A = fake_A
                        # with torch.no_grad():
                        # TRAIN ON BATCH VALID
                        self.dA_optimizer.zero_grad()
                        dA = self.d_A(i_A)
                        dA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)

                        if self.fp16:
                            self.dA_optimizerscaler.scale(dA_loss_real).backward()
                            self.dA_optimizerscaler.step(self.dA_optimizer)
                            self.dA_optimizerscaler.update()
                        else:
                            dA_loss_real.backward(retain_graph=True)
                            self.dA_optimizer.step()
                        # TRAIN ON BATCH FAKE
                        self.dA_optimizer.zero_grad()
                        dA_f = self.d_A(f_A)
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
                        dB = self.d_B(i_B)
                        dB_loss_real = nn.BCEWithLogitsLoss()(dB, valid)
                        if self.fp16:
                            self.dB_optimizerscaler.scale(dB_loss_real).backward()
                            self.dB_optimizerscaler.step(self.dB_optimizer)
                            self.dB_optimizerscaler.update()
                        else:
                            dB_loss_real.backward(retain_graph=True)
                            self.dB_optimizer.step()

                        # TRAIN ON BATCH FAKE
                        self.dB_optimizer.zero_grad()
                        dB_f = self.d_B(f_B)
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
                else:
                    dA_loss = 0
                    dB_loss = 0
                # ABBA
                b = datetime.datetime.now()
                d_loss = (1.0 / dis_train_amount) * 0.5 * np.add(dA_loss, dB_loss)

                # print("Dis train time", b - a)
                # TRAIN THE CYCLE DISCRIMINATORS
                if self.cycle_dis:
                    a = datetime.datetime.now()

                    fake_ABBA = self.g_BA(fake_B)
                    fake_BAAB = self.g_AB(fake_A)
                    self.dABBA_optimizer.zero_grad()
                    dA = self.d_ABBA(torch.cat([fake_B,imgs_A],1))
                    dA_r = self.d_ABBA(torch.cat([fake_B,fake_ABBA],1))
                    dABBA_loss_real = CycleCondLoss()(dA,dA_r)
                    # dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)
                    if self.fp16:
                        self.dABBA_optimizerscaler.scale(dABBA_loss_real).backward()
                        self.dABBA_optimizerscaler.step(self.dABBA_optimizer)
                        self.dABBA_optimizerscaler.update()
                    else:
                        dABBA_loss_real.backward()
                        self.dABBA_optimizer.step()

                    self.dBAAB_optimizer.zero_grad()
                    dB = self.d_BAAB(torch.cat([fake_A, imgs_B], 1))
                    dB_r = self.d_BAAB(torch.cat([fake_A, fake_BAAB], 1))
                    dBAAB_loss_real = CycleCondLoss()(dB, dB_r)
                    # dABBA_loss_real = nn.BCEWithLogitsLoss()(dA, valid)
                    if self.fp16:
                        self.dBAAB_optimizerscaler.scale(dBAAB_loss_real).backward()
                        self.dBAAB_optimizerscaler.step(self.dBAAB_optimizer)
                        self.dBAAB_optimizerscaler.update()
                    else:
                        dBAAB_loss_real.backward()
                        self.dBAAB_optimizer.step()



                    d_cycle_loss = 0.5 * ( dBAAB_loss_real.item() + dABBA_loss_real.item())
                    b = datetime.datetime.now()
                    # print("Cycle discriminator train time", b - a)

                else:
                    d_cycle_loss = 0
                # Calculate the max margin loss for A->B, B->A
                ## Max margin AB and BA
                if self.one_way_mm:
                    self.g_AB_optimizer.zero_grad()
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
                # with torch.no_grad():
                valid_A = self.d_A(fake_A)
                valid_B = self.d_B(fake_B)
                valid_A_loss = nn.BCEWithLogitsLoss()(valid_A, valid)
                valid_B_loss = nn.BCEWithLogitsLoss()(valid_B, valid)
                id_a = fake_B
                id_b = fake_A
                if self.id_loss:
                    gamma = 1.0
                    mae_id_abba = gamma*torch.nn.L1Loss()(id_a, imgs_A)
                    mae_id_baab = gamma*torch.nn.L1Loss()(id_b, imgs_B)
                else:
                    mae_id_abba = mae_id_baab = 0
                fake_ABBA = self.g_BA(fake_B)
                fake_BAAB = self.g_AB(fake_A)
                if self.cycle_mm:
                    mm_abba = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_ABBA, imgs_A)
                    mm_baab = MaxMargin_Loss(batch_size=imgs_A.shape[0])(fake_BAAB, imgs_B)
                else:
                    mm_abba = mm_baab = 0

                if self.cycle_loss:
                    mae_abba = torch.nn.L1Loss()(fake_ABBA, imgs_A)
                    mae_baab = torch.nn.L1Loss()(fake_BAAB, imgs_B)
                else:
                    mae_abba = 0
                    mae_baab = 0
                if self.cycle_dis:
                    dA = self.d_ABBA(torch.cat([fake_B, imgs_A], 1))
                    dA_r = self.d_ABBA(torch.cat([fake_B, fake_ABBA], 1))
                    dABBA_loss_real = CycleCondLoss()(dA, dA_r)
                    dB = self.d_BAAB(torch.cat([fake_A, imgs_B], 1))
                    dB_r = self.d_BAAB(torch.cat([fake_A, fake_BAAB], 1))
                    dBAAB_loss_real = CycleCondLoss()(dB, dB_r)
                else:
                    dABBA_loss_real = 0
                    dBAAB_loss_real = 0
                g_loss = valid_A_loss + valid_B_loss + \
                         self.cycle_mm_weight*mm_abba + self.cycle_mm_weight*mm_baab + \
                         mae_abba + mae_baab + \
                         self.id_loss_weight*mae_id_abba + self.id_loss_weight*mae_id_baab +\
                         dBAAB_loss_real + dABBA_loss_real
                if self.fp16:
                    self.combined_optimizerscaler.scale(g_loss).backward()
                    self.combined_optimizerscaler.step(self.combined_optimizer)
                    self.combined_optimizerscaler.update()
                else:
                    g_loss.backward()
                    self.combined_optimizer.step()
                b = datetime.datetime.now()
                # print("Combined gen train time", b - a)

                if batch_i % 50 == 0 and batch_i!=0:
                    print(
                        "Epoch", epoch, "/", training_epochs,
                        "Batch:", batch_i, len(dataloader),
                        "Global Step",count,
                        "Discriminator loss:", d_loss,
                        # "Discriminator acc:", "{:.2f}".format(100 * np.mean(accs)),
                        "Combined loss:", "{:.2f}".format(g_loss.item()),
                        "MM_ABBA_CYCLE:", "{:.2f}".format(mm_abba.item() if self.cycle_mm else 0),
                        "MM_BAAB_CYCLE:", "{:.2f}".format(mm_baab.item() if self.cycle_mm else 0),
                        "abba acc:", "{:.2f}".format(mae_abba.item() if self.cycle_loss else 0),
                        "baab acc:", "{:.2f}".format(mae_baab.item() if self.cycle_loss else 0),
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
                        # "discriminator_acc": np.mean(accs),
                        "combined_loss": g_loss.item(),
                        "loss":g_loss.item()+d_loss,
                        "MM_ABBA_CYCLE": mm_abba.item() if self.cycle_mm else 0,
                        "MM_BAAB_CYCLE": mm_baab.item() if self.cycle_mm else 0,
                        "abba_mae": mae_abba.item() if self.cycle_loss else 0,
                        "baab_mae": mae_baab.item() if self.cycle_loss else 0,
                        "cycle_da": valid_A_loss.item(),
                        "cycle_db": valid_B_loss.item(),
                        "idloss_ab": mae_id_abba.item() if self.id_loss else 0,
                        "idloss_ba": mae_id_baab.item() if self.id_loss else 0,
                        "mm_ab_loss": mm_a_loss if self.one_way_mm else 0,
                        "mm_ba_loss": mm_b_loss if self.one_way_mm else 0,
                        "discriminator_cycle_loss": d_cycle_loss
                    }
                    wandb.log(scalars,step=count)
                    writer.add_scalars("run", tag_scalar_dict=scalars, global_step=count)
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

                    sl, sv, c= self.test(dataset)

                    writer.add_scalar("simlex",sl,global_step=count)
                    writer.add_scalar("simverb", sv,global_step=count)
                    writer.add_scalar("card", c,global_step=count)
                    wandb.log({"simlex":sl,"card": c,"simverb":sv,"epoch":epoch},step=count)
                    writer.flush()

                    if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                        self.save_model(name="checkpoint")

                    res.append((sl, sv,c))

                    # self.combined_callback.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})
                    # wandbcb.on_epoch_end(epoch, {"simlex": sl, "simverb": sv})

                    print(res)
                    print("\n")
            else:
                epoch = 0
                running = True
                while running:
                    for batch_i, (imgs_A, imgs_B) in enumerate(dataloader):
                        if count >= iters:
                            running=False
                            break
                        # run_batch(batch_i,imgs_A,imgs_B,epoch,count,iters%len(dataloader))
                        run_batch(batch_i,imgs_A,imgs_B,epoch,count,iters/len(dataloader))
                        count += 1
                    epoch+=1
                    print("\n")
                    sl, sv, c = self.test(dataset)
                    writer.add_scalar("simlex", sl, global_step=count)
                    writer.add_scalar("simverb", sv, global_step=count)
                    writer.add_scalar("card", c, global_step=count)

                    wandb.log({"simlex":sl,"simverb":sv,"card":c},step=count)
                    writer.flush()

                    if epoch % epochs_per_checkpoint == 0 and epoch != 0:
                        self.save_model(name="checkpoint")

                    res.append((sl, sv,c))
                    print(res)
                    print("\n")
        print("Actual training")
        train_(epochs,iters=iters)
        print("Final performance")
        sl, sv,c = self.test(dataset)
        res.append((sl, sv, c))

        self.save_model(name="final")
        return res

    def test(self, dataset, simlex="testing/simlexorig999.txt", simverb="testing/simverb3500.txt",card="testing/card660.tsv",
             fasttext="fasttext_model/cc.en.300.bin",
             prefix="en_"):
        self.to("cpu")
        self.put_eval()
        sl = tools.test_sem(self.g_AB, dataset, dataset_location=simlex,
                            fast_text_location=fasttext, prefix=prefix,pt=True)[0]
        sv = tools.test_sem(self.g_AB, dataset, dataset_location=simverb,
                            fast_text_location=fasttext, prefix=prefix,pt=True)[0]
        c = tools.test_sem(self.g_AB, dataset, dataset_location=card,
                            fast_text_location=fasttext, prefix=prefix,pt=True)[0]
        self.to(self.device)
        self.put_train()
        return sl, sv,c
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

    @staticmethod
    def load_model(path,device="cpu"):
        try:
            print("Trying to load model...")
            return torch.load(path,map_location=device)
        except Exception as e:
            print(e)
            return None