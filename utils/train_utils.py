import logging
import os
import time
import warnings
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from PIL import Image

from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import wandb

import models
from datasets.ImageDatasets import GetData

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

label_dim = 6
nz = 100
SX = 64
SY = 64
N_CHANNELS = 1
EXTRINSIC_DIM = 6


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class train_utils(object):
    def __init__(self, args, save_dir):
        # save_dir为模型保存的路径
        self.args = args
        self.save_dir = save_dir
    def get_data(self, data_path, image_subpath):
        DATA_PATH = data_path
        IMAGE_SUBPATH = image_subpath
        df = pd.read_csv(os.path.join(DATA_PATH, "Dataset_descriptor.csv"), sep=",")
        # shuffling dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        Y_COLUMNS=['temperature [C]', 'Al-concentration [at.%]', 'O-concentration [at.%]',
                    'ionization degree [a.u.]', 'average ion energy [eV]', 'pressure [Pa]']
        scaler = MinMaxScaler()
        # 取我们需要用的值
        y = df[Y_COLUMNS]
        #做最大最小值归一化
        y_scaled = scaler.fit_transform(y)
        # 用tsne对标签进行降维, 默认是降到2维
        tsne = TSNE(init='pca', learning_rate='auto')
        y_tsne = tsne.fit_transform(y_scaled)

        #number of bins per each extrinsic properties
        BINS=6

        #for Y_column in Y_COLUMNS:
            # print(Y_column)
            #fig, axs= plt.subplots(1,2, figsize=(7,3))
            #im=axs[0].scatter(y_tsne[:,0], y_tsne[:,1], c= df[Y_column])
            #fig.colorbar(im, ax = axs[0])
            #h=axs[1].hist(df[Y_column], bins=BINS)
            #plt.show()
            

        binning_labels_dict = {}
        Y_COLUMNS_BINNING = []
        for Y_column in Y_COLUMNS:
            # histogram means "直方图", bins是均匀分组后的边界
            cnt, bins = np.histogram(df[Y_column], bins=BINS)
            # 为什么要对这个减1？因为后面做searchsorted时，对这个边界的索引可能出问题
            bins[0] -= 1
            col_name = Y_column + "_binning"
            # 做这个操作是为了把所有的原始数据分成6组, 用1~6作为标签
            binning_labels_dict[col_name] = np.searchsorted(bins, df[Y_column].values)
            Y_COLUMNS_BINNING.append(col_name)

        binning_df = pd.DataFrame(binning_labels_dict)
        # 给其增加一列“T”，赋值全为1,用来计数，count之后T的值就是这种标签的个数
        binning_df["T"] = 1
        # 分别按照Y_COLUMNS_BINNING中的指标分组，统计每组的个数
        # 这里按照这个列表去groupby,会自动列出所有可能的取值，然后再count，从而计算出每一类的数量，以及所有的类别，这个方法真的是特别巧妙
        count_bins_df = binning_df.groupby(Y_COLUMNS_BINNING).count()
        # 每个指标都有
        # 这里补充一个reset_index的操作，原来的代码没有这个操作导致后面出问题了
        count_bins_df = count_bins_df.reset_index()
        max_count=count_bins_df["T"].max()
        # 与原始数据连接
        df = pd.concat((df, binning_df), axis=1)
        df.drop(labels="T", axis=1, inplace=True)
        count_bins_df = count_bins_df.reset_index()

        df=pd.merge(df,count_bins_df,on=Y_COLUMNS_BINNING)

        df["weight"] = max_count/df["T"]

        # Image cropping & resizing 
        BOX_SIZE = 128
        BATCH_SIZE = 16
        CROP_PER_IMAGE = 128

        # ceil是向上取整
        batch_count = int(np.ceil(len(df)/BATCH_SIZE))

        YS = []
        WEIGHTS = []
        XS = []

        for b in range(0, batch_count):
        # print(b*BATCH_SIZE, "->", min(len(df), b*BATCH_SIZE+BATCH_SIZE)-1)
            batch_indices = np.arange(b*BATCH_SIZE,min(len(df),b*BATCH_SIZE+BATCH_SIZE))

            for ind in batch_indices:
                fname = df.loc[ind, "file name"]
                fname = os.path.join(DATA_PATH, IMAGE_SUBPATH, fname)
                im = Image.open(fname)
                for rep in range(CROP_PER_IMAGE):
                    # 剪裁
                    left = np.random.randint(0, im.width - BOX_SIZE)
                    upper = np.random.randint(0, im.height - BOX_SIZE)

                    box = left, upper, left+BOX_SIZE, upper+BOX_SIZE

                    sub_image = im.crop(box)

                    sub_image = sub_image.resize((SX, SY), resample=Image.Resampling.LANCZOS)

                    sub_image_np = np.array(sub_image)
                    sub_image_np = ((sub_image_np-127.5)/127.5).reshape(SX, SY, 1)
                    # YS存储该图片对应的information
                    YS.append(df.loc[ind, Y_COLUMNS].values)
                    # WEIGHTS，存储这个图片的权重
                    WEIGHTS.append(df.loc[ind, "weight"])
                    # XS存储图片本身
                    XS.append(sub_image_np)

        YS = np.array(YS)
        XS = np.array(XS)
        WEIGHTS = np.array(WEIGHTS)
        WEIGHTS_SUM = WEIGHTS.sum()

        # 对标签信息做归一化
        YS_scaled = scaler.fit_transform(YS)
        XS = XS.reshape(-1, 1, 64, 64)

        return XS, YS, YS_scaled, WEIGHTS, WEIGHTS_SUM

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # 求设备的gpu数量
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))

        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))
    
        self.XS, self.YS, self.YS_scaled, self.WEIGHTS, self.WEIGHTS_SUM = self.get_data(args.data_path, args.image_subpath)

        if args.train:
            self.dataset = GetData(self.XS, self.YS_scaled, self.WEIGHTS, self.device)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        # Define the model
        fmodel = getattr(models, args.model_name)

        self.G = getattr(fmodel, 'Generator')()
        self.G.apply(weights_init)

        self.D = getattr(fmodel, 'Discriminator')()
        self.D.apply(weights_init)
        # Define the optimizer
        if args.train:
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr= args.lr, betas=(args.beta1, 0.999))#, betas=(beta1, 0.999))
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr= args.lr, betas=(args.beta1, 0.999))#, betas=(beta1, 0.999))
        # Invert the model and define the loss
        self.G.to(self.device)
        self.D.to(self.device)
        # Loss function
        self.criterion = torch.nn.BCELoss(reduction='none')

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, latent_dim)
        # generate labels
        labels = np.random.choice(len(self.YS),size=n_samples)
        weights = self.WEIGHTS[labels]
        labels = self.YS_scaled[labels]    
        return torch.from_numpy(z_input).to(self.device), torch.from_numpy(labels).to(self.device), torch.from_numpy(weights).to(self.device)

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        z_input, labels_input, weights_input = self.generate_latent_points(nz, n_samples)
        # predict outputs
        images = self.G(z_input.float(), labels_input.float())
        # create class labels
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y, weights_input

    def create_randuni_process_vector(self, num_samples, set_zero,addition):
        process_vector = np.concatenate([
            np.reshape(np.random.uniform(20,600,num_samples),(num_samples,1)),
            np.reshape(np.random.uniform(0,70,num_samples),(num_samples,1)),
            np.reshape(np.random.uniform(0,10,num_samples),(num_samples,1)),
            np.reshape(np.random.uniform(0.1,1.2,num_samples),(num_samples,1)),
            np.reshape(np.random.uniform(1,200,num_samples),(num_samples,1)),
            np.reshape(np.random.uniform(0.5,1,num_samples),(num_samples,1))
        ],axis=1)
        process_vector = np.reshape(process_vector,(num_samples,6))
        process_vector[:,set_zero]=0
        if addition is not None and len(addition)>0:
            process_vector[:,:]=process_vector[:,:]+addition
        scaler = MinMaxScaler()
        sc = scaler.fit(process_vector)
        process_vector = sc.transform(process_vector)

        return process_vector

    def gSZD_imscatter(self, x, y,img, ax,zoom):
        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            
            image = img[i].reshape(self.plen, self.plen)
            # image = cv2.cvtColor(image*255,cv2.COLOR_GRAY2RGB)
            # Note: OpenCV uses BGR and plt uses RGB
            image = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))
    
            ax.update_datalim(np.column_stack([x, y]))
            ax.autoscale()


    def train_cGAN(self, batch_size):
        for i, (data, label, weight) in tqdm(enumerate(self.dataloader)):

            '''
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Training Images")
            plt.imshow(np.transpose(vutils.make_grid(data, padding=2).cpu(),(1,2,0)))
            plt.show()
            '''
    
            ## Train with all-real batch        
            self.D_opt.zero_grad()

            # 真实数据 
            x_real = data.to(self.device).float()
            y_real = torch.ones(batch_size, ).to(self.device)
            # label_onehot = onehot[label]
            label = label.to(self.device).float()
            y_real_predict = self.D(x_real, label).squeeze()      
            d_real_loss = self.criterion(y_real_predict, y_real)
            d_real_loss = d_real_loss * weight / self.WEIGHTS_SUM
            wandb.log({'d_real_loss':d_real_loss.mean()})
            d_real_loss.mean().backward()

            ## Train with all-fake batch

            # noise = torch.randn(batch_size, nz, 1, 1, device = device)
            # noise_label = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
            #print(noise_label)
            # noise_label_onehot = onehot[noise_label].to(device)  # Genera label in modo casuale (-1,)
            # x_fake = G(noise, noise_label_onehot)       #Genera immagini false
            # y_fake = torch.zeros(batch_size, ).to(device)    # Assegna label 0
            # y_fake_predict = D(x_fake, noise_label_onehot).squeeze()

            # 生成fake数据
            [x_fake, noise_label], y_fake, weight_fake= self.generate_fake_samples(batch_size)
            y_fake_predict = self.D(x_fake.float(), noise_label.float()).squeeze()
            y_fake = torch.from_numpy(y_fake).reshape(-1).float().to(self.device)
            d_fake_loss = self.criterion(y_fake_predict, y_fake)
            d_fake_loss = d_fake_loss * weight_fake / self.WEIGHTS_SUM
            wandb.log({'d_fake_loss':d_fake_loss.mean()})
            d_fake_loss.mean().backward()
            self.D_opt.step()
         
            # (2) Update G network: maximize log(D(G(z)))         
            self.G_opt.zero_grad()
         
            #noise = torch.randn(batch_size, z_dim, 1, 1, device = device)
            #noise_label = (torch.rand(batch_size, 1) * label_dim).type(torch.LongTensor).squeeze()
            #noise_label_onehot = onehot[noise_label].to(device)   # Genera label in modo casuale (-1,)
            # x_fake = G(noise, noise_label_onehot)
            [x_fake, noise_label], y_fake, weight_fake= self.generate_fake_samples(batch_size)
            y_fake = torch.ones(batch_size, ).to(self.device)    # Il y_fake qui è lo stesso di y_real sopra, entrambi sono 1
            y_fake_predict = self.D(x_fake.float(), noise_label.float()).squeeze()
            g_loss = self.criterion(y_fake_predict, y_real)    # Usa direttamente y_real per essere più intuitivo
            g_loss = g_loss * weight_fake / self.WEIGHTS_SUM
            wandb.log({'g_loss':g_loss.mean()})
            g_loss.mean().backward()
            self.G_opt.step()

            err_D = d_fake_loss.mean().item() + d_real_loss.mean().item()
            err_G = g_loss.mean().item()
            wandb.log({'err_D':err_D, 'err_G':err_G})
            '''
            if i%50 == 0:
                with torch.no_grad():
                    out_imgs = G(fixed_noise.to(device), fixed_label.to(device))
                save_image(out_imgs,f"{PATH}{i}.png", nrow = 10) #aggiungi percorso: "path/iterazione_classe.png" es "pippo/20000_3.png"
            '''
        return err_D, err_G


    def train(self):
        args = self.args

        wandb.config = {"learning_rateD":args.lr,
                        "learning_rateG":args.lr,
                        "epochs":args.epochs,
                        "batch_size":args.batch_size
                        }
        D_loss = []
        G_loss = []
        for epoch in tqdm(range(args.epochs)):
            if epoch % 10 == 0:
                self.G_opt.param_groups[0]['lr'] /= 2
                self.D_opt.param_groups[0]['lr'] /= 2
        
            # training
            err_D, err_G = self.train_cGAN(args.batch_size)
            logging.info('Epoch: [{}/{}], Err_D: {:.4f} Err_G: {:.4f},'.format(epoch, args.epochs, err_D, err_G))
            D_loss.append(err_D)
            G_loss.append(err_G)

        wandb.finish()
        sub_dir = args.model_name + '_' + args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        save_dir = os.path.join(args.checkpoint_dir, sub_dir)
        torch.save(self.G.state_dict(), './outputs/'+save_dir+'.pt')

    def evaluate(self):
        args = self.args
        model = torch.load(args.model_dir, map_location='cpu')
        self.G.load_state_dict(model)
        self.G.eval()
        scaler = MinMaxScaler()
        plt.gray()
        num_pred = 3000
        self.plen = 64

        latent_points, labels, weights = self.generate_latent_points(nz, num_pred)
        del labels

        add_ = [0,0,1.0,1.0,40,0.5]
        pp = self.create_randuni_process_vector(num_pred,[2,3,4,5],add_)
        with torch.no_grad():
            X_test  = self.G(torch.as_tensor(latent_points, dtype=torch.float32), torch.as_tensor(pp, dtype=torch.float32))
        # scale from [-1,1] to [0,1]
        X_test = (X_test + 1) / 2.0
        X_test = X_test.numpy() 
        sc = scaler.fit(pp)
        pp=sc.inverse_transform(pp)
        if not os.path.exists('eval_images'):
            os.makedirs('eval_images')

        fig, ax = plt.subplots(figsize=(16, 12))
        self.gSZD_imscatter(pp[:,0],pp[:,1],X_test,ax,.5)
        plt.ylabel('Al-concentration [at.%]',fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.xlabel('deposition temperature [°C]',fontsize=20)
        plt.savefig(args.save_dir+'/'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')+'.jpg')
        plt.show()
