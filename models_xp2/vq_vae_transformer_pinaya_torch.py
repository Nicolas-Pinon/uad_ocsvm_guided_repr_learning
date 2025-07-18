import os
import shutil
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from cv2 import resize, INTER_NEAREST
from sklearn.cluster import k_means
from hilbertcurve.hilbertcurve import HilbertCurve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class PinayaTransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, seq_length, dropout):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(vocab_size, seq_length, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoder(d_model, num_heads, dff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.losses_name = ['loss', 'accuracy', 'top 5 accuracy', "unique true", "unique predictions", "perplexity"]

    def forward(self, inputs):
        x = self.embed(inputs)
        x = self.embed_drop(x)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.dropout(x)
        output = self.final_layer(x)
        return output

    def train_step(self, batch, optimizer):
        x_train, y_train = batch
        optimizer.zero_grad()
        
        y_pred = self(x_train)
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_train.view(-1))
        
        # Calculate accuracy metrics
        with torch.no_grad():
            _, predicted = torch.max(y_pred, -1)
            correct = (predicted == y_train).float()
            acc = correct.mean()
            
            # Top-5 accuracy
            _, top5_pred = y_pred.topk(5, -1)
            top5_correct = top5_pred.eq(y_train.unsqueeze(-1)).any(-1).float()
            top5_acc = top5_correct.mean()
            
            # Perplexity
            perpl = torch.exp(loss)
            
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": acc.item(),
            "top 5 accuracy": top5_acc.item(),
            "perplexity": perpl.item()
        }

    def test_step(self, batch):
        x_train, y_train = batch
        with torch.no_grad():
            y_pred = self(x_train)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_train.view(-1))
            
            # Calculate metrics
            _, predicted = torch.max(y_pred, -1)
            correct = (predicted == y_train).float()
            acc = correct.mean()
            
            # Top-5 accuracy
            _, top5_pred = y_pred.topk(5, -1)
            top5_correct = top5_pred.eq(y_train.unsqueeze(-1)).any(-1).float()
            top5_acc = top5_correct.mean()
            
            # Perplexity
            perpl = torch.exp(loss)
            
        return {
            "loss": loss.item(),
            "accuracy": acc.item(),
            "top 5 accuracy": top5_acc.item(),
            "perplexity": perpl.item()
        }


class ImageRestorer:
    def __init__(self, transformer, vqvae, sequence_generator, thr):
        self.transformer = transformer
        self.vqvae = vqvae
        self.sequence_generator = sequence_generator
        self.seq_len = self.sequence_generator.seq_len
        self.latent_dims = self.sequence_generator.latent_dims
        self.thr = thr
        print('THRESHOLD', self.thr)

    def __call__(self, x, batch_size=32, return_reconstructions=True):
        warnings.warn("entered __call__ from image restorer")
        restored_images, up_resampling_maks = self.restore_images(x, batch_size=batch_size)
        warnings.warn("finished restorating and resampling masks")
        
        if return_reconstructions:
            with torch.no_grad():
                recons = self.vqvae(x)
            warnings.warn("finished recons")
            return restored_images, up_resampling_maks, recons
        else:
            return restored_images, up_resampling_maks

    def restore_images(self, data, batch_size=32):
        discrete_latent_indexes = self._get_latent_indexes(data, batch_size=batch_size)
        restored_sequences, resampling_masks = self.restore_sequences(discrete_latent_indexes, batch_size=batch_size)
        
        resampling_masks = self.sequence_generator.reverse_process_features(resampling_masks)
        restored_sequences = self.sequence_generator.reverse_process_features(restored_sequences)
        
        up_resampling_masks = np.zeros(data.shape[:3])  # (B,X,Y)
        for k in range(resampling_masks.shape[0]):
            up_resampling_masks[k] = gaussian_filter(
                resize(resampling_masks[k], data.shape[1:3], interpolation=INTER_NEAREST), 
                sigma=5
            )
            
        with torch.no_grad():
            restored_images = self.vqvae.decoder(self.vqvae.get_encodings_from_indices(restored_sequences))

        return restored_images, up_resampling_masks

    def restore_sequences(self, seq, batch_size=32):
        flat_seq = np.hstack((np.zeros((seq.shape[0], 1)), self.sequence_generator.process_features(seq)))
        preds_inds = flat_seq[:, 1:]
        
        with torch.no_grad():
            logits = self.transformer(torch.from_numpy(flat_seq[:, :-1]).long())
            probas = F.softmax(logits, dim=-1).numpy()
            
        resampling_masks = np.zeros((seq.shape[0], self.seq_len))
        restored_sequences = np.copy(flat_seq[:, 1:])
        
        for i in range(resampling_masks.shape[0]):
            for j in range(resampling_masks.shape[1]):
                if probas[i, j, int(preds_inds[i, j])] < self.thr:
                    resampling_masks[i, j] = 1.
                    new = 0
                    while new == 0:
                        new = np.random.choice(np.arange(self.vqvae.num_embeddings + 1), p=probas[i, j])
                        restored_sequences[i, j] = new
                        
        return restored_sequences - 1, resampling_masks

    def _get_latent_indexes(self, images, batch_size=32):
        latent_indexes = np.zeros((images.shape[0], *self.latent_dims))
        for k in range(int(np.ceil(images.shape[0] / batch_size))):
            batch = images[k * batch_size:(k + 1) * batch_size]
            latent_indexes[k * batch_size:(k + 1) * batch_size] = (
                self.vqvae.get_embeds_indices_from_images(batch) + 1
            )
        return latent_indexes

    def _softmax(self, x, axis=None):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)


class EMAWarmUpCallback:
    def __init__(self, decay_start, decay_end=None, steps=None):
        self.decay_start = decay_start
        if decay_start is not None and steps is not None:
            self.decay_step = (decay_end - decay_start)/steps
        self.n_steps = steps if steps is not None else 0
        self.current_decay = decay_start

    def on_train_begin(self, model):
        model.vq.vector_quantizer.set_decay(self.decay_start)

    def on_epoch_end(self, epoch, model):
        if epoch < self.n_steps:
            self.current_decay += self.decay_step
            model.vq.vector_quantizer.set_decay(self.current_decay)


sequences_orderings = {
    # Same as original
    0:{'ordering': 'raster', 'flip': 0, 'rotate':False},
    # ... (keep all the same ordering configurations)
    24:{'ordering': 'random', 'flip': 0, 'rotate':False}
}


class Encoder2DPinaya(nn.Module):
    def __init__(self, filters=256, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size, stride, padding="same"),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride, padding="same"),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride, padding="same"),
            nn.ReLU()
        )
        self.residuals = nn.Sequential(*[ResidualBlock(filters) for _ in range(n_residuals)])

    def forward(self, x):
        return self.residuals(self.convs(x))


class Decoder2DPinaya(nn.Module):
    def __init__(self, nb_channels, filters=256, kernel_size=(4, 4), stride=(2, 2), activation='relu', n_residuals=3):
        super().__init__()
        self.residuals = nn.Sequential(*[ResidualBlock(filters) for _ in range(n_residuals)])
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(filters, filters, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filters, filters, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filters, filters, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, nb_channels, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(self.residuals(x))


class VQVAEGeneric(nn.Module):
    def __init__(self, nb_channels, nb_filters=(8, 12, 16), num_embeddings=512, patch_size=(63, 63), type_="zara", name="vqvae",
                 latent_dim=None, dropout=0, GRN_kernel=(3, 3), apply_batch_norm=True,
                 codebook_learning="loss", reservoir_size=1024, reestimate_step_iter=100, epoch_start=2, num_epoch_train=2, beta=0.25, compression="high",
                 preprocessing_layer=True):
        super().__init__()
        self.nb_channels = nb_channels
        if (type_ == "gatedresnet" and latent_dim is not None):
            self.latent_dim = latent_dim
        elif isinstance(nb_filters, int):
            self.latent_dim = nb_filters
        elif isinstance(nb_filters, list):
            self.latent_dim = nb_filters[-1]
            
        self.num_embeddings = num_embeddings
        self.codebook_learning = codebook_learning
        self.reservoir_size = reservoir_size
        self.reestimate_step_iter = reestimate_step_iter
        self.epoch_start = epoch_start
        self.num_epoch_train = num_epoch_train
        self.beta = beta
        self.preprocessing = preprocessing_layer
        
        if self.preprocessing:
            self.preprocess_layer = nn.Sequential(
                RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
                RandomBrightness((-0.1, 0.1), (0., 1.)),
                RandomContrast(0.1)
            )
            
        self.encoder = Encoder2DPinaya(filters=nb_filters if isinstance(nb_filters, int) else nb_filters[-1])
        self.decoder = Decoder2DPinaya(nb_channels=nb_channels, filters=nb_filters if isinstance(nb_filters, int) else nb_filters[-1])
        self.vq = BottleneckVQ(self.num_embeddings, self.latent_dim, codebook_learning=self.codebook_learning, 
                             reservoir_size=self.reservoir_size, reestimate_step_iter=self.reestimate_step_iter, 
                             epoch_start=self.epoch_start, num_epoch_train=self.num_epoch_train)

    def forward(self, x, training=False):
        if training and self.preprocessing:
            x = self.preprocess_layer(x)
            
        out_enc = self.encoder(x)
        encodings, out_vq = self.vq(out_enc, training=training)
        out_dec = self.decoder(out_vq)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(out_dec, x)
        
        # Perplexity calculation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {
            'output': out_dec,
            'reconstruction_loss': reconstruction_loss,
            'perplexity': perplexity,
            'encodings': encodings
        }

    def get_vq_usage(self, x):
        out_enc = self.encoder(x)
        encodings = self.vq(out_enc)[0]
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return {
            'encodings': encodings,
            'perplexity': perplexity
        }

    def get_embeds_indices_from_images(self, x_batch, apply_preprocessing=True, batch_size=32):
        indices = []
        with torch.no_grad():
            for i in range(0, x_batch.size(0), batch_size):
                batch = x_batch[i:i+batch_size]
                if apply_preprocessing:
                    batch = self.preprocess_layer(batch)
                out_enc = self.encoder(batch)
                flattened = out_enc.view(-1, self.latent_dim)
                batch_indices = self.vq.vector_quantizer.get_code_indices(flattened)
                indices.append(batch_indices.view(out_enc.shape[:-1]).cpu())
        return torch.cat(indices, dim=0).numpy()

    def get_encodings_from_indices(self, encoding_indices):
        encodings = F.one_hot(torch.from_numpy(encoding_indices).long(), self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.vq.vector_quantizer.embeddings.t())
        return quantized

    def train_step(self, x_batch, optimizer):
        optimizer.zero_grad()
        
        outputs = self(x_batch, training=True)
        reconstruction_loss = outputs['reconstruction_loss']
        perplexity = outputs['perplexity']
        
        if self.codebook_learning == "loss":
            codebook_loss = self.vq.vector_quantizer.codebook_loss
            commitment_loss = self.vq.vector_quantizer.commitment_loss
            total_loss = reconstruction_loss + self.beta * codebook_loss + commitment_loss
        else:
            commitment_loss = self.vq.vector_quantizer.commitment_loss
            total_loss = reconstruction_loss + commitment_loss
        
        total_loss.backward()
        optimizer.step()
        
        res = {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "perplexity": perplexity.item()
        }
        
        if self.codebook_learning == "loss":
            res["codebook_loss"] = codebook_loss.item()
            
        return res

    def test_step(self, x_batch):
        with torch.no_grad():
            outputs = self(x_batch, training=False)
            reconstruction_loss = outputs['reconstruction_loss']
            perplexity = outputs['perplexity']
            
            if self.codebook_learning == "loss":
                codebook_loss = self.vq.vector_quantizer.codebook_loss
                commitment_loss = self.vq.vector_quantizer.commitment_loss
                total_loss = reconstruction_loss + codebook_loss + commitment_loss
            else:
                commitment_loss = self.vq.vector_quantizer.commitment_loss
                total_loss = reconstruction_loss + commitment_loss
                
            res = {
                "total_loss": total_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "commitment_loss": commitment_loss.item(),
                "perplexity": perplexity.item()
            }
            
            if self.codebook_learning == "loss":
                res["codebook_loss"] = codebook_loss.item()
                
            return res


class VQVAEPinaya(VQVAEGeneric):
    def __init__(self, nb_channels, nb_filters=256, num_embeddings=32, type_="pinaya", name="vqvae", codebook_learning="ema", **kwargs):
        super().__init__(nb_channels, nb_filters=nb_filters, num_embeddings=num_embeddings, type_=type_, name=name, codebook_learning=codebook_learning, **kwargs)


class SequenceGenerator(Dataset):
    def __init__(self, data=None, model=None, ordering=None, preprocessing=True, batch_size=16, seed=15):
        super().__init__()
        self.model = model
        self.number = data.shape[0]
        self.preprocessing = preprocessing
        
        if model is None:
            self.sequence_data = data
            self.latent_dims = data.shape[1:3]
        else:
            self.data = torch.from_numpy(data).float()
            self.sequence_data = self._get_sequences_data_from_images()
            self.latent_dims = self.sequence_data.shape[1:3]
            
        self.batch_size = batch_size
        self.nb_samples = data.shape[0]
        self.seed = seed if isinstance(seed, int) else torch.randint(0, 1000000, (1,)).item()
        self.splits = [self.batch_size*k for k in range(1, self.number//self.batch_size)]
        self.seq_len = self.latent_dims[0] * self.latent_dims[1]
        
        self._get_ordering_config(ordering)
        self._get_ordering_path(self.order)
        self._generate_indexes()

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        sequences = self.sequence_data[self.indexes[index]]
        sequences = self.process_features(sequences)
        return torch.cat((torch.zeros((self.batch_size, 1)), sequences[:,:-1]), sequences

    def on_epoch_end(self):
        self.seed += 12345
        self._generate_indexes()
        if self.preprocessing:
            self.sequence_data = self._get_sequences_data_from_images()

    def _get_sequences_data_from_images(self):
        with torch.no_grad():
            indices = self.model.get_embeds_indices_from_images(self.data, apply_preprocessing=self.preprocessing)
        return indices + 1

    # ... (keep all other SequenceGenerator methods the same, just replace tf/np operations with torch equivalents)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, training_mode="loss", decay=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.training_mode = training_mode
        self.decay = decay
        self.epsilon = 1e-5
        
        # Initialize embeddings
        self.embeddings = nn.Parameter(
            torch.randn(embedding_dim, num_embeddings),
            requires_grad=training_mode == "loss"
        )
        
        if training_mode == "ema":
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_dw', torch.zeros_like(self.embeddings))
            self.register_buffer('counter', torch.tensor(0))
            
        self.codebook_loss = 0
        self.commitment_loss = 0

    def forward(self, x, training=False):
        # Quantization
        encoding_indices = self.get_code_indices(x)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embeddings.t())
        
        # Calculate losses
        if self.training_mode == "loss":
            self.codebook_loss = F.mse_loss(quantized, x.detach())
        elif self.training_mode == "ema" and training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, dim=0)
            dw = torch.matmul(x.t(), encodings)
            self.ema_dw = self.ema_dw * self.decay + (1 - self.decay) * dw
            
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / 
                                    (n + self.num_embeddings * self.epsilon) * n)
            
            normalised_updated_ema_w = self.ema_dw / self.ema_cluster_size.unsqueeze(0)
            self.embeddings.data = normalised_updated_ema_w
            
        self.commitment_loss = F.mse_loss(quantized.detach(), x)
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return encodings, quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate distances
        similarity = torch.matmul(flattened_inputs, self.embeddings)
        distances = (
            torch.sum(flattened_inputs**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings**2, dim=0) -
            2 * similarity
        )
        
        # Derive indices
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def set_decay(self, decay):
        self.decay = decay


class BottleneckVQ(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, codebook_learning="ema", reservoir_size=1024, 
                 reestimate_step_iter=1024, epoch_start=2, num_epoch_train=5, name="vq"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.reestimate_step_iter = reestimate_step_iter
        self.epoch_start = epoch_start
        self.last_epoch_train = epoch_start + num_epoch_train
        self.reservoir_size = reservoir_size
        
        self.current_iter = 0
        self.current_epoch = 0
        
        assert codebook_learning in ("kmeans", "loss", "ema")
        if codebook_learning == "kmeans":
            self.reservoir = Reservoir(max_samples=self.reservoir_size)
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="kmeans")
        elif codebook_learning == "loss":
            self.reservoir = None
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="loss")
        elif codebook_learning == "ema":
            self.reservoir = None
            self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, training_mode="ema")

    def reestimate(self):
        if self.reservoir is None or self.current_epoch < self.epoch_start:
            return
            
        if self.current_epoch >= self.last_epoch_train:
            self.reservoir = None
            return
            
        samples_reservoir = self.reservoir.content()
        if samples_reservoir is None or samples_reservoir.shape[0] < self.num_embeddings or self.current_iter < self.reestimate_step_iter:
            return
            
        encodings, *_ = k_means(samples_reservoir, n_clusters=self.num_embeddings)
        print('\nUpdating codebook vectors...')
        self.vector_quantizer.embeddings.data = torch.from_numpy(encodings.T).float()
        self.reservoir.reset()
        self.current_iter = 0

    def forward(self, batch, training=False):
        self.reestimate()
        input_shape = batch.shape
        flattened = batch.view(-1, self.embedding_dim)
        
        if self.reservoir is not None:
            if training:
                self.reservoir.add(flattened.detach().cpu().numpy())
                
        if training and self.reservoir is not None and self.current_epoch < self.epoch_start:
            encodings = torch.zeros(input_shape)
            quantized = batch
        else:
            encodings, quantized = self.vector_quantizer(flattened, training=training)
            
        out_vq = quantized.view(input_shape)
        return encodings, out_vq


class ResidualBlock(nn.Module):
    def __init__(self, filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return residual + x


class RandomBrightness(nn.Module):
    def __init__(self, factor, value_range):
        super().__init__()
        self.factor = sorted(factor)
        self.value_range = sorted(value_range)

    def forward(self, batch, training=False):
        if training:
            br = torch.empty(batch.shape[0], *[1]*len(batch.shape[1:]), 
                          device=batch.device).uniform_(*self.factor)
            batch = batch + br
            return torch.clamp(batch, *self.value_range)
        return batch


class RandomContrast(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, training=False):
        if training:
            mean = x.mean(dim=(1,2,3), keepdim=True)
            factor = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
                1-self.factor, 1+self.factor
            )
            return (x - mean) * factor + mean
        return x


class RandomTranslation(nn.Module):
    def __init__(self, height_factor, width_factor):
        super().__init__()
        self.height_factor = height_factor
        self.width_factor = width_factor

    def forward(self, x, training=False):
        if training:
            batch_size, _, height, width = x.shape
            tx = torch.empty(batch_size, device=x.device).uniform_(*self.width_factor) * width
            ty = torch.empty(batch_size, device=x.device).uniform_(*self.height_factor) * height
            
            grid = self._get_translation_grid(batch_size, height, width, tx, ty, x.device)
            return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        return x

    def _get_translation_grid(self, batch_size, height, width, tx, ty, device):
        # Create meshgrid
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)
        xx, yy = torch.meshgrid(x, y)
        
        # Add translation
        xx = xx.unsqueeze(0) - 2 * tx.reshape(-1, 1, 1) / width
        yy = yy.unsqueeze(0) - 2 * ty.reshape(-1, 1, 1) / height
        
        # Stack coordinates
        grid = torch.stack((xx, yy), dim=-1)
        return grid


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_length, embedding_dim):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_length, embedding_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.token_embed(x) + self.pos_embed(positions)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x