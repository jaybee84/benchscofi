#coding:utf-8

## source: https://raw.githubusercontent.com/jilljenn/vae/1d7f09af3bcaebfc5d8fa8cc18033d8bb8ca19bc/vfm-torch.py

import torch
from torch import nn, distributions
import numpy as np

#LINK = nn.functional.softplus
LINK = torch.abs

class CF(nn.Module):
    """
    Recommender system
    """
    def __init__(self, embedding_size, N_VARIATIONAL_SAMPLES, N=1, M=1, output='reg'): #N=nusers, M=nitems
        super().__init__()
        self.N, self.M = N, M
        self.embedding_size = embedding_size
        self.N_VARIATIONAL_SAMPLES = N_VARIATIONAL_SAMPLES
        self.output = output
        self.alpha = nn.Parameter(torch.Tensor([1e9]), requires_grad=True)
        self.global_bias_mean = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.global_bias_scale = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_global_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_item_bias_prior = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.prec_user_entity_prior = nn.Parameter(torch.ones(self.embedding_size), requires_grad=True)
        self.prec_item_entity_prior = nn.Parameter(torch.ones(self.embedding_size), requires_grad=True)
        # self.alpha = torch.Tensor([1.])
        nn.init.uniform_(self.alpha)
        
        # bias_init = torch.cat((torch.randn(self.N + self.M, 1), torch.ones(self.N + self.M, 1) * (0.02 ** 0.5)), axis=1)
        # entity_init = torch.cat((
        #     torch.randn(self.N + self.M, self.embedding_size),
        #     torch.ones(self.N + self.M, self.embedding_size) * (0.02 ** 0.5),
        # ), axis=1)
        self.bias_params = nn.Embedding(self.N + self.M, 2)#.from_pretrained(bias_init)  # w
        self.entity_params = nn.Embedding(self.N + self.M, 2 * self.embedding_size)#.from_pretrained(entity_init)  # V

        self.saved_global_biases = []
        self.saved_mean_biases = []
        self.saved_mean_entities = []
        self.mean_saved_global_biases = None
        self.mean_saved_mean_biases = None
        self.mean_saved_mean_entities = None

        self.global_bias_prior = distributions.normal.Normal(0, 1)
        self.bias_prior = distributions.normal.Normal(0, 1)
        self.entity_prior = distributions.normal.Normal(0, 1)
        #     torch.zeros(self.N + self.M),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_bias_prior.repeat(N),
        #         self.prec_item_bias_prior.repeat(M)
        #     )))
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     torch.zeros(self.N + self.M, self.embedding_size),
        #     torch.nn.functional.softplus(torch.cat((
        #         self.prec_user_entity_prior.repeat(N, 1),
        #         self.prec_item_entity_prior.repeat(M, 1)
        #     )))
        # )

    def save_weights(self):
        self.saved_global_biases.append(self.global_bias_mean.detach().numpy().copy())
        self.saved_mean_biases.append(self.bias_params.weight[:, 0].detach().numpy().copy())
        self.saved_mean_entities.append(self.entity_params.weight[:, :self.embedding_size].detach().numpy().copy())
        self.mean_saved_global_biases = np.array(self.saved_global_biases).mean(axis=0)
        self.mean_saved_mean_biases = np.array(self.saved_mean_biases).mean(axis=0)
        self.mean_saved_mean_entities = np.array(self.saved_mean_entities).mean(axis=0)
        # print('size of saved', np.array(self.saved_mean_entities).shape)
        # print('test', np.array(self.saved_mean_biases)[:3, 0])

    def forward(self, x):
        # print(("x",np.isnan(x.detach().numpy()).any()))
        uniq_entities, entity_pos, nb_occ_in_batch = torch.unique(x, return_inverse=True, return_counts=True)
        # print(("uniq_entities",np.isnan(uniq_entities.detach().numpy()).any()))
        uniq_users, nb_occ_user_in_batch = torch.unique(x[:, 0], return_counts=True)
        uniq_items, nb_occ_item_in_batch = torch.unique(x[:, 1], return_counts=True)
        # nb_uniq_users = len(uniq_users)
        # nb_uniq_items = len(uniq_items)
        # print('uniq', uniq_entities.shape, 'pos', entity_pos.shape)

        # self.global_bias_prior = distributions.normal.Normal(
        #     torch.Tensor([0.]), torch.nn.functional.softplus(self.prec_global_bias_prior))
        # Global bias
        global_bias_sampler = distributions.normal.Normal(
            self.global_bias_mean,
            LINK(self.global_bias_scale)
        )
        # Biases and entities
        bias_batch = self.bias_params(x)
        entity_batch = self.entity_params(x)
        uniq_bias_batch = self.bias_params(uniq_entities)#.reshape(-1, 2)
        uniq_entity_batch = self.entity_params(uniq_entities)#.reshape(-1, 2 * self.embedding_size)
        uniq_entity_batch = torch.nan_to_num(uniq_entity_batch, nan=1e-6) ###
        # print(("uniq_entity_batch",np.isnan(uniq_entity_batch.detach().numpy()).any()))
        # print('first', bias_batch.shape, entity_batch.shape)
        # print('samplers', uniq_bias_batch.shape, uniq_entity_batch.shape)
        # scale_bias = torch.ones_like(scale_bias) * 1e-6
        bias_sampler = distributions.normal.Normal(
            uniq_bias_batch[:, 0],
            LINK(uniq_bias_batch[:, 1])
        )
        # user_bias_posterior = distributions.normal.Normal(
        #     bias_batch[:, :, 0],
        #     LINK(bias_batch[:, :, 1])
        # )
        # diag_scale_entity = nn.functional.softplus(entity_batch[:, self.embedding_size:])
        # diag_scale_entity = torch.ones_like(diag_scale_entity) * 1e-6
        # print('scale entity', entity_batch.shape, scale_entity.shape)
        # print(("uniq_entity_batch2",np.isnan(uniq_entity_batch.detach().numpy()).any()))
        # print(("LINK uniq_entity_batch",np.isnan(LINK(uniq_entity_batch[:, self.embedding_size:]).detach().numpy()).any()))
        entity_sampler = distributions.normal.Normal(
            loc=uniq_entity_batch[:, :self.embedding_size],
            scale=LINK(uniq_entity_batch[:, self.embedding_size:])
        )
        # entity_posterior = distributions.normal.Normal(
        #     loc=entity_batch[:, :, :self.embedding_size],
        #     scale=LINK(entity_batch[:, :, self.embedding_size:])
        # )
        # self.entity_prior = distributions.normal.Normal(
        #     loc=torch.zeros_like(entity_batch[:, :, :self.embedding_size]),
        #     scale=torch.ones_like(entity_batch[:, :, :self.embedding_size])
        # )

        # print('batch shapes', entity_sampler.batch_shape, self.entity_prior.batch_shape)
        # print('event shapes', entity_sampler.event_shape, self.entity_prior.event_shape)
        global_bias = global_bias_sampler.rsample((self.N_VARIATIONAL_SAMPLES,))
        biases = bias_sampler.rsample((self.N_VARIATIONAL_SAMPLES,))#.reshape(
            # self.N_VARIATIONAL_SAMPLES, -1, 2)
        entities = entity_sampler.rsample((self.N_VARIATIONAL_SAMPLES,))#.reshape(
            # self.N_VARIATIONAL_SAMPLES, -1, 2, self.embedding_size)  # N_VAR_SAMPLES x BATCH_SIZE x 2 (user, item) x self.embedding_size
        # print('hola', biases.shape, entities.shape)
        sum_users_items_biases = biases[:, entity_pos].sum(axis=2).mean(axis=0).squeeze()
        users_items_emb = entities[:, entity_pos].prod(axis=2).sum(axis=2).mean(axis=0)
        # print('final', sum_users_items_biases.shape, users_items_emb.shape)

        if self.mean_saved_mean_biases is not None:
            last_global_bias = self.saved_global_biases[-1]
            last_bias_term = self.saved_mean_biases[-1][x].sum(axis=1).squeeze()
            last_embed_term = self.saved_mean_entities[-1][x].prod(axis=1).sum(axis=1)

            mean_global_bias = self.mean_saved_global_biases
            mean_bias_term = self.mean_saved_mean_biases[x].sum(axis=1).squeeze()
            mean_embed_term = self.mean_saved_mean_entities[x].prod(axis=1).sum(axis=1)
            # print(self.mean_saved_mean_biases[x].shape, mean_bias_term.shape)
            # print(self.mean_saved_mean_entities[x].shape, mean_embed_term.shape)
            last_logits = last_global_bias + last_bias_term + last_embed_term
            mean_logits = mean_global_bias + mean_bias_term + mean_embed_term #  + 
        else:
            last_logits = None
            mean_logits = None

        std_dev = torch.sqrt(1 / LINK(self.alpha))
        unscaled_pred = global_bias + sum_users_items_biases + users_items_emb

        if self.output == 'reg':
            likelihood = distributions.normal.Normal(unscaled_pred, std_dev)
        else:
            likelihood = distributions.bernoulli.Bernoulli(logits=unscaled_pred)
        # print('global bias sampler', global_bias_sampler)
        # print('global bias prior', self.global_bias_prior)
        # print('bias sampler', bias_sampler)
        # print('bias prior', self.bias_prior)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        # a = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # b = distributions.normal.Normal(torch.zeros(2, 3), torch.ones(2, 3))
        # print('oh hey', distributions.kl.kl_divergence(a, b))
        # print('oh hey', distributions.kl.kl_divergence(entity_sampler, entity_sampler))
        # print('oh hiya', distributions.kl.kl_divergence(entity_sampler, self.entity_prior))
        # print('oh hey', distributions.kl.kl_divergence(self.entity_prior, self.entity_prior))

        # print(
        #     distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior).shape,
        #     distributions.kl.kl_divergence(bias_sampler, self.bias_prior).sum(axis=1).shape,
        #     distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=[1, 2]).shape#.sum(axis=3).shape,#.sum(axis=2).shape
        # )

        kl_bias = distributions.kl.kl_divergence(bias_sampler, self.bias_prior)
        # print(('kl_bias', kl_bias.shape, kl_bias))
        # print('kl bias', kl_bias.shape)
        # print('bias sampler', bias_sampler)
        # print('entity sampler', entity_sampler)
        # print('entity prior', self.entity_prior)
        kl_entity = distributions.kl.kl_divergence(entity_sampler, self.entity_prior).sum(axis=1)
        # print('kl entity', kl_entity.shape)
        # print(('kl_entity', kl_entity.shape, kl_entity))

        nb_occ_in_train = nb_occ_in_batch[uniq_entities]
        nb_occ_user_in_train = nb_occ_in_train[uniq_users]
        nb_occ_item_in_train = nb_occ_in_train[uniq_items]
        # print(('nb_occ_in_train', nb_occ_in_train))
        # print(('nb_occ_in_batch', nb_occ_in_batch))
        # print(('nb_occ_in_train/nb_occ_in_batch', nb_occ_in_train/nb_occ_in_batch))
        # nb_occ_batch = torch.bincount(x.flatten())
        # print('nboccs', nb_occ_in_batch.shape, nb_occ_in_train.shape)
        # nb_occ_batch[x]

        user_normalizer = (nb_occ_user_in_batch / nb_occ_user_in_train).sum(axis=0)
        item_normalizer = (nb_occ_item_in_batch / nb_occ_item_in_train).sum(axis=0)
        # print('normalizers', user_normalizer.shape, item_normalizer.shape)

        # print('begin', ((kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train)).shape)
        # print('ent', x)
        # print('ent', x <= self.N)
        # print('ent', (x <= self.N) * self.N)
        # print(('user_normalizer', user_normalizer.shape, user_normalizer.sum()))
        # print(('item_normalizer', item_normalizer.shape, item_normalizer.sum()))

        kl_rescaled = (
            (kl_bias + kl_entity) * (nb_occ_in_batch / nb_occ_in_train) *
            ((uniq_entities <= self.N) * self.N / user_normalizer + (uniq_entities > self.N) * self.M / item_normalizer)
        ).sum(axis=0)
        # print(('rescaled', kl_rescaled.shape, kl_rescaled))

        return (likelihood,
            last_logits, mean_logits,
            distributions.kl.kl_divergence(global_bias_sampler, self.global_bias_prior) +
            kl_rescaled
        )
