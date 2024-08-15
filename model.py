import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math

class Invariant_Learning(nn.Module):
    def __init__(self, tau, K):
        super(Invariant_Learning, self).__init__()
        self.tau = tau
        self.K = K

    def invariant_loss(self, in_list):
        def self_sim(z1, z2):
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return (z1 * z2).sum(1)

        def compute(z1, z2):
            f = lambda x: torch.exp(x / self.tau)
            between_sim = f(self_sim(z1, z2))
            rand_item = torch.randperm(z1.shape[0])
            neg_sim = f(self_sim(z1, z2[rand_item])) + f(self_sim(z2, z1[rand_item]))
            return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

        n_entities = in_list[0].shape[0]
        T = self.tau
        choose_index = self.get_compare_index()
        all_loss = []
        
        for i, j in choose_index:
            x1, x2 = in_list[i], in_list[j] # torch.Size([49545, 64])
            loss = compute(x1, x2)
            all_loss.append(loss.mean().view(-1))

        total_loss = torch.cat(all_loss, dim=0)
        return total_loss

    def get_compare_index(self):
        
        K = self.K
        # if K == 2:
        #     return [(0, 1)]
        out = []
        for i in range(K - 1):
            j = i + 1
            for _ in range(K - i - 1):
                out.append((i, j))
                j += 1
        choose_index = random.sample(out, K)
        return choose_index

    def forward(self, user_list, entity_list):      
        
        inv_u_loss = self.invariant_loss(user_list)
        inv_kg_loss = self.invariant_loss(entity_list)
        # inv_ckg_loss_i = self.invariant_loss(item_list)
        # inv_ckg_loss_u = self.invariant_ckg_loss(user_list, )
        # inv_ckg_loss_i = self.invariant_ckg_loss(item_list)
        return inv_u_loss, inv_kg_loss

class Invariant_Capture(nn.Module):
    def __init__(self, n_users, channel):
        super(Invariant_Capture, self).__init__()
        self.n_users = n_users
        self.alpha = 0.1
        self.kg_W_r = nn.Parameter(torch.Tensor(channel, channel))
        self.kg_W_rQ = nn.Parameter(torch.Tensor(channel, channel))
        self.kg_W_rK = nn.Parameter(torch.Tensor(channel, channel))             # global
        self.kg_W_trip = nn.Parameter(torch.Tensor(channel, channel)) 
        self.ui_Wh = nn.Parameter(torch.Tensor(channel, channel))           # local
        self.ui_We = nn.Parameter(torch.Tensor(channel*2, 1))  
        # self.fc = nn.Linear(channel*2, channel)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.kg_heads = 2
        self.d_k = channel // self.kg_heads
        self._init()

    def _init(self):
        initializer = nn.init.xavier_uniform_
        initializer(self.kg_W_r)
        initializer(self.kg_W_rQ)
        initializer(self.kg_W_rK)
        initializer(self.kg_W_trip)
        initializer(self.ui_Wh)
        initializer(self.ui_We)

    def relation_aware_att(self, entity_emb, edge_index, edge_type, relation_weight, g_mask=None): 
        
        head = edge_index[0]
        if g_mask != None:
            relation_embed = relation_weight[edge_type - 1] * g_mask
            head_embed = entity_emb[head] * g_mask
        else:
            relation_embed = relation_weight[edge_type - 1] 
            head_embed = entity_emb[head]

        query = (head_embed @ self.kg_W_r).view(-1, self.kg_heads, self.d_k)
        key = (relation_embed @ self.kg_W_r).view(-1, self.kg_heads, self.d_k)
        score_r = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        score_r = score_r.mean(-1)
        rel_attn_score = scatter_softmax(score_r, head)
        return rel_attn_score

    def kg_inv_capture(self, entity_emb, edge_index, edge_type, relation_weight, rel_scores, g_mask=None):

        head, tail = edge_index
        if g_mask != None:
            h_emb, t_emb = entity_emb[head] * g_mask, entity_emb[tail] * g_mask
            r_emb = relation_weight[edge_type - 1] * rel_scores.unsqueeze(1) * g_mask
        else:
            h_emb, t_emb = entity_emb[head], entity_emb[tail]
            r_emb = relation_weight[edge_type - 1] * rel_scores.unsqueeze(1)

        hr = torch.cat((h_emb, r_emb), dim=1)
        tr = torch.cat((t_emb, r_emb), dim=1)
        hr, tr = hr.unsqueeze(1), tr.unsqueeze(1).permute(0, 2, 1)
        score_trip = torch.bmm(hr, tr).squeeze(1).squeeze(1)
        kg_attn_score = scatter_softmax(score_trip, head)
        return kg_attn_score            #   eq5
    
    def forward(self, entity_emb, user_emb, edge_index, edge_type, relation_weight, g_mask=None):
        
        if g_mask != None:
            g_mask = g_mask.unsqueeze(1)
        rel_scores = self.relation_aware_att(entity_emb, edge_index, edge_type, relation_weight, g_mask) # eq(3-4)
        kg_invariant_score = self.kg_inv_capture(entity_emb, edge_index, edge_type, relation_weight, rel_scores, g_mask) # eq5
        return kg_invariant_score # return eq5 and eq10

class EnvGenerator(nn.Module):

    def __init__(self, K, input_dim, mlp_dim, device):
        super(EnvGenerator, self).__init__()

        self.K = K
        self.input_dim = input_dim
        self.device = device
        self.mlp_dim = mlp_dim
        # self.temperature = 1.0
        self.temperature = 1.0
        self.bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        self.edge_mlp_list = nn.ModuleList()

        for i in range(K):
            mlp = nn.Sequential(nn.Linear(input_dim*3, mlp_dim).to(device), 
                                nn.ReLU(), 
                                nn.Linear(mlp_dim, 1).to(device))
            # fea.data.fill_(1e-7)
            self.edge_mlp_list.append(mlp)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, triplet_emb, edge_index):

        bias = self.bias
        head, tail = edge_index[0], edge_index[1]
        edge_weight_list = []
        hard_weight_list = []
        for k in range(self.K):
            edge_logits = self.edge_mlp_list[k](triplet_emb)
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + edge_logits) / self.temperature
            edge_weight = torch.sigmoid(gate_inputs).squeeze()
            hard_edge_weight = self.gumbel_tohard(edge_weight)
            edge_weight_list.append((edge_weight))
            hard_weight_list.append((hard_edge_weight))
        return edge_weight_list, hard_weight_list

    def gumbel_tohard(self, soft_gumbel):
        threshold = 0.5 
        hard_gumbel = (soft_gumbel >= threshold).float()
        hard_gumbel = (hard_gumbel - soft_gumbel).detach() + soft_gumbel
        return hard_gumbel

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_factors):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att, kg_mask=None):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]

        if kg_mask is not None:
            neigh_relation_emb = neigh_relation_emb * kg_mask.unsqueeze(1)

        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]      # beta(u_p) [23566, 4, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),           # disen_weight -- e_p [23566, 4, 64]
                                weight).expand(n_users, n_factors, channel)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.K = 3
        self.temperature = 0.2
        self.invariant_generator = Invariant_Capture(n_users, channel)

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, augment_weight, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices], augment_weight[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False, g_mask=None):

        cor = self._cul_cor()
        if node_dropout:
            # if edge_index.shape[1] > 1000000:
            #     kg_dr = 0.2
            # else:
            #     kg_dr = self.node_dropout_rate
            # edge_index, edge_type, g_mask = self._edge_sampling(edge_index, edge_type, g_mask, kg_dr)
            edge_index, edge_type, g_mask = self._edge_sampling(edge_index, edge_type, g_mask, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        invariant_mask = self.invariant_generator(entity_emb, user_emb, edge_index, edge_type, self.weight, g_mask)
        # invariant_mask = torch.ones_like(invariant_mask)
        entity_res_emb, user_res_emb = self.conv_compute(entity_emb, user_emb, latent_emb, 
                                                         edge_index, edge_type, interact_mat, 
                                                         mess_dropout, invariant_mask)
        return entity_res_emb, user_res_emb, cor
    
    def forward_eval(self, user_emb, entity_emb, latent_emb, edge_index, edge_type, interact_mat):
        
        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        invariant_mask = self.invariant_generator(entity_emb, user_emb, edge_index, edge_type, self.weight)
        # invariant_mask = torch.ones_like(invariant_mask)
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att, invariant_mask)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        
        return entity_res_emb, user_res_emb

    def conv_compute(self, entity_emb, user_emb, latent_emb, edge_index, edge_type, 
                     interact_mat, mess_dropout,invariant_mask):

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(len(self.convs)):

            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att, invariant_mask)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        return entity_res_emb, user_res_emb


    
class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat, augmenter=None):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind

        self.K = args_config.K
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.gcn = self._init_model()
        self.inv_compute = Invariant_Learning(args_config.tau, args_config.K)
        self.augmenter = augmenter

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_res_emb, user_res_emb = self.gcn.forward_eval(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat)
        return entity_res_emb, user_res_emb

    def forward(self, batch=None):
        
        inv_mean, inv_var = torch.tensor([0]).to(self.device), torch.tensor([0]).to(self.device)
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        h_embed, t_embed = item_emb[self.edge_index[0]], item_emb[self.edge_index[1]]
        r_embed = self.gcn.weight[self.edge_type - 1]
        triplet_embed = torch.cat((h_embed, r_embed, t_embed), dim=1)
        env_gmask_list, hard_gmask_list = self.augmenter(triplet_embed, self.edge_index)

        rec_loss = 0
        user_gcn_emb_list, entity_gcn_emb_list = [], []

        loop = self.K
        for i in range(loop):
            g_mask = env_gmask_list[i]
            entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb, item_emb,
                                                        self.latent_emb, self.edge_index,
                                                        self.edge_type, self.interact_mat,
                                                        mess_dropout=self.mess_dropout,
                                                        node_dropout=self.node_dropout,
                                                        g_mask=g_mask)
            u_e = user_gcn_emb[user]
            pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
            loss_i = self.create_bpr_loss(u_e, pos_e, neg_e, cor)
            rec_loss += loss_i
            user_gcn_emb_list.append(user_gcn_emb)
            entity_gcn_emb_list.append(entity_gcn_emb)
        rec_loss = rec_loss / loop
        batch_user_list = self.get_batch_emb(user, user_gcn_emb_list)
        batch_entity_list = self.get_batch_emb(torch.cat((pos_item, neg_item), dim=0), entity_gcn_emb_list)
        inv_mean, inv_var = self.get_inv_loss(batch_user_list, batch_entity_list)
        return rec_loss, inv_mean, inv_var

    
    def get_inv_loss(self, batch_user_list, batch_entity_list):

        inv_u_loss, inv_kg_loss = self.inv_compute(batch_user_list, batch_entity_list)
        var_inv, mean_inv = torch.var_mean(inv_u_loss + inv_kg_loss)
        return mean_inv, var_inv

    def get_batch_emb(self, index, emb_list):
        idex_list =[]
        for emb in emb_list:
            idex_emb = emb[index]
            idex_list.append(idex_emb)
        return idex_list

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss
