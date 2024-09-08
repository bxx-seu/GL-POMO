import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state, return_h_mean=False):
        self.encoded_nodes = self.encoder(reset_state.problems, reset_state.data_knn, reset_state.knn_idx)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)
        self.decoder.set_mean_q(self.encoded_nodes)

        if return_h_mean:
            return self.decoder.projection(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.ninf_mask.size(0)
        pomo_size = state.ninf_mask.size(1)

        if self.training:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            self.decoder.set_q1(_get_encoding(self.encoded_nodes, state.first_node))
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, step_state=state, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            prob = probs.gather(dim=-1,index=state.next_opt_node)
            # shape: (batch, sub_tours_num, 1or2)
            return prob
        else:
            if state.current_node is None:

                selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))

                encoded_first_node = _get_encoding(self.encoded_nodes, selected)
                # shape: (batch, pomo, embedding)
                self.decoder.set_q1(encoded_first_node)

            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                # shape: (batch, pomo, embedding)
                probs = self.decoder(encoded_last_node, step_state=state, ninf_mask=state.ninf_mask)
                # shape: (batch, pomo, problem)
                if self.training:
                    return probs
                elif self.model_params['eval_type'] == 'softmax':
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.SUB_TOURS_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                else:
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.local_feature = model_params['local_feature']
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)

        if self.local_feature:
            self.layers = nn.ModuleList([EncoderLayer_with_local(**model_params) for _ in range(encoder_layer_num)])
        else:
            self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, data_knn=None, data_knn_idx=None):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        if self.local_feature:
            for layer in self.layers:
                out = layer(out,data_knn,data_knn_idx)
        else:
            for layer in self.layers:
                out = layer(out)

        return out


class EncoderLayer_with_local(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        embedding_dim_local = self.model_params['embedding_dim_local']
        head_num_local = self.model_params['head_num_local']
        qkv_dim_local = self.model_params['qkv_dim_local']

        self.local_embedding = nn.Linear(3, embedding_dim_local)
        self.Wq_local = nn.Linear(embedding_dim, head_num_local * qkv_dim_local, bias=False)
        self.Wk_local = nn.Linear(embedding_dim_local, head_num_local * qkv_dim_local, bias=False)
        self.Wv_local = nn.Linear(embedding_dim, head_num_local * qkv_dim_local, bias=False)
        # self.Wv_local = nn.Linear(embedding_dim_local, head_num_local * qkv_dim_local, bias=False)
        self.multi_head_combine_local = nn.Linear(head_num_local * qkv_dim_local, embedding_dim)

        self.combine_global_local = nn.Parameter(torch.Tensor(embedding_dim))
        nn.init.ones_(self.combine_global_local)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, data_knn, data_knn_idx):
        # Global Feature
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)
        multi_head_out_global = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        # Local Feature
        # data_knn.shape: (batch, problem, k, 3)
        B,N,K,_ = data_knn.shape
        head_num_local = self.model_params['head_num_local']
        local_embedded_input = self.local_embedding(data_knn).reshape(B*N,K,-1)
        ###  Original
        q_local = reshape_by_heads(self.Wq_local(input1.reshape(B*N,1,-1)), head_num=head_num_local).reshape(B,N,head_num_local,1,-1)
        k_local = reshape_by_heads(self.Wk_local(local_embedded_input), head_num=head_num_local).reshape(B,N,head_num_local,K,-1)
        v_local = reshape_by_heads(self.Wv_local(input1), head_num=head_num_local)
        # q_local shape: (B, N, HEAD_NUM, 1, KEY_DIM)
        # k_local shape: (B, N, HEAD_NUM, K, KEY_DIM)
        # v_local shape: (B, HEAD_NUM, N, KEY_DIM)
        out_concat_local = multi_head_attention_local(q_local, k_local, v_local, data_knn_idx).squeeze()
        # shape: (B, N, HEAD_NUM*KEY_DIM)
        ###  Not Original
        # q_local = reshape_by_heads(self.Wq_local(input1.reshape(B * N, 1, -1)), head_num=head_num_local).reshape(B*N,head_num_local,1, -1)
        # k_local = reshape_by_heads(self.Wk_local(local_embedded_input), head_num=head_num_local).reshape(B*N,head_num_local,K, -1)
        # v_local = reshape_by_heads(self.Wv_local(local_embedded_input), head_num=head_num_local).reshape(B * N,head_num_local,K, -1)
        # out_concat_local = multi_head_attention(q_local, k_local, v_local).reshape(B,N,-1)
        ### End


        multi_head_out_local = self.multi_head_combine_local(out_concat_local)
        # shape: (B, N, EMBEDDING_DIM)

        # Combine Global and Local
        multi_head_out = multi_head_out_global + self.combine_global_local * multi_head_out_local
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_mean = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        self.q_mean = None  # saved q_mean, for problem adaptation

        # SGE
        self.encoded_nodes = None
        self.graph_emb = None
        self.sub_graph_emb_sum = None
        self.sub_graph_emb = None
        self.sub_graph_step_keep = 0

        if self.model_params['sub_graph_emb']:
            self.W_subg_mean = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)


    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.encoded_nodes = encoded_nodes
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def projection(self, encoded_nodes):
        if self.training:
            projected_nodes = self.proj(encoded_nodes)
        else:
            projected_nodes = encoded_nodes
        return projected_nodes

    def set_mean_q(self, encoded_nodes):
        head_num = self.model_params['head_num']

        graph_embed = encoded_nodes.mean(1)

        x = self.Wq_mean(graph_embed)[:, None, :]

        self.q_mean = reshape_by_heads(x, head_num=head_num)

        self.sub_graph_step_keep = 0  # SGE visited node count

    def forward(self, encoded_last_node, step_state, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        graph_emb = self.get_sub_graph_emb(step_state, encoded_last_node)
        q = self.q_first + q_last + graph_emb
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

    def get_sub_graph_emb(self, step_state, encoded_last_node=None):
        # ninf_mask.shape: (batch, pomo, problem)
        if not self.model_params['sub_graph_emb']:
            return self.q_mean

        if self.training:
            half_sub_tours_num = step_state.sub_tours_size // 2
            gather_index1 = torch.stack([step_state.opt_tours[step_state.BATCH_IDX[:, :half_sub_tours_num],
                                          step_state.first_node_idx[:, :half_sub_tours_num] - i - 1]
                             for i in range(step_state.problem_size-step_state.length_of_sub_tour)], dim=-1)
            gather_index2 = torch.stack([step_state.opt_tours[step_state.BATCH_IDX[:, half_sub_tours_num:],
                                            (step_state.first_node_idx[:, half_sub_tours_num:] + i + 1) % step_state.problem_size]
                             for i in range(step_state.problem_size - step_state.length_of_sub_tour)], dim=-1)
            gather_index = torch.cat((gather_index1, gather_index2), dim=1)[:, :, :, None].expand(
                gather_index1.size(0),2*gather_index1.size(1),gather_index1.size(2),self.encoded_nodes.size(-1))
            # (batch, problem, embedding), index: (batch, sub_tours_num, unvisited_number, embedding)
            sub_graph_emb_ = self.encoded_nodes[:,None,:,:].expand(self.encoded_nodes.size(0),gather_index.size(1),self.encoded_nodes.size(1),self.encoded_nodes.size(2)) \
                                                               .gather(dim=-2, index=gather_index)
            sub_graph_emb = sub_graph_emb_.mean(dim=-2)
            # (batch, sub_tours_num, embedding)
            sub_graph_emb = self.W_subg_mean(sub_graph_emb)
            self.sub_graph_emb = reshape_by_heads(sub_graph_emb, head_num=self.model_params['head_num'])

        else:
            if self.sub_graph_step_keep == 0:
                pomo_size = encoded_last_node.shape[-2]
                self.sub_graph_emb_sum = self.encoded_nodes.sum(1)[:, None, :].repeat(1, pomo_size,1)
                # (batch_size, pomo_size, embedding_size)

            assert encoded_last_node is not None
            self.sub_graph_emb_sum = self.sub_graph_emb_sum - encoded_last_node
            self.sub_graph_step_keep += 1
            if self.sub_graph_step_keep%self.model_params['sub_graph_steps'] == 0:
                sub_graph_emb_ = self.sub_graph_emb_sum/(self.encoded_nodes.shape[1]-self.sub_graph_step_keep)  # problem_size - pre_steps - cur_step
                sub_graph_emb = self.W_subg_mean(sub_graph_emb_)
                self.sub_graph_emb = reshape_by_heads(sub_graph_emb, head_num=self.model_params['head_num'])
        # (batch_size,head_num, pomo_size,key_dim)
        return self.sub_graph_emb

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

def multi_head_attention_local0(q, k, v, knn_idx, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q_local shape: (B, N, HEAD_NUM, 1, KEY_DIM)
    # k_local shape: (B, N, HEAD_NUM, K, KEY_DIM)  K on KNN
    # v_local shape: (B, N, HEAD_NUM, N, KEY_DIM)
    # knn_idx shape: (B, N, K)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    N = q.size(1)
    head_num = q.size(2)
    n = q.size(3)  # n==1
    key_dim = q.size(4)
    input_s = k.size(3)
    K = k.size(3)

    score = torch.matmul(q, k.transpose(3, 4))
    # shape: (B, N, head_num, 1, K)
    score_scaled_ = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    score_ninf = torch.ones((batch_s, N, head_num, 1, N))*float('-inf')
    score_scaled = score_ninf.scatter_(dim=-1,index=knn_idx[:,:,None,None,:].expand(-1,-1,head_num,1,-1),src=score_scaled_)
    # shape: (B, N, head_num, 1, K)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=-1)(score_scaled)
    # shape: (batch, N, head_num, n, N)
    out = torch.matmul(weights, v)
    # shape: (batch, N, head_num, n, key_dim)
    out_transposed = out.transpose(2, 3)
    # shape: (batch, N, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, N, n, head_num * key_dim)
    # shape: (batch, N, n, head_num*key_dim)
    return out_concat

def multi_head_attention_local(q, k, v, knn_idx, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q_local shape: (B, N, HEAD_NUM, 1, KEY_DIM)
    # k_local shape: (B, N, HEAD_NUM, K, KEY_DIM)  K on KNN
    # v_local shape: (B, HEAD_NUM, N, KEY_DIM)
    # knn_idx shape: (B, N, K)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    N = q.size(1)
    head_num = q.size(2)
    n = q.size(3)  # n==1
    key_dim = q.size(4)
    input_s = k.size(3)
    K = k.size(3)

    score = torch.matmul(q, k.transpose(3, 4))
    # shape: (B, N, head_num, 1, K)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    # shape: (B, N, head_num, 1, K)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    weights = nn.Softmax(dim=-1)(score_scaled)
    # shape: (batch, N, head_num, n, K)

    # knn_v = v.unsqueeze(1).expand(-1,N,-1,-1,-1).gather(dim=3,index=knn_idx[:,[i],:].unsqueeze(-1).expand(-1,head_num,-1,key_dim)).unsqueeze(1)
    knn_v = v.unsqueeze(1).expand(-1, N, -1, -1, -1).gather(dim=3,
                        index=knn_idx.unsqueeze(2).unsqueeze(-1).expand(-1, -1, head_num,-1,key_dim))
    # (B, N, HEAD_NUM, K, KEY_DIM): (B, N, HEAD_NUM, N, KEY_DIM) -> (B, N, HEAD_NUM, K, KEY_DIM)
    out = torch.matmul(weights, knn_v)
    # shape: (batch, N, head_num, n, key_dim)
    out_transposed = out.transpose(2, 3)
    # shape: (batch, N, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, N, n, head_num * key_dim)
    # shape: (batch, N, n, head_num*key_dim)
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
