import numpy as np
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '../utils/')
import time
import pdb
from tqdm.notebook import tqdm

class Nested_MNL(nn.Module):
    """
    The stratified model by rank. May contain fixed effects.
    """
    def __init__(self, num_items, ism, num_nests, nest_memberships, 
                 scale=1.0, fixed_effects=False, linear_terms=False, 
                 covariates=None, no_ego=True, num_ranks=5):
        """
        Inputs: 
        num_items - the number of items in the choice system modeled (number of programs)
        ism - if dataset is multi-set, in which case padding is used
        num_nests - the number of nests in the choice system modeled
        nest_memberships - list of length num_items dictating which nest the alternative belongs to
        scale - float in (0,1]. corresponds to amount of correlation within nests.
        fixed_effects - if True, item-level fixed effects are used in utility
        linear_terms - if True, linear terms are used in utility
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        """
        super().__init__()
        self.num_items = num_items
        
        self.num_nests = int(num_nests)
        self.nest_memberships = t.from_numpy(nest_memberships).type(t.long)
        self.B = {i:[] for i in range(num_nests)}
        [self.B[nest].append(i) for i, nest in enumerate(nest_memberships)]
        
        self.ism = ism
        self.fixed_effects = fixed_effects
        self.linear_terms = linear_terms
        self.no_ego = no_ego
        num_agents, _, self.num_features = covariates.shape if linear_terms else (0,0,0)
        if linear_terms:
            pad_vec = np.zeros([num_agents, self.num_features], dtype=np.float32)
            covariates = np.hstack([covariates, pad_vec[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            self.covariates =  t.from_numpy(covariates) # num_items+1 x num_features tensor
        else:
            self.covariates = covariates
            
        self.num_ranks = num_ranks
        self.last_loss = None
        self.__build_model()

    def __build_model(self):
        """
        Helper function to initialize the model
        """    
        padding_idx = self.num_items
        
        self.scale_parameter = Embedding(
            num_embeddings=self.num_nests,  # +1 for the padding
            embedding_dim=1,
            _weight=t.ones([self.num_nests, 1])) # requires_grad=True

        self.logits = Embedding(
            num_embeddings=self.num_items + 1,  # +1 for the padding
            embedding_dim=1,
            padding_idx=padding_idx) # requires_grad=True
            
        self.beta = t.nn.Linear(self.num_features, 1, bias=False)
        
    def compute_utils(self, x_extra, covariates=None, inf_weight=float('-inf')):
        batch_size = x_extra.size()[0]
        utilities = t.zeros((batch_size, self.num_items+1))
        
        if self.fixed_effects:
            utilities += self.logits.weight.flatten()[None, :] #batch_size x (num_items+1)
            pass
        
        if self.linear_terms:
            cov = covariates[x_extra] # batch_size x (num_items + 1) x num_features matrix
            utilities += self.beta(cov).squeeze()
            pass
        
        return utilities[:,:-1]

    def compute_nested_probabilities(self, x, x_extra, C, covariates, sampling=False):
        '''
        :type x: t.Tensor(int) of size (batch_size, max_seq_len)
            Padded tensor of choice sets. 
            For j < choice_set_len, x[i,j] is some item in the current choice 
            set of observation i. 
        :type x_extra: t.Tensor(int) of size (batch_size, )
            Indicates ballot owner.
        :type C: np.ndarray(int) of size (batch_size, num_items). 
            Indicators for which items are in the choice set. 
            C[i,j] is 1 if item j is in the choice set of observation i, 
            and 0 otherwise.
        :type covariates: t.Tensor(float) of size (batch_size, num_items+1)
        '''
        batch_size = x.size()[0]
        
        V = self.compute_utils(x_extra, covariates=covariates)
        elem_to_scale_param = t.ones(V.shape)
        for i in range(self.num_nests):
            elem_to_scale_param[:, self.B[i]] = self.scale_parameter(i)

        # Compute scaled utilities for all agents and items
        scaled_V = t.div(V, elem_to_scale_param)

        # Calculate the e^(scaled_V) = exp(V / lambda)
        exp_scaled_V = t.exp(scaled_V)
                
        # Compute elements of P_lower
        a_num = t.mul(exp_scaled_V,C) # Mask out utilities not in the choice set.
        a_denom = t.ones(a_num.size())
        for i in range(self.num_nests):
            a_denom[:, self.B[i]] = a_num[:, self.B[i]].sum(1, keepdim=True)
        zero_idx = (a_denom == 0.) # The nest is not in the choice set
        a_denom[zero_idx] = 1. # These elements will not be used anyway -- set to 1.

        # Compute numerator of P_upper.
        # assert(t.all(a_denom>0.0))
        a_temp = t.pow(a_denom, elem_to_scale_param)
        b_num = t.zeros((batch_size, self.num_nests))
        for i in range(self.num_nests):
            b_num[:,i] = a_temp[:, self.B[i][0]]
                
        # Compute denominator of P_upper.
        b_denom = b_num.sum(1,keepdim=True)

        # Compute P_lower and P_upper
        P_lower = t.zeros((batch_size, self.num_items))
        P_lower[a_denom!=0.] = t.div(a_num[a_denom!=0.], a_denom[a_denom!=0.]) # (batch_size, num_items)
        P_upper = t.div(b_num, b_denom) # (batch_size, num_items)

        # Add padding dimension for indexing with x, as x contains padding of 'self.num_items'
        P_lower = t.hstack((P_lower, t.zeros((batch_size, 1))))
        P_upper = t.hstack((P_upper, t.zeros((batch_size, 1))))

        # Nest membership of each item in x
        x_to_memberships = t.full(x.size(), fill_value=self.num_nests)
        x_to_memberships[x!=self.num_items] = self.nest_memberships[x[x!=self.num_items]]
        
        # Select elements from lower and upper tensors according to choice sets
        # Multiply for final probabilities
        P_lower_selected = P_lower[np.arange(batch_size)[:,None], x]
        P_upper_selected = P_upper[np.arange(batch_size)[:,None], x_to_memberships]
        
        # Compute final probabilities
        P = t.mul(P_lower_selected, P_upper_selected)
        if t.any(t.all(P<=0., dim=1)):
            print('Found all 0s row')
            return
        
        if sampling:
            return P
        else:
            zero_idx = (P == 0.)
            P[zero_idx] = 1. # Will not be seen anyway
            return P
                
    def forward(self, x, x_extra, covariates=None, inf_weight=float('-inf')):
        """
        Computes the stratified distribution
        
        Inputs: 
        x - item indices involved in the choice set or chosen set of interest. size: batch_size x maximum sequence length x 2
        x_extra - Whose choice, len of choice set, and len of chosen set. Used to determine padding. siz: batch_size x 3
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """
        batch_size = x.size()[0]
        if covariates is None:
            covariates=self.covariates
        else:
            num_agents, num_alternatives, num_features = covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            covariates = t.from_numpy(covariates)
        
        # utilities = self.compute_utils(x_extra, covariates=covariates)
        
        choice_set_indicators = t.zeros((batch_size, self.num_items+1))
        choice_set_indicators[np.arange(batch_size)[:,None], x[...,0]] = 1
        
        p = self.compute_nested_probabilities(x[...,0], x_extra[:,0], choice_set_indicators[:,:-1], covariates)
        return t.log(p)
    
    def loss_func(self, y_hat, y, x_extra=None, train=True):
        """
        Evaluates the Choice Model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (must be less than x_lengths)
        x_lengths - the size of the choice set, used to determine padding. 
        The current implementation assumes that y are less than x_lengths, so this
        is unused.
        """
        terms = []
        for i in range(self.num_ranks):
            rows=x_extra[:,2]==i
            terms.append(F.nll_loss(y_hat[rows], y[rows].long()))
        terms.append(t.tensor(0.))
        loss = F.nll_loss(y_hat, y.long())
        self.last_loss = loss
        return (loss, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()

class Embedding(nn.Module):
    """
    Redefining torch.nn.Embedding (see docs for that function)
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, _weight=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        if _weight is None:
            self.weight = nn.Parameter(t.randn([self.num_embeddings, self.embedding_dim])/np.sqrt(self.num_embeddings))
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, x):
        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)

        return self.weight[x]

class DataLoader():
    """
    Redefining torch.utils.data.DataLoader, see docs for that function
    Done so because it is faster for CPU only use.
    """
    def __init__(self, data, batch_size=None, shuffle=False):
        # data must be a list of tensors
        self.data = data
        self.data_size = data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration_flag:
            self.stop_iteration_flag = False
            raise StopIteration()
        if self.batch_size is None or self.batch_size >= self.data_size:
            self.stop_iteration_flag = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration_flag = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]
            return batch

class StratifiedLinear(nn.Module):
    """
    The stratified model by rank. May contain fixed effects.
    """
    def __init__(self, num_items, ism, fixed_effects=False, 
                 linear_terms=False, covariates=None, 
                 context=False, top_k=None, embedding_dim=5, no_ego=True, 
                 k=1, lambda_reg=1.0, order_reg=2, eps=1e-2, num_ranks=5):
        """
        Initializes the Stratified Model, as in https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf
        
        Inputs: 
        num_items - the number of items in the choice system modeled (number of programs)
        ism - if dataset is multi-set, in which case padding is used
        fixed_effects - if True, item-level fixed effects are used in utility
        linear_terms = if True, linear terms are used in utility
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        k - number of stratified models to learn
        lambda_reg - weight of regularization
        order_reg - order of regularization
        """
        super().__init__()
        self.num_items = num_items
        self.ism = ism
        self.fixed_effects = fixed_effects
        self.linear_terms = linear_terms
        self.context = context
        self.top_k = num_items if top_k is None else top_k
        self.embedding_dim = embedding_dim
        self.no_ego = no_ego
        num_agents, _, self.num_features = covariates.shape if linear_terms else (0,0,0)
        if linear_terms:
            pad_vec = np.zeros([num_agents, self.num_features], dtype=np.float32)
            covariates = np.hstack([covariates, pad_vec[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            self.covariates =  t.from_numpy(covariates) # num_items+1 x num_features tensor
        else:
            self.covariates = covariates
        self.k = k
        self.lambda_reg = lambda_reg
        self.order_reg = order_reg
        self.eps = eps
        self.last_loss = None
        self.num_ranks = num_ranks
        self.__build_model()

    def __build_model(self):
        """
        Helper function to initialize the model
        """    
        padding_idx = self.num_items
        
        logits = []
        if self.fixed_effects:
            for i in range(self.k):
                logits.append(Embedding(
                num_embeddings=self.num_items + 1,  # +1 for the padding
                embedding_dim=1,
                padding_idx=padding_idx)) # requires_grad=True
        self.logits = nn.ModuleList(logits)
            
        beta = []
        if self.linear_terms:
            for i in range(self.k):
                beta.append(t.nn.Linear(self.num_features, 1, bias=False))
        self.beta = nn.ModuleList(beta)
        
        target=[]
        context=[]
        if self.context:
            for i in range(self.k):
                target.append(Embedding(num_embeddings=self.num_items+1,
                                        embedding_dim=self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=t.zeros([self.num_items+1, self.embedding_dim])))
                context.append(Embedding(num_embeddings=self.num_items+1,
                                         embedding_dim=self.embedding_dim,
                                         padding_idx=padding_idx))
        self.target_embeddings = nn.ModuleList(target)
        self.context_embeddings = nn.ModuleList(context)

    def reg(self):
        '''
        Computes regularization term for loss

        self.beta is a ModuleList of t.nn.Linear() weights
        self.targets & self.context are ModuleLists of Embeddings
        self.order_reg is the order of regularization (either L1 or L2 in this case)
        self.k is the number of stratified models learned (ie. length of self.beta)
        '''
        fe_reg = t.zeros(1) if not self.fixed_effects else t.sum(t.stack([t.linalg.norm(t.sub(self.logits[i].weight, self.logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        linear_reg = t.zeros(1) if not self.linear_terms else t.sum(t.stack([t.linalg.norm(t.sub(self.beta[i].weight, self.beta[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        context_reg = t.zeros(1) if ((not self.context) | (self.k<=2)) else t.sum(t.stack([t.linalg.norm(t.sub(self.target_embeddings[i].weight, self.target_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.context_embeddings[i].weight, self.context_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)]))
        
        return fe_reg + linear_reg + context_reg
    
    def forward(self, x, x_extra=None, covariates=None, inf_weight=float('-inf')):
        """
        Computes the Stratified y_hat
        
        Inputs: 
        x - item indices involved in the choice set or chosen set of interest. size: batch_size x maximum sequence length x 2
        x_extra - Whose choice, len of choice set, and len of chosen set. Used to determine padding. siz: batch_size x 3
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        utilities = t.zeros((batch_size, seq_len, 1))
        if covariates is None:
            covariates=self.covariates
        else:
            num_agents, num_alternatives, num_features = covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            covariates = t.from_numpy(covariates)

        
        if self.fixed_effects:
            logits = t.zeros((batch_size, seq_len, 1))
            for i in range(self.k-1):
                rows = x_extra[:,2]==i
                logits[rows, :, :] = self.logits[i](x[rows, :, 0])
            logits[x_extra[:,2]>=(self.k-1), :, :] = self.logits[-1](x[x_extra[:,2]>=(self.k-1), :, 0])
            utilities += logits #batch_size x max_set_len
            pass
        
        if self.linear_terms:
            cov = covariates[x_extra[:,0,None], x[:,:,0]] # batch_size x max_set_len x num_features matrix
            linear = t.zeros((batch_size, seq_len, 1))
            for i in range(self.k-1):
                rows = x_extra[:,2]==i
                linear[rows, :, :] = self.beta[i](cov[rows, :, :])
            rows = x_extra[:,2]>=(self.k-1)
            linear[rows, :, :] = self.beta[-1](cov[rows, :, :])
            utilities += linear
            pass
        
        if self.no_ego & self.context:
            for i in range(1,self.k-1):
                rows = x_extra[:,2]==i
                context_vecs = self.context_embeddings[i](x[rows,:self.top_k,1]).sum(-2, keepdim=True) #self.layer1(self.target_embedding(x))
                # context_vecs = self.context_embeddings[i](x[rows,:,1]).sum(-2, keepdim=True)#self.layer1(self.target_embedding(x))
                # context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
                utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) * context_vecs).sum(-1,keepdim=True)/float(i)
            rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,2]!=0.0)
            context_vecs = self.context_embeddings[-1](x[rows,:self.top_k,1]).sum(-2, keepdim=True) #self.layer1(self.target_embedding(x))
            # context_vecs = self.context_embeddings[-1](x[rows,:,1])
            # context_vecs = context_vecs.sum(-2, keepdim=True)
            # - context_vecs
            utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) * context_vecs).sum(-1,keepdim=True).div(x_extra[rows,2][:,None,None]) # average context effect
        elif (not self.no_ego) & self.context:
            for i in range(1,self.k-1):
                rows = x_extra[:,2]==i
                context_vecs = self.context_embeddings[i](x[rows,:self.top_k,1])
                context_vecs = context_vecs.sum(-2)[...,None]
                utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) @ context_vecs)/float(i)
            rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,2]!=0.0)
            context_vecs = self.context_embeddings[-1](x[rows,:self.top_k,1])
            context_vecs = context_vecs.sum(-2)[...,None]
            utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) @ context_vecs).div(x_extra[rows,2][:,None,None])
        else:
            pass

        if self.ism:
            utilities[t.arange(seq_len)[None, :] >= x_extra[:, 1, None]] = inf_weight
        return F.log_softmax(utilities, 1).squeeze()
    
    def loss_func(self, y_hat, y, x_extra=None, train=True):
        """
        Evaluates the Choice Model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (must be less than x_lengths)
        x_lengths - the size of the choice set, used to determine padding. 
        The current implementation assumes that y are less than x_lengths, so this
        is unused.
        """
        if (self.k<=1):
            terms = []
            for i in range(self.num_ranks):
                rows=x_extra[:,2]==i
                terms.append(F.nll_loss(y_hat[rows], y[rows].long()))
            terms.append(t.tensor(0.))
            loss = F.nll_loss(y_hat, y.long())
            self.last_loss = loss
            return (loss, terms)
        else:
            terms = []
            for i in range(self.num_ranks):
                rows=x_extra[:,2]==i
                terms.append(F.nll_loss(y_hat[rows], y[rows].long()))
            # terms.append(F.nll_loss(y_hat[x_extra[:,2]>=self.k-1], y[x_extra[:,2]>=self.k-1, None].long()))
            tl = F.nll_loss(y_hat, y.long())
            if train:
                rl = self.lambda_reg*self.reg()
                terms.append(rl)
                return (tl + rl, terms)
            else:
                return (tl, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()

# class StratifiedLinear_Parallelized(nn.Module):
#     """
#     The stratified model by rank. May contain fixed effects.
#     """
#     def __init__(self, num_items, ism, fixed_effects=False, 
#                  linear_terms=False, covariates=None, 
#                  context=False, embedding_dim=5, no_ego=True, 
#                  k=1, lambda_reg=1.0, order_reg=2, eps=1e-2):
#         """
#         Initializes the Stratified Model, as in https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf
        
#         Inputs: 
#         num_items - the number of items in the choice system modeled (number of programs)
#         ism - if dataset is multi-set, in which case padding is used
#         fixed_effects - if True, item-level fixed effects are used in utility
#         linear_terms = if True, linear terms are used in utility
#         covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
#         k - number of stratified models to learn
#         lambda_reg - weight of regularization
#         order_reg - order of regularization
#         """
#         super().__init__()
#         self.num_items = num_items
#         self.ism = ism
#         self.fixed_effects = fixed_effects
#         self.linear_terms = linear_terms
#         self.context = context
#         self.embedding_dim = embedding_dim
#         self.no_ego = no_ego
#         num_agents, _, self.num_features = covariates.shape if linear_terms else (0,0,0)
#         if linear_terms:
#             pad_vec = np.zeros([num_agents, self.num_features], dtype=np.float32)
#             covariates = np.hstack([covariates, pad_vec[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
#             self.covariates =  t.from_numpy(covariates) # num_items+1 x num_features tensor
#         else:
#             self.covariates = covariates
#         self.k = k
#         self.lambda_reg = lambda_reg
#         self.order_reg = order_reg
#         self.eps = eps
#         self.last_loss = None
#         self.__build_model()

#     def __build_model(self):
#         """
#         Helper function to initialize the model
#         """    
#         padding_idx = self.num_items
        
#         if self.fixed_effects:
#             self.logits = nn.Parameter(t.zeros([self.k,self.num_items]))

#         if self.linear_terms:
#             self.beta = nn.Parameter(t.zeros([self.k,self.num_features]))
        
#         if self.context:
#             self.target = nn.Parameter(t.zeros([self.k, self.num_items, self.embedding_dim]))
#             self.context = nn.Parameter(t.zeros([self.k, self.num_items, self.embedding_dim]))

#     def reg(self):
#         '''
#         Computes regularization term for loss

#         self.beta is a ModuleList of t.nn.Linear() weights
#         self.targets & self.context are ModuleLists of Embeddings
#         self.order_reg is the order of regularization (either L1 or L2 in this case)
#         self.k is the number of stratified models learned (ie. length of self.beta)
#         '''
#         if self.context:
#             return t.sum(t.stack([t.linalg.norm(t.sub(self.logits[i].weight, self.logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)])) +\
#         t.sum(t.stack([t.linalg.norm(t.sub(self.beta[i].weight, self.beta[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)])) +\
#         t.sum(t.stack([t.linalg.norm(t.sub(self.target_embeddings[i].weight, self.target_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)])) +\
#         t.sum(t.stack([t.linalg.norm(t.sub(self.context_embeddings[i].weight, self.context_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)]))
#         else:
#             return t.sum(t.stack([t.linalg.norm(t.sub(self.logits[i].weight, self.logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)])) + \
#         t.sum(t.stack([t.linalg.norm(t.sub(self.beta[i].weight, self.beta[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
    
#     def forward(self, x, x_extra=None, inf_weight=float('-inf')):
#         """
#         Computes the Stratified y_hat
        
#         Inputs: 
#         x - item indices involved in the choice set or chosen set of interest. size: batch_size x maximum sequence length x 2
#         x_extra - Whose choice, len of choice set, and len of chosen set. Used to determine padding. siz: batch_size x 3
#         inf_weight - used to "zero out" padding terms. Should not be changed.
#         """    
#         batch_size, seq_len, _ = x.size()
#         choice_sets = x[:,:,0]
#         chosen_sets = x[:,:,1]
#         utilities = t.zeros((batch_size, seq_len, 1))
        
#         if self.fixed_effects:
#             logits = t.zeros((batch_size, seq_len, 1))
#             print(np.minimum(x_extra[:,2], self.k-1).size())
#             print(self.logits.size())
#             print(self.logits[np.minimum(x_extra[:,2], self.k-1),:].size())
#             print(x[:, :, 0].size())
#             print(self.logits[np.minimum(x_extra[:,2], self.k-1), :][np.arange(batch_size)[:,None], choice_sets[choice_sets!=self.num_items]])
#             logits[np.arange(batch_size)[:,None], choice_sets!=self.num_items] += self.logits[np.minimum(x_extra[:,2], self.k-1), :][np.arange(batch_size)[:,None], choice_sets[choice_sets!=self.num_items]]
# #             logits = t.index_select(self.logits[np.minimum(x_extra[:,2], self.k-1),:], x[:, :, 0])
# #             for i in range(self.k-1):
# #                 logits[x_extra[:,2]==i, :, :] = self.logits[i](x[x_extra[:,2]==i, :, 0])
# #             logits[x_extra[:,2]>=(self.k-1), :, :] = self.logits[-1](x[x_extra[:,2]>=(self.k-1), :, 0])
#             utilities += logits #batch_size x max_set_len
#             pass
        
#         if self.linear_terms:
#             cov = self.covariates[x_extra[:,0,None], x[:,:,0]] # batch_size x max_set_len x num_features matrix
# #             linear = t.zeros((batch_size, seq_len, 1))
#             linear = self.beta[np.minimum(x_extra[:,2], self.k-1)](cov[:, :, :])
# #             for i in range(self.k-1):
# #                 rows = x_extra[:,2]==i
# #                 linear[rows, :, :] = self.beta[i](cov[rows, :, :])
# #             rows = x_extra[:,2]>=(self.k-1)
# #             linear[rows, :, :] = self.beta[-1](cov[rows, :, :])
#             utilities += linear
#             pass
        
#         if self.no_ego & self.context:
#             for i in range(1,self.k-1):
#                 rows = x_extra[:,2]==i
#                 context_vecs = self.context_embeddings[i](x[rows,:,1]) #self.layer1(self.target_embedding(x))
#                 context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
#                 utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) * context_vecs).sum(-1,keepdim=True)/float(i)
#             rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,2]!=0.0)
#             context_vecs = self.context_embeddings[-1](x[rows,:,1])
#             context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
#             utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) * context_vecs).sum(-1,keepdim=True).div(x_extra[rows,2][:,None,None])
#         elif (not self.no_ego) & self.context:
#             for i in range(1,self.k-1):
#                 rows = x_extra[:,2]==i
#                 context_vecs = self.context_embeddings[i](x[rows,:,1])
#                 context_vecs = context_vecs.sum(-2)[...,None]
#                 utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) @ context_vecs)/float(i)
#             rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,2]!=0.0)
#             context_vecs = self.context_embeddings[-1](x[rows,:,1])
#             context_vecs = context_vecs.sum(-2)[...,None]
#             utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) @ context_vecs).div(x_extra[rows,2][:,None,None])
#         else:
#             pass

#         if self.ism:
#             utilities[t.arange(seq_len)[None, :] >= x_extra[:, 1, None]] = inf_weight
#         return F.log_softmax(utilities, 1)
    
#     def loss_func(self, y_hat, y, x_extra=None):
#         """
#         Evaluates the Choice Model
#         Inputs: 
#         y_hat - the log softmax values that come from the forward function
#         y - actual labels - the choice in a set (must be less than x_lengths)
#         x_lengths - the size of the choice set, used to determine padding. 
#         The current implementation assumes that y are less than x_lengths, so this
#         is unused.
#         """
#         if (self.k<=1):
#             loss = F.nll_loss(y_hat, y[:, None].long())
#             self.last_loss = loss
#             return (loss, [loss, t.zeros(1)])
#         else:
#             terms = []
#             for i in range(self.k-1):
#                 terms.append(F.nll_loss(y_hat[x_extra[:,2]==i], y[x_extra[:,2]==i, None].long()))
#             terms.append(F.nll_loss(y_hat[x_extra[:,2]>=self.k-1], y[x_extra[:,2]>=self.k-1, None].long()))
            
#             tl = F.nll_loss(y_hat, y[:, None].long())
#             rl = self.lambda_reg*self.reg()
#             terms.append(rl)
#             return (tl + rl, terms)

#     def acc_func(self, y_hat, y, x_lengths=None):
#         return (y_hat.argmax(1).int() == y[:,None].int()).float().mean()

def get_data(train_ds, val_ds, batch_size=None):
    # Note: can change val_bs to 2* batch_size if ever becomes a problem
    if batch_size is not None:
        tr_bs, val_bs = (batch_size, len(val_ds[0]))
    else: 
        tr_bs, val_bs = (len(train_ds[0]), len(val_ds[0]))

    train_dl = DataLoader(train_ds, batch_size=tr_bs, shuffle=batch_size is not None)
    val_dl = DataLoader(val_ds, batch_size=val_bs)
    return train_dl, val_dl

def loss_batch(model, xb, yb, xlb, opt=None, retain_graph=None):
    if opt is not None:
        loss, terms = model.loss_func(model(xb, xlb), yb, xlb)

        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()
    else:
        with t.no_grad():
            loss, terms = model.loss_func(model(xb, xlb), yb, xlb)

    return loss, terms

def acc_batch(model, xb, yb, xlb):
    with t.no_grad():
        return model.acc_func(model(xb, xlb), yb, xlb)

def fit(epochs, model, opt, train_dl, verbose=True, epsilon=1e-4, val_dl=None):
    val_loss = t.zeros(1)
    losses=[]
    loss1=-1.0
    loss2=0.0
    epoch=0
    while (np.abs(loss2-loss1)>epsilon) & (epoch<epochs):
        loss1=loss2
        epoch+=1
        model.train()  # good practice because these are used by nn.BatchNorm2d and nn.Dropout
        for xb, xlb, yb in train_dl:
            loss, terms = loss_batch(model, xb, yb, xlb, opt, retain_graph=None if epoch != epochs - 1 else True)
        loss2 = float(loss.detach().numpy())
        losses.append([float(t.detach().numpy()) for t in terms])
        if val_dl is not None:
            model.eval() # good practice like model.train()
            val_loss = [loss_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            val_loss = sum(val_loss)/len(val_loss)
            val_acc = [acc_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            val_acc = sum(val_acc) / len(val_acc)
            if (epoch%25==0) & verbose:
                print(f'Epoch: {epoch}, Training Loss: {loss2}, Val Loss: {val_loss}, \
                    Val Accuracy {val_acc}')
        else:
            if (epoch%25==0) & verbose:
                print(f'Epoch: {epoch}, Training Loss: {loss2}')

    return loss2, epoch, losses, val_loss.numpy()

def train_stratified(ds, num_items, ism=True, batch_size=None, epochs=500,
    lr=1e-3, seed=2, wd=1e-4, Model=StratifiedLinear, val_ds=None, verbose=True, **kwargs):
    
    tr_bs = batch_size if batch_size is not None else 1000
    if val_ds is not None:
        train_dl, val_dl = get_data(ds, val_ds, batch_size=batch_size)
    else:
        train_dl = DataLoader(ds, batch_size=tr_bs, shuffle=batch_size is not None)
        val_dl = None
    if seed is not None:
        t.manual_seed(seed)
    model = Model(num_items, ism, **kwargs)
    opt = t.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # opt = t.optim.Adam(model.parameters(), lr=lr)

    s = time.time()
    tr_loss, num_epochs, losses, val_loss = fit(epochs, model, opt, train_dl, verbose=verbose, val_dl=val_dl)
    if verbose:
        print(f'Runtime: {time.time() - s}')
        print(f'Loss: {tr_loss}')
        
    return model, tr_loss, num_epochs, losses, val_loss
