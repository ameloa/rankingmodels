import numpy as np
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm

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
        if shuffle:
            random_idx = np.arange(self.data_size)
            np.random.shuffle(random_idx)
            self.data = [item[random_idx] for item in self.data]
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
    The (rank) stratified MNL model, as presented in the following paper:
        Rank-heterogeneous preference models for school choice
    by authors Amel Awadelkarim, Arjun Seshadri, Itai Ashlagi, Irene Lo, and Johan Ugander
    """
    def __init__(self, num_items, ism, 
                 fixed_effects=False, item_to_school=None, item_to_program=None,
                 linear_terms=False, covariates=None, 
                 context=False, forward_dependent=False, embedding_dim=5, 
                 k=1, lambda_reg=1.0, order_reg=2, 
                 top_k=None, no_ego=True, eps=1e-2, num_ranks=5):
        """
        Initializes the stratified MNL model. 
        Utility can include fixed-effects, linear terms, and/or context-effects.
        
        Inputs: 
        num_items - int, total number of items in the choice system modeled
        ism - bool, True if dataset is multi-set, in which case padding is used
        fixed_effects - bool, if True, item-level fixed effects are used in utility
        item_to_school - array mapping offerings to their respective school index (only needed if fixed_effects is True)
        item_to_program - array mapping offerings to their respective program type index (only needed if fixed_effects is True)
        linear_terms = bool, if True, linear terms are used in utility
        covariates - (n_obs, num_items, num_features) array of covariates
        context - bool, if True, context-effects are used in utility (ie. model is a CDM)
        forward_dependent - bool, if True, forward-dependent CDM is used, else backward-dependent (unused in paper)
        embedding_dim - int > 0, embedding dimension of the low-rank CDM.
        k - int > 0, number of stratification buckets
        lambda_reg - float, weight of Laplacian regularization
        order_reg - int > 0, order of Laplacian regularization (always 2 in paper)
        top_k - int or None, rank of the last considered context effect in the top-k CDM (only relevant if forward_dependent is False)
        no_ego - always True in paper
        eps - float, minimum loss improvement for early stopping.
        num_ranks - int > 0, number of rank positions to return loss.
        """
        super().__init__()
        self.num_items = num_items
        self.ism = ism
        self.fixed_effects = fixed_effects
        if self.fixed_effects:
            self.num_schools=np.unique(item_to_school).size
            self.num_program_types=np.unique(item_to_program).size
            self.item_to_school = t.from_numpy(np.append(item_to_school, self.num_schools)).long()
            self.item_to_program_type = t.from_numpy(np.append(item_to_program, self.num_program_types)).long()
        self.linear_terms = linear_terms
        self.context = context
        self.forward_dependent = forward_dependent
        if forward_dependent & (top_k is not None):
            raise Exception("Forward + top_k CDMs are not compatible.")
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
        school_logits = []
        program_logits = []
        if self.fixed_effects:
            for i in range(self.k):
                school_logits.append(Embedding(
                num_embeddings=self.num_schools + 1,  # +1 for the padding
                embedding_dim=1,
                padding_idx=self.num_schools)) # requires_grad=True
                program_logits.append(Embedding(
                num_embeddings=self.num_program_types + 1,  # +1 for the padding
                embedding_dim=1,
                padding_idx=self.num_program_types)) # requires_grad=True
        self.program_logits = nn.ModuleList(program_logits)
        self.school_logits = nn.ModuleList(school_logits)
                   
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
                                        padding_idx=self.num_items,
                                        _weight=t.zeros([self.num_items+1, self.embedding_dim])))
                context.append(Embedding(num_embeddings=self.num_items+1,
                                         embedding_dim=self.embedding_dim,
                                         padding_idx=self.num_items))
        self.target_embeddings = nn.ModuleList(target)
        self.context_embeddings = nn.ModuleList(context)

    def reg(self):
        '''
        Computes regularization term for loss

        self.school_logits & self.program_logits are ModuleLists of Embeddings
        self.beta is a ModuleList of t.nn.Linear()
        self.targets & self.context are ModuleLists of Embeddings
        self.order_reg is the order of regularization (either L1 or L2 in this case)
        self.k is the number of stratified models learned (ie. length of self.beta)
        '''
        fe_reg = t.zeros(1) if not self.fixed_effects else t.sum(t.stack([t.linalg.norm(t.sub(self.school_logits[i].weight, self.school_logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.program_logits[i].weight, self.program_logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        linear_reg = t.zeros(1) if not self.linear_terms else t.sum(t.stack([t.linalg.norm(t.sub(self.beta[i].weight, self.beta[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        context_reg = t.zeros(1) if ((not self.context) | (self.k<=2)) else t.sum(t.stack([t.linalg.norm(t.sub(self.target_embeddings[i].weight, self.target_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.context_embeddings[i].weight, self.context_embeddings[i-1].weight), ord=self.order_reg)**2 for i in range(2,self.k)]))
        
        return fe_reg + linear_reg + context_reg
    
    def forward(self, x, x_extra=None, covariates=None, inf_weight=float('-inf')):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len, _ = x.size()
        utilities = t.zeros((batch_size, seq_len))
        if covariates is None:
            covariates=self.covariates
        else:
            num_agents, num_alternatives, num_features = covariates.shape
            pad_mat = np.zeros((num_agents, num_features), dtype=np.float32)
            covariates = np.hstack([covariates, pad_mat[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
            covariates = t.from_numpy(covariates)

        
        if self.fixed_effects:
            logits = t.zeros((batch_size, seq_len))
            
            for i in range(self.k-1):
                rows = x_extra[:,2]==i
                row_data = x[rows,:,0]
                rows_to_programs = self.item_to_program_type[row_data]
                program_logits = self.program_logits[i](rows_to_programs).squeeze()
                rows_to_schools = self.item_to_school[row_data]
                school_logits = self.school_logits[i](rows_to_schools).squeeze()
                logits[rows, :] = school_logits + program_logits
                
            rows = x_extra[:,2]>=(self.k-1)
            row_data = x[rows,:,0]
            rows_to_programs = self.item_to_program_type[row_data]
            program_logits = self.program_logits[-1](rows_to_programs).squeeze()
            rows_to_schools = self.item_to_school[row_data]
            school_logits = self.school_logits[-1](rows_to_schools).squeeze()
            logits[rows, :] = school_logits + program_logits

            utilities += logits # (batch_size, seq_len)
            pass
        
        if self.linear_terms:
            cov = covariates[x_extra[:,0,None], x[:,:,0]] 
            linear = t.zeros((batch_size, seq_len, 1))
            for i in range(self.k-1):
                rows = x_extra[:,2]==i
                linear[rows, :, :] = self.beta[i](cov[rows, :, :])
            rows = x_extra[:,2]>=(self.k-1)
            linear[rows, :, :] = self.beta[-1](cov[rows, :, :])
            utilities += linear
            pass
        
        if self.context & self.forward_dependent:
            for i in range(self.k-1):
                rows = (x_extra[:,2]==i) & (x_extra[:,1]>=2)
                context_vecs = self.context_embeddings[i](x[rows,:,0]) # num_rows x num_items x embedding_dim
                context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
                utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) * context_vecs).sum(-1).div((x_extra[rows,1]-1.)[:,None])
            rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,1]>=2)
            context_vecs = self.context_embeddings[-1](x[rows,:,0])
            context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
            utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) * context_vecs).sum(-1).div((x_extra[rows,1]-1.)[:,None]) 
        elif self.context & ~self.forward_dependent:
            for i in range(1,self.k-1):
                rows = x_extra[:,2]==i
                context_vecs = self.context_embeddings[i](x[rows,:self.top_k,1])
                context_vecs = context_vecs.sum(-2, keepdim=True)
                utilities[rows] += (self.target_embeddings[i](x[rows,:,0]) * context_vecs).sum(-1).div(x_extra[rows,2][:,None])
            rows = (x_extra[:,2]>=(self.k-1)) & (x_extra[:,2]>0)
            context_vecs = self.context_embeddings[-1](x[rows,:self.top_k,1])
            context_vecs = context_vecs.sum(-2, keepdim=True) 
            utilities[rows] += (self.target_embeddings[-1](x[rows,:,0]) * context_vecs).sum(-1).div(x_extra[rows,2][:,None]) # average context effect
        else:
            pass

        if self.ism:
            utilities[t.arange(seq_len)[None, :] >= x_extra[:, 1, None]] = inf_weight
        return F.log_softmax(utilities, 1).squeeze()
    
    def loss_func(self, y_hat, y, x_extra=None, train=True):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        train - bool. if True, returns full training loss (incl. regularization loss)
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
            tl = F.nll_loss(y_hat, y.long())
            if train:
                rl = self.lambda_reg*self.reg()
                terms.append(rl)
                return (tl + rl, terms)
            else:
                return (tl, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()

class NestedMNL(nn.Module):
    """
    Model class for the nested MNL. 
    
    Representative utility may include alternative fixed-effects and linear 
    terms, but no context terms.
    """
    def __init__(self, num_items, ism, item_to_school, item_to_program,
                 num_nests, nest_memberships, covariates, 
                 k=1, lambda_reg=1.0, order_reg=2, num_ranks=5):
        """
        Inputs: 
        num_items - the number of items in the choice system modeled (number of programs)
        ism - if dataset is multi-set, in which case padding is used
        item_to_school - array mapping offerings to their respective school index
        item_to_program - array mapping offerings to their respective program type index
        num_nests - the number of nests in the choice system modeled
        nest_memberships - list of length num_items dictating which nest the alternative belongs to
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
        k - int > 0, number of stratification buckets
        lambda_reg - float, weight of Laplacian regularization
        order_reg - int > 0, order of Laplacian regularization (always 2 in paper)
        num_ranks - int > 0, number of rank positions to return loss.
        """
        super().__init__()
        self.num_items = num_items
        self.num_schools=np.unique(item_to_school).size
        self.num_program_types=np.unique(item_to_program).size
        self.item_to_school = t.from_numpy(np.append(item_to_school, self.num_schools)).long()
        self.item_to_program_type = t.from_numpy(np.append(item_to_program, self.num_program_types)).long()
        
        self.num_nests = int(num_nests)
        self.nest_memberships = t.from_numpy(nest_memberships).type(t.long)
        self.B = {i:[] for i in range(num_nests)}
        [self.B[nest].append(i) for i, nest in enumerate(nest_memberships)]

        self.ism = ism
        num_agents, _, self.num_features = covariates.shape
        pad_vec = np.zeros([num_agents, self.num_features], dtype=np.float32)
        covariates = np.hstack([covariates, pad_vec[:,None,:]]) # add zeros matrix to second dimension of covariates (column-wise)
        self.covariates =  t.from_numpy(covariates) # num_items+1 x num_features tensor
            
        self.num_ranks = num_ranks
        self.last_loss = None
        
        self.k = k
        self.lambda_reg = lambda_reg
        self.order_reg = order_reg

        self.__build_model()

    def __build_model(self):
        """
        Helper function to initialize the model
        """    
        
        scale = []
        for i in range(self.k):
            scale.append(Embedding(
            num_embeddings=self.num_nests,
            embedding_dim=1,
            _weight=t.ones([self.num_nests, 1]))) # requires_grad=True
        self.scale_parameter = nn.ModuleList(scale)
                
        school_logits = []
        program_logits = []
        for i in range(self.k):
            school_logits.append(Embedding(
            num_embeddings=self.num_schools + 1,  # +1 for the padding
            embedding_dim=1,
            padding_idx=-1)) # requires_grad=True
            program_logits.append(Embedding(
            num_embeddings=self.num_program_types + 1,  # +1 for the padding
            embedding_dim=1,
            padding_idx=-1)) # requires_grad=True
        self.school_logits = nn.ModuleList(school_logits)
        self.program_logits = nn.ModuleList(program_logits)
           
        beta = []
        for i in range(self.k):
            beta.append(t.nn.Linear(self.num_features, 1, bias=False))
        self.beta = nn.ModuleList(beta)
        
    def compute_utils(self, x_extra, covariates=None, inf_weight=float('-inf')):
        batch_size = x_extra.size()[0]
        utilities = t.zeros((batch_size, self.num_items+1))
        
        logits = t.zeros(utilities.size())
        for i in range(self.k-1):
            rows = x_extra[:,2]==i
            logits[rows, :] = (self.school_logits[i](self.item_to_school) + self.program_logits[i](self.item_to_program_type)).reshape(1, self.num_items+1)
        rows = x_extra[:,2]>=(self.k-1)
        logits[rows, :] = (self.school_logits[-1](self.item_to_school) + self.program_logits[-1](self.item_to_program_type)).reshape(1, self.num_items+1)
        utilities += logits
        
        cov = covariates[x_extra[:,0]]
        linear = t.zeros(utilities.size())
        for i in range(self.k-1):
            rows = x_extra[:,2]==i
            linear[rows, :] = self.beta[i](cov[rows]).squeeze()
        rows = x_extra[:,2]>=(self.k-1)
        linear[rows, :] = self.beta[-1](cov[rows]).squeeze()
        utilities += linear
        
        return utilities[:,:-1]

    def compute_nested_logprobabilities(self, x, x_extra, C, covariates, sampling=False, inf_weight=float('-inf')):
        '''
        Inputs:
        x -  t.Tensor(int) of size (batch_size, max_seq_len). Padded tensor of choice sets. For j < choice_set_len, x[i,j] is some item in the current choice set of observation i. 
        x_extra - t.Tensor(int) of size (batch_size, 3). Indicates ballot owner, choice_set_len, and chosen_set_len (which rank position the observation corresponds to).
        C - np.ndarray(int) of size (batch_size, num_items). Indicators for which items are in the choice set. C[i,j] is 1 if item j is in the choice set of observation i, and 0 otherwise.
        covariates - t.Tensor(float) of size (batch_size, num_items+1, d)
        sampling - bool, if True returns log-probabilities tensor with -infs to be sampled from.
        inf_weight - used to "zero out" probabilities. Should not be changed.
        '''
        batch_size = x.size()[0]
        
        V = self.compute_utils(x_extra, covariates=covariates)
        elem_to_scale_param = t.ones(V.shape)
        for i in range(self.k-1):
            rows = x_extra[:,2]==i
            row_mask = t.tile(rows[:,None], (1, self.num_items))
            for j in range(self.num_nests):
                col_mask = t.full(elem_to_scale_param.shape, False)
                col_mask[:, self.B[j]] = True
                mask = row_mask & col_mask
                elem_to_scale_param[mask] = self.scale_parameter[i](j)
        rows = x_extra[:,2]>=(self.k-1)
        row_mask = t.tile(rows[:,None], (1, self.num_items))
        for j in range(self.num_nests):
            col_mask = t.full(elem_to_scale_param.shape, False)
            col_mask[:, self.B[j]] = True
            mask = row_mask & col_mask
            elem_to_scale_param[mask] = self.scale_parameter[-1](j)

        # Compute scaled utilities for all agents and items
        scaled_V = t.div(V, elem_to_scale_param)
                
        # Compute elements of P_lower
        scaled_V[~C.bool()] = inf_weight # Mask out utilities not in the choice set.
        a_num = scaled_V
        a_denom = t.zeros(a_num.size())
        for i in range(self.num_nests):
            a_denom[:, self.B[i]] = a_num[:, self.B[i]].logsumexp(1, keepdim=True)
        zero_idx = (a_denom == inf_weight) # The nest is not in the choice set
        a_denom[zero_idx] = 0. # These elements will not be used anyway -- set to 1.

        # Compute numerator of P_upper.
        a_temp = t.mul(a_denom, elem_to_scale_param)
        b_num = t.full((batch_size, self.num_nests), fill_value=inf_weight)
        for i in range(self.num_nests):
            b_num[:,i] = a_temp[:, self.B[i][0]]
                
        # Compute P_lower and P_upper
        log_P_lower = a_num - a_denom # (batch_size, num_items) 
        log_P_upper = F.log_softmax(b_num, 1) # (batch_size, num_items)

        # Add padding dimension for indexing with x, as x contains padding of 'self.num_items'
        log_P_lower = t.hstack((log_P_lower, t.full((batch_size, 1), inf_weight)))
        log_P_upper = t.hstack((log_P_upper, t.full((batch_size, 1), inf_weight)))
            
        # Nest membership of each item in x
        x_to_memberships = t.full(x.size(), fill_value=self.num_nests)
        x_to_memberships[x!=self.num_items] = self.nest_memberships[x[x!=self.num_items]]
        
        # Select elements from lower and upper tensors according to choice sets
        # Multiply for final probabilities
        log_P_lower_selected = log_P_lower[np.arange(batch_size)[:,None], x]
        log_P_upper_selected = log_P_upper[np.arange(batch_size)[:,None], x_to_memberships]
        
        # Compute final probabilities
        log_P = log_P_lower_selected + log_P_upper_selected
        if t.any(t.all(log_P <= inf_weight, dim=1)):
            print('Found all 0s row')
            return
        
        if sampling:
            return log_P
        else:
            zero_idx = (log_P == inf_weight)
            log_P[zero_idx] = 0. # Will not be seen anyway
            return log_P
                
    def forward(self, x, x_extra, covariates=None, inf_weight=float('-inf')):
        """
        Forward propagation, computing y_hat
        
        Inputs: 
        x - (batch_size, maximum sequence length, 2) array of item indices involved in the choice set or chosen set of interest.
        x_extra - (batch_size, 3) array, columns representing chooser, length of choice set, and length of chosen set for each observation in x.
        covariates - (num_agents, num_items, num_features)-sized numpy array of covariates
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
        
        choice_set_indicators = t.zeros((batch_size, self.num_items+1))
        choice_set_indicators[np.arange(batch_size)[:,None], x[...,0]] = 1
        
        log_p = self.compute_nested_logprobabilities(x[...,0], x_extra, choice_set_indicators[:,:-1], covariates, inf_weight=inf_weight)
        return log_p
    
    def reg(self):
        '''
        Computes regularization term for loss

        self.school_logits & self.program_logits are ModuleLists of Embeddings
        self.beta is a ModuleList of t.nn.Linear()
        self.scale_parameter is ModuleList of Embeddings
        self.order_reg is the order of regularization (either L1 or L2 in this case)
        self.k is the number of stratified models learned (ie. length of self.beta)
        '''
        fe_reg = t.sum(t.stack([t.linalg.norm(t.sub(self.school_logits[i].weight, self.school_logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)])) + t.sum(t.stack([t.linalg.norm(t.sub(self.program_logits[i].weight, self.program_logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        # t.sum(t.stack([t.linalg.norm(t.sub(self.logits[i].weight, self.logits[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        linear_reg = t.sum(t.stack([t.linalg.norm(t.sub(self.beta[i].weight, self.beta[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        scale_reg = t.sum(t.stack([t.linalg.norm(t.sub(self.scale_parameter[i].weight, self.scale_parameter[i-1].weight), ord=self.order_reg)**2 for i in range(1,self.k)]))
        
        return fe_reg + linear_reg + scale_reg
    
    def loss_func(self, y_hat, y, x_extra=None, train=True):
        """
        Evaluates the model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (i-th entry must be less than x_extra[i,1])
        x_extra - observation metadata (same as in forward function)
        train - bool. if True, returns full training loss (incl. regularization loss)
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
            tl = F.nll_loss(y_hat, y.long())
            if train:
                rl = self.lambda_reg*self.reg()
                terms.append(rl)
                return (tl + rl, terms)
            else:
                return (tl, terms)

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y.int()).float().mean()

    
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

def train(ds, num_items, ism=True, batch_size=None, epochs=500, lr=1e-3, seed=2, wd=1e-4, Model=StratifiedLinear, val_ds=None, verbose=True, **kwargs):
    
    tr_bs = batch_size if batch_size is not None else 1000
    if val_ds is not None:
        train_dl, val_dl = get_data(ds, val_ds, batch_size=batch_size)
    else:
        train_dl = DataLoader(ds, batch_size=tr_bs, shuffle=batch_size is not None)
        val_dl = None
    if seed is not None:
        t.manual_seed(seed)
    model = Model(num_items, ism, **kwargs)
    no_params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])    
    print('No. params: ', no_params)
    opt = t.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    s = time.time()
    tr_loss, num_epochs, losses, val_loss = fit(epochs, model, opt, train_dl, verbose=verbose, val_dl=val_dl)
    if verbose:
        print(f'Runtime: {time.time() - s}')
        print(f'Loss: {tr_loss}')
        
    return model, tr_loss, num_epochs, losses, val_loss
