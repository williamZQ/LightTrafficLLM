import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from gpt4ts import GPT4TS


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features, input_dim=3):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.input_dim = input_dim
 
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):

        if self.input_dim >= 3:
            day_emb = x[..., 1]
            week_emb = x[..., 2]
        elif self.input_dim == 2:
            day_emb = x[..., 1]
            week_emb = torch.zeros_like(day_emb)
        else:
            batch_size, seq_len, num_nodes = x.shape[0], x.shape[1], x.shape[2]
            device = x.device
            
            day_emb = torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)
            day_emb = day_emb.expand(batch_size, seq_len, num_nodes) / seq_len
            
            week_emb = torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)
            week_emb = week_emb.expand(batch_size, seq_len, num_nodes) / 7.0
            week_emb = week_emb % 1.0  # Normalize to [0, 1)
        

        day_indices = (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        day_indices = torch.clamp(day_indices, 0, self.time - 1)
        time_day = self.time_day[day_indices]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_indices = (week_emb[:, -1, :]).type(torch.LongTensor)
        week_indices = torch.clamp(week_indices, 0, 6)  # 7 days in a week
        time_week = self.time_week[week_indices]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


class TreeBasedSpatialEmbedding(nn.Module):
    
    def __init__(self, num_nodes, embedding_dim, max_depth=3, max_width=10, adj_matrix_path=None):
        super(TreeBasedSpatialEmbedding, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        self.max_width = max_width
        
        self.embed_node = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.embed_node)

        self.W_self = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_child = nn.Linear(embedding_dim, embedding_dim, bias=False)

        if adj_matrix_path:
            self.adj_matrix = self._load_adj_matrix(adj_matrix_path)
        else:
            self.adj_matrix = None

        self.tree_structures = self._build_tree_structures()
        
    def _load_adj_matrix(self, adj_matrix_path):

        try:
            data = np.load(adj_matrix_path)
            adj_matrix = data['adj_matrix']
            print(f"成功加载邻接矩阵，形状: {adj_matrix.shape}")
            return torch.tensor(adj_matrix, dtype=torch.float32)
        except Exception as e:
            print(f"加载邻接矩阵失败: {e}")
            return None
    
    def _build_tree_structures(self):

        if self.adj_matrix is None:
            return None
            
        trees = []
        adj_np = self.adj_matrix.numpy()
        
        for root_node in range(self.num_nodes):
            tree = self._build_tree_for_node(root_node, adj_np)
            trees.append(tree)
        
        return trees
    
    def _build_tree_for_node(self, root_node, adj_matrix):
 
        tree = {}
        visited = set()
        queue = deque([(root_node, 0)])  
        
        while queue:
            current_node, depth = queue.popleft()
            
            if current_node in visited or depth > self.max_depth:
                continue
                
            visited.add(current_node)
            
      
            neighbors = np.where(adj_matrix[current_node] > 0)[0]
    
            if len(neighbors) > self.max_width:
                neighbors = neighbors[:self.max_width]
            
            tree[current_node] = {
                'depth': depth,
                'children': neighbors.tolist(),
                'parent': None  
            }
            
            for child in neighbors:
                if child not in visited:
                    queue.append((child, depth + 1))
        
        return tree
    
    def _tree_aggregate(self, node_embeddings, tree):

        nodes_by_depth = {}
        for node, info in tree.items():
            depth = info['depth']
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
        
     
        depths = sorted(nodes_by_depth.keys(), reverse=True)
        
   
        current_embeddings = node_embeddings.clone()
        
  
        for depth in depths:
            if depth == 0: 
                continue
                
            for node in nodes_by_depth[depth]:
                children = tree[node]['children']
                
                if children:
                    children_embeddings = current_embeddings[children]
                    
                    self_contribution = self.W_self(current_embeddings[node])
                    children_contribution = torch.sum(self.W_child(children_embeddings), dim=0)
                    
                    updated_embedding = F.relu(self_contribution + children_contribution)
              
                    new_embeddings = current_embeddings.clone()
                    new_embeddings[node] = updated_embedding
                    current_embeddings = new_embeddings
        
        return current_embeddings
    
    def forward(self, batch_size=None):

        node_emb = self.embed_node
        
        if self.tree_structures is not None:
       
            aggregated_embeddings = []
            
            for root_node in range(self.num_nodes):
                tree = self.tree_structures[root_node]
                
                if tree:
                 
                    root_embedding = self._tree_aggregate(node_emb, tree)
               
                    aggregated_embeddings.append(root_embedding[root_node])
                else:
          
                    aggregated_embeddings.append(node_emb[root_node])
            
        
            node_emb = torch.stack(aggregated_embeddings)
        
  
        if batch_size is not None:
            node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            node_emb = node_emb.transpose(1, 2).unsqueeze(-1)
        
        return node_emb


class LightTrafficLLM(nn.Module):
    def __init__(
        self,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=6,
        U=1,
        device="cuda:7",
        lora_rank=8,
        llm_path=None
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.device = device
        self.lora_rank = lora_rank
        time = 48
        if num_nodes == 170 or num_nodes == 307 or num_nodes == 157:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48

        gpt_channel = 256
        
        # 根据模型类型确定隐藏维度
        if llm_path and 'llama' in llm_path.rstrip('/').split('/')[-1].lower():
            to_gpt_channel = 3072  
        elif 'qwen3' in llm_path.rstrip('/').split('/')[-1].lower():
            to_gpt_channel = 1024   
        else:
            to_gpt_channel = 768  

        self.Temb = TemporalEmbedding(time, gpt_channel, input_dim=self.input_dim)


        from config import get_training_config
        config = get_training_config()
        self.spatial_embedding = TreeBasedSpatialEmbedding(
            num_nodes=self.num_nodes,
            embedding_dim=gpt_channel,
            max_depth=3,
            max_width=10,
            adj_matrix_path=config.adj_matrix_path
        )

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

 
        self.gpt = GPT4TS(device=self.device, gpt_layers=self.llm_layer, lora_rank=self.lora_rank, llm_path=llm_path)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )


        self.regression_layer = nn.Conv2d(
            to_gpt_channel, self.output_len, kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(p=0.3)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape
        history_data = history_data.permute(0, 3, 2, 1)

        tem_emb = self.Temb(history_data)
  
        node_emb = []
        spatial_emb = self.spatial_embedding(batch_size)
        node_emb.append(spatial_emb)

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_data = self.start_conv(input_data)

  
        input_residual = input_data
        
        data_st = torch.cat(
            [input_data] + [tem_emb] + node_emb, dim=1
        )
        data_st = self.feature_fusion(data_st)
        

        if data_st.shape == input_residual.shape:
            data_st = data_st + input_residual
        
        # data_st = F.leaky_relu(data_st)

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.dropout(data_st)
        
   
        gpt_residual = data_st
        
        data_st = self.gpt(data_st)
 
        if data_st.shape == gpt_residual.shape:
            data_st = data_st + gpt_residual
            
        data_st = self.dropout(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st)

        return prediction