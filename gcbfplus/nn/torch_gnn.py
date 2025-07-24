import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A simple multi-layer perceptron module.
    """
    
    def __init__(self, input_dim, hidden_sizes, output_dim, activation=F.relu, final_activation=None):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activation = activation
        self.final_activation = final_activation
        
        # Input layer
        layer_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Initialize weights
            nn.init.orthogonal_(self.layers[-1].weight, gain=1.0)
            nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation function (except for the last layer if final_activation is None)
            if i < len(self.layers) - 1 or self.final_activation is not None:
                x = self.activation(x)
                
        if self.final_activation is not None:
            x = self.final_activation(x)
            
        return x


class GNNLayer(nn.Module):
    """
    A single GNN layer with message passing, attention-based aggregation, and node updates.
    """
    
    def __init__(self, node_dim, edge_dim, msg_dim, output_dim, msg_hidden_sizes=(64, 64), 
                 aggr_hidden_sizes=(64,), update_hidden_sizes=(64, 64)):
        super(GNNLayer, self).__init__()
        
        # Message network: Computes messages based on edge features and connected node features
        message_input_dim = edge_dim + node_dim * 2  # edge features + sender features + receiver features
        self.message_net = MLP(
            input_dim=message_input_dim,
            hidden_sizes=msg_hidden_sizes,
            output_dim=msg_dim
        )
        
        # Attention network: Computes attention weights for message aggregation
        self.attention_net = MLP(
            input_dim=msg_dim,
            hidden_sizes=aggr_hidden_sizes,
            output_dim=1
        )
        
        # Update network: Updates node features based on aggregated messages
        update_input_dim = node_dim + msg_dim
        self.update_net = MLP(
            input_dim=update_input_dim,
            hidden_sizes=update_hidden_sizes,
            output_dim=output_dim
        )
        
        self.msg_dim = msg_dim
        self.output_dim = output_dim
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Performs a single round of message passing, aggregation, and node updates.
        
        Args:
            node_features: Tensor of shape [num_nodes, node_dim]
            edge_index: Tensor of shape [2, num_edges] containing [sender_idx, receiver_idx]
            edge_features: Tensor of shape [num_edges, edge_dim]
            
        Returns:
            Tensor of shape [num_nodes, output_dim] with updated node features
        """
        senders, receivers = edge_index[0], edge_index[1]
        num_nodes = node_features.shape[0]
        
        # Get features for senders and receivers
        sender_features = node_features[senders]    # [num_edges, node_dim]
        receiver_features = node_features[receivers]  # [num_edges, node_dim]
        
        # Compute messages
        message_inputs = torch.cat([edge_features, sender_features, receiver_features], dim=-1)
        messages = self.message_net(message_inputs)  # [num_edges, msg_dim]
        
        # Compute attention weights
        attention_scores = self.attention_net(messages)  # [num_edges, 1]
        
        # Apply softmax to get attention weights per receiver
        # First, initialize attention to zeros
        attention_weights = torch.zeros_like(attention_scores)
        
        # For each receiver node, apply softmax over its incoming messages
        for i in range(num_nodes):
            mask = (receivers == i)
            if mask.sum() > 0:  # If the node has incoming messages
                node_scores = attention_scores[mask]
                node_weights = F.softmax(node_scores, dim=0)
                attention_weights[mask] = node_weights
        
        # Apply attention weights to messages
        weighted_messages = messages * attention_weights
        
        # Aggregate messages for each receiver
        aggregated_messages = torch.zeros(num_nodes, self.msg_dim, device=node_features.device)
        for i in range(weighted_messages.shape[0]):
            aggregated_messages[receivers[i]] += weighted_messages[i]
        
        # Update node features
        update_inputs = torch.cat([node_features, aggregated_messages], dim=-1)
        updated_features = self.update_net(update_inputs)
        
        return updated_features


class GNN(nn.Module):
    """
    Graph Neural Network for multi-agent systems.
    Can be used for both policy and CBF networks.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, output_dim=1, n_layers=2,
                 msg_hidden_sizes=(64, 64), aggr_hidden_sizes=(64,), update_hidden_sizes=(64, 64)):
        super(GNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        
        # First layer takes node_dim as input
        first_layer = GNNLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            msg_dim=hidden_dim,
            output_dim=hidden_dim,
            msg_hidden_sizes=msg_hidden_sizes,
            aggr_hidden_sizes=aggr_hidden_sizes,
            update_hidden_sizes=update_hidden_sizes
        )
        self.layers.append(first_layer)
        
        # Intermediate layers take hidden_dim as input
        for i in range(1, n_layers - 1):
            layer = GNNLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                msg_dim=hidden_dim,
                output_dim=hidden_dim,
                msg_hidden_sizes=msg_hidden_sizes,
                aggr_hidden_sizes=aggr_hidden_sizes,
                update_hidden_sizes=update_hidden_sizes
            )
            self.layers.append(layer)
        
        # Last layer outputs output_dim
        last_layer = GNNLayer(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            msg_dim=hidden_dim,
            output_dim=output_dim,
            msg_hidden_sizes=msg_hidden_sizes,
            aggr_hidden_sizes=aggr_hidden_sizes,
            update_hidden_sizes=update_hidden_sizes
        )
        self.layers.append(last_layer)
    
    def forward(self, graph_dict):
        """
        Forward pass through the GNN.
        
        Args:
            graph_dict: Dictionary containing:
                - node_features: Tensor of shape [num_nodes, node_feat_dim]
                - edge_index: Tensor of shape [2, num_edges]
                - edge_features: Tensor of shape [num_edges, edge_feat_dim]
                
        Returns:
            Tensor of shape [num_nodes, output_dim] with final node features
        """
        node_features = graph_dict['node_features']
        edge_index = graph_dict['edge_index']
        edge_features = graph_dict['edge_features']
        
        # Process through each GNN layer
        for layer in self.layers:
            node_features = layer(node_features, edge_index, edge_features)
        
        return node_features


class PolicyGNN(nn.Module):
    """
    Graph Neural Network for control policy.
    Takes graph features as input and outputs control actions.
    """
    
    def __init__(self, node_dim, edge_dim, action_dim=2, hidden_dim=64, n_layers=2):
        super(PolicyGNN, self).__init__()
        
        self.gnn = GNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            n_layers=n_layers
        )
    
    def forward(self, graph_dict):
        """
        Compute actions for each agent based on the graph structure.
        
        Args:
            graph_dict: Dictionary with graph structure
                
        Returns:
            Tensor of shape [num_agents, action_dim] with action for each agent
        """
        # The GNN computes features for all nodes
        all_actions = self.gnn(graph_dict)
        
        # For a policy, we typically only care about the actions for agent nodes
        # In this implementation, we assume all nodes are agents for simplicity
        return all_actions


class CBFGNN(nn.Module):
    """
    Graph Neural Network for Control Barrier Function.
    Takes graph features as input and outputs scalar CBF values.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim=64, n_layers=2):
        super(CBFGNN, self).__init__()
        
        self.gnn = GNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # CBF outputs a scalar value per node
            n_layers=n_layers
        )
    
    def forward(self, graph_dict):
        """
        Compute CBF values for each agent based on the graph structure.
        
        Args:
            graph_dict: Dictionary with graph structure
                
        Returns:
            Tensor of shape [num_agents, 1] with CBF value for each agent
        """
        # The GNN computes features for all nodes
        all_cbf_values = self.gnn(graph_dict)
        
        # For CBF, we only care about the values for agent nodes
        # In this implementation, we assume all nodes are agents for simplicity
        return all_cbf_values 