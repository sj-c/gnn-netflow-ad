# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
import glob
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle

from scipy import sparse


def compute_node_embeddings(edge_index, num_nodes, embedding_dim, node_names, projection_matrix, num_iterations=3,
                            distance_weights=None):
    """
    Compute node embeddings using IPv4 addresses as initial embeddings,
    with distance-based weighting over multiple iterations.

    Args:
        edge_index (np.ndarray): Edge indices of shape (2, num_edges).
        num_nodes (int): Number of nodes in the graph.
        embedding_dim (int): Desired dimension of the embeddings.
        node_names (dict): Dictionary mapping node indices to IPv4 address strings.
        num_iterations (int): Number of iterations (k).
        distance_weights (list or None): Weights for each distance from 0 to k.
                                         If None, default weights are used.

    Returns:
        np.ndarray: Node embeddings of shape (num_nodes, embedding_dim).
    """
    # Ensure edge_index is a NumPy array
    if not isinstance(edge_index, np.ndarray):
        edge_index = np.asarray(edge_index)

    # Extract rows and columns
    rows = edge_index[0].flatten()
    cols = edge_index[1].flatten()

    # Create data array
    data = np.ones(len(rows), dtype=np.float32)

    # Create adjacency matrix A
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

    # Compute inverse degree matrix D_inv
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = np.zeros_like(degrees)
    np.divide(1.0, degrees, out=D_inv, where=degrees != 0)
    D_inv_mat = sparse.diags(D_inv)

    # Compute normalized adjacency matrix: D_inv_A = D_inv * A
    D_inv_A = D_inv_mat.dot(A)

    # Step 1: Vectorized processing of IP addresses

    # Create a list of IP addresses aligned with node indices
    ip_addresses = [''] * num_nodes
    for node_idx, ip_address in node_names.items():
        ip_addresses[node_idx] = ip_address

    # Replace empty strings with '0.0.0.0' or another default IP
    ip_addresses_list = ['0.0.0.0' if ip == '' else ip for ip in ip_addresses]

    # Convert to NumPy array
    ip_addresses_np = np.array(ip_addresses_list)

    # Use np.char methods to split IP addresses into octets
    octets = np.char.split(ip_addresses_np, sep='.')

    # Function to pad or truncate octets to length 4
    def pad_or_truncate(lst):
        if len(lst) == 4:
            return lst
        elif len(lst) < 4:
            return lst + ['0'] * (4 - len(lst))
        else:
            return lst[:4]

    # Apply the function vectorized over the array
    octets_padded = np.array([pad_or_truncate(o) for o in octets])

    # Flatten and check if entries are digits
    octets_flat = octets_padded.flatten()
    is_digit = np.char.isdigit(octets_flat)

    # Replace non-digit entries with '0'
    octets_flat[~is_digit] = '0'

    # Convert to integers
    octets_int_flat = octets_flat.astype(np.int32)

    # Reshape back to (num_nodes, 4)
    octets_int = octets_int_flat.reshape((num_nodes, 4))

    # Normalize by dividing by 255.0
    ip_embeddings_np = octets_int.astype(np.float32) / 255.0

    # Convert to NumPy array
    ip_embeddings = np.asarray(ip_embeddings_np)

    # Optional: Project IP embeddings to the desired embedding dimension
    if embedding_dim != 4:
        # Initialize a projection matrix
        # np.random.seed(42)
        # projection_matrix = np.random.uniform(-1, 1, size=(4, embedding_dim)).astype(np.float32)

        # Project the embeddings
        e0 = ip_embeddings.dot(projection_matrix)
    else:
        e0 = ip_embeddings

    # Normalize initial embeddings
    norms = np.linalg.norm(e0, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Prevent division by zero
    e0_normalized = e0 / norms

    # If distance_weights is None, define default weights inversely proportional to distance
    if distance_weights is None:
        # Distance 0 to num_iterations
        distance_weights = [1.0 / (d + 1) for d in range(num_iterations + 1)]
    else:
        assert len(distance_weights) == num_iterations + 1, "Length of distance_weights must be num_iterations + 1"

    # Precompute powers of D_inv_A
    D_inv_A_powers = [sparse.identity(num_nodes, format='csr'), D_inv_A]
    for _ in range(2, num_iterations + 1):
        D_inv_A_powers.append(D_inv_A_powers[-1].dot(D_inv_A))

    # Initialize tensor to hold embeddings at each distance
    e_all = np.zeros((num_iterations + 1, num_nodes, embedding_dim), dtype=np.float32)

    # Compute embeddings for distance 0 (initial embeddings)
    e_all[0] = e0_normalized * distance_weights[0]

    # Compute embeddings for distances 1 to num_iterations
    for d in range(1, num_iterations + 1):
        # Compute influence from nodes at distance d
        e_d = D_inv_A_powers[d].dot(e0_normalized)

        # Subtract influence from closer distances to isolate distance d
        for i in range(1, d):
            e_d -= D_inv_A_powers[d - i].dot(e_all[i])

        # Normalize embeddings
        norms = np.linalg.norm(e_d, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        e_d_normalized = e_d / norms

        # Multiply by weight and store
        e_all[d] = e_d_normalized * distance_weights[d]

    # Sum over distances to get final embeddings
    e_final = np.sum(e_all, axis=0)

    return e_final


def create_random_tensor(n, m, min_value, max_value):
    # Generate a tensor of shape (n, m) with values between 0 and 1
    random_tensor = torch.rand(n, m, dtype=torch.float32)

    # Scale and shift the values to be within the range [min_value, max_value]
    scaled_tensor = random_tensor * (max_value - min_value) + min_value

    return scaled_tensor


# Port-range boundaries (IANA)
WELL_KNOWN_PORT_MAX = 1023
REGISTERED_PORT_MAX = 49151

# Edge columns encoded as one-hot blocks instead of scaled numerics.
# Values are numeric protocol/port codes: z-scoring them would invent
# ordering/distance that doesn't exist.
CATEGORICAL_EDGE_COLUMNS = ("PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT")

# Derived numeric columns computed from raw NetFlow fields. The source
# fields only need to exist in the dataframe, not in edge_columns.
DERIVED_EDGE_COLUMNS = {
    "TOTAL_PKTS": lambda df: df["IN_PKTS"].to_numpy(dtype=np.float64) + df["OUT_PKTS"].to_numpy(dtype=np.float64),
    "TOTAL_BYTES": lambda df: df["IN_BYTES"].to_numpy(dtype=np.float64) + df["OUT_BYTES"].to_numpy(dtype=np.float64),
}


class NetflowPreprocessor:

    def __init__(self, df: pd.DataFrame, edge_columns: [str], node_dim: int, normalize=True, src_ip='IPV4_SRC_ADDR',
                 dst_ip='IPV4_DST_ADDR', label='Label', log1p_columns=None, top_k_ports=16,
                 top_k_protocols=8, protocol_vocab=None, dst_port_vocab=None):
        self.edge_columns = edge_columns
        self.node_dim = node_dim
        self.edge_scaler = None
        self.scale = normalize
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.label = label
        self.top_k_ports = top_k_ports
        self.top_k_protocols = top_k_protocols
        # order-preserving split: numerics go through the scaler, categoricals become one-hot blocks
        self.numeric_columns = [c for c in edge_columns if c not in CATEGORICAL_EDGE_COLUMNS]
        self.categorical_columns = [c for c in edge_columns if c in CATEGORICAL_EDGE_COLUMNS]
        # optional: columns to log1p-compress before scaling, so heavy-tailed
        # outliers don't dominate the scaler's mean/scale; empty list = raw features
        self.log1p_columns = [c for c in (log1p_columns or []) if c in self.numeric_columns]
        np.random.seed(42)
        self.projection_matrix = np.random.uniform(-1, 1, size=(4, self.node_dim)).astype(np.float32)
        # Assign the dataframe directly
        self.df = df.dropna()

        # Categorical vocabularies come from the (benign) training data; pass them in
        # explicitly (e.g. from a checkpoint) to reproduce a fitted encoding at inference.
        self.protocol_vocab = None
        self.dst_port_vocab = None
        if "PROTOCOL" in self.categorical_columns:
            if protocol_vocab is not None:
                self.protocol_vocab = list(protocol_vocab)
            else:
                # top-K most frequent protocols only; a single protocol scan in the
                # training data would otherwise blow the vocab up to ~254 entries.
                # Everything outside the top K falls into "other".
                counts = self.df["PROTOCOL"].value_counts()
                self.protocol_vocab = sorted(counts.head(self.top_k_protocols).index.tolist())
        if "L4_DST_PORT" in self.categorical_columns:
            if dst_port_vocab is not None:
                self.dst_port_vocab = list(dst_port_vocab)
            else:
                # top-K most frequent non-ephemeral destination ports; ephemeral ports
                # get their own bucket, everything else falls into "other"
                ports = self.df["L4_DST_PORT"]
                counts = ports[ports <= REGISTERED_PORT_MAX].value_counts()
                self.dst_port_vocab = counts.head(self.top_k_ports).index.tolist()

        if self.scale and self.numeric_columns:
            self.edge_scaler = StandardScaler().fit(self._numeric_features(self.df))

    @property
    def feature_dim(self):
        return len(self.feature_names)

    @property
    def feature_names(self):
        """Final edge-feature layout: scaled numerics first, then one-hot blocks."""
        names = list(self.numeric_columns)
        for col in self.categorical_columns:
            if col == "PROTOCOL":
                names += [f"PROTOCOL={p}" for p in self.protocol_vocab] + ["PROTOCOL=other"]
            elif col == "L4_SRC_PORT":
                names += ["SRC_PORT=wellknown", "SRC_PORT=registered", "SRC_PORT=ephemeral"]
            elif col == "L4_DST_PORT":
                names += [f"DST_PORT={p}" for p in self.dst_port_vocab] + ["DST_PORT=other", "DST_PORT=ephemeral"]
        return names

    def _column_values(self, df, col):
        if col in DERIVED_EDGE_COLUMNS:
            return DERIVED_EDGE_COLUMNS[col](df)
        return df[col].to_numpy(dtype=np.float64)

    def _numeric_features(self, df):
        """Numeric edge features, with log1p applied to any configured columns, before scaling."""
        features = np.column_stack([self._column_values(df, c) for c in self.numeric_columns])
        if self.log1p_columns:
            idx = [self.numeric_columns.index(c) for c in self.log1p_columns]
            features[:, idx] = np.log1p(np.clip(features[:, idx], 0, None))
        return features

    @staticmethod
    def _one_hot(indices, num_classes):
        block = np.zeros((len(indices), num_classes), dtype=np.float64)
        block[np.arange(len(indices)), indices] = 1.0
        return block

    def _categorical_features(self, df):
        """One-hot blocks for categorical columns; 0/1 values, deliberately unscaled."""
        blocks = []
        for col in self.categorical_columns:
            if col == "PROTOCOL":
                mapping = {p: i for i, p in enumerate(self.protocol_vocab)}
                idx = df["PROTOCOL"].map(mapping).fillna(len(mapping)).to_numpy(dtype=np.int64)
                blocks.append(self._one_hot(idx, len(mapping) + 1))
            elif col == "L4_SRC_PORT":
                # source ports are usually ephemeral noise: coarse range buckets only
                ports = df["L4_SRC_PORT"].to_numpy(dtype=np.int64)
                idx = np.where(ports <= WELL_KNOWN_PORT_MAX, 0,
                               np.where(ports <= REGISTERED_PORT_MAX, 1, 2))
                blocks.append(self._one_hot(idx, 3))
            elif col == "L4_DST_PORT":
                mapping = {p: i for i, p in enumerate(self.dst_port_vocab)}
                other, ephemeral = len(mapping), len(mapping) + 1
                ports = df["L4_DST_PORT"]
                idx = ports.map(mapping).fillna(other).to_numpy(dtype=np.int64)
                idx[ports.to_numpy(dtype=np.int64) > REGISTERED_PORT_MAX] = ephemeral
                blocks.append(self._one_hot(idx, ephemeral + 1))
        return np.concatenate(blocks, axis=1)

    def transform_edge_features(self, df):
        """Final edge-feature matrix: [scaled numerics | one-hot categoricals]."""
        parts = []
        if self.numeric_columns:
            numeric = self._numeric_features(df)
            if self.scale:
                numeric = np.asarray(self.edge_scaler.transform(numeric))
            parts.append(numeric)
        if self.categorical_columns:
            parts.append(self._categorical_features(df))
        return np.concatenate(parts, axis=1)

    def _generate_windows(self, df, window_size, step_size):
        """ Generate row-based windows instead of time-based ones. """
        windows = []
        total_rows = len(df)
        for start in range(0, total_rows - window_size + 1, step_size):
            windows.append((start, start + window_size))
        return windows

    def save_scaler(self, path='edge_scaler.pkl'):

        assert self.edge_scaler is not None, "Scaler must be initialized to save"

        with open(path, 'wb') as f:
            pickle.dump(self.edge_scaler, f)

    def load_scaler(self, path='edge_scaler.pkl'):

        with open(path, 'rb') as f:
            self.edge_scaler = pickle.load(f)

        self.scale = True

    def process_single(self, df):
        """ Converts a given pandas dataframe into a single graph. """

        ## Construct node indices
        unique_ips = pd.concat([df[self.src_ip], df[self.dst_ip]]).unique()
        ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
        idx_to_ip = {value: key for key, value in ip_to_idx.items()}
        src_idx = df[self.src_ip].map(ip_to_idx).astype(np.int64)
        dst_idx = df[self.dst_ip].map(ip_to_idx).astype(np.int64)
        edge_index = np.vstack([src_idx, dst_idx])

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index has incorrect shape: {edge_index.shape}")

        # Get edge features (scaled numerics + one-hot categoricals)
        edge_features = self.transform_edge_features(df)
        edge_labels = np.zeros(len(df))  # Dummy feature array

        # Create edge attributes tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32).unsqueeze(1)
        edge_attr = torch.cat([edge_features, edge_labels], dim=1)

        num_nodes = len(unique_ips)
        # Compute node embeddings using NumPy
        embeddings = compute_node_embeddings(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=self.node_dim,
            node_names=idx_to_ip,
            projection_matrix=self.projection_matrix,
            num_iterations=3,  # You can adjust the number of iterations
        )

        x = torch.tensor(embeddings, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data, ip_to_idx

    def construct_graph_list(self, df=None, window_size=1000, step_size=500):
        """ Constructs a list of graphs based on window size and step size. """
        if df is None:
            df = self.df

        # Generate row-based windows
        windows = self._generate_windows(df, window_size, step_size)

        # Create a list to store PyTorch Geometric Data objects
        data_list = []
        ip_map = []
        data_windows = []

        # Iterate through each window and generate PyTorch Geometric graph
        for start, end in tqdm(windows, desc='Graph Windows'):
            window_df = df.iloc[start:end]
            data_windows.append(window_df.copy())

            if not window_df.empty:
                # Create a mapping from IPs to node indices
                unique_ips = pd.concat([window_df[self.src_ip], window_df[self.dst_ip]]).unique()
                ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}
                idx_to_ip = {value: key for key, value in ip_to_idx.items()}
                ip_map.append(ip_to_idx)

                # Create edges and edge labels using pandas and NumPy
                src_idx = window_df[self.src_ip].map(ip_to_idx).astype(np.int64)
                dst_idx = window_df[self.dst_ip].map(ip_to_idx).astype(np.int64)
                edge_index = np.vstack([src_idx, dst_idx])

                # Ensure edge_index has shape (2, num_edges)
                if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                    raise ValueError(f"edge_index has incorrect shape: {edge_index.shape}")

                edge_features = self.transform_edge_features(window_df)
                edge_labels = window_df[self.label].astype(np.float32)

                # Convert to PyTorch tensors
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_features = torch.tensor(edge_features, dtype=torch.float32)
                edge_labels = torch.tensor(edge_labels.values, dtype=torch.float32).unsqueeze(1)

                # Combine edge features with labels
                edge_attr = torch.cat([edge_features, edge_labels], dim=1)

                # Use a zero-filled node feature tensor of size (number of nodes, node_dim)
                num_nodes = len(unique_ips)
                # Compute node embeddings using NumPy
                embeddings = compute_node_embeddings(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    embedding_dim=self.node_dim,
                    node_names=idx_to_ip,
                    projection_matrix=self.projection_matrix,
                    num_iterations=3,  # You can adjust the number of iterations
                )

                # Convert embeddings to PyTorch tensor
                x = torch.tensor(embeddings, dtype=torch.float32)

                # Create PyTorch Geometric Data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

        return data_list, ip_map, data_windows