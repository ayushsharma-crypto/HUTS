import numpy as np
from tqdm import tqdm as tq
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
from itertools import combinations, chain, product
import matplotlib.pyplot as plt
import networkx as nx
import argparse

def perform_kmeans(data, k):
	model = KMeans(n_clusters=k, init="k-means++",  random_state=42) 
	return model.fit(data)

def kmean_score_list(data, k_list=range(2,120)):
	silhouette_scores = []
	elbow_scores = []
	for k in k_list:
		model = perform_kmeans(data, k)  					 # applied kmeans clustering fitted estimator
		es = model.inertia_ 								 # calculate elbow method score
		ss = metrics.silhouette_score(data, model.labels_)   # calculate silhouette_score
		silhouette_scores += [ss]
		elbow_scores += [es]
	return k_list, silhouette_scores, elbow_scores

def top_rank_cluster_number(k_list, silhouette_scores, elbow_scores):
	elbow = KneeLocator(k_list, elbow_scores, curve='convex', direction='decreasing').knee
	silhouette = np.argmax(silhouette_scores) + 1
	return elbow, silhouette

def compute_cluster_number_frequency(data, iterations=100, k_list=range(2,120)):
	k_freq = np.zeros((2,len(k_list)+1))
	for i in tq(range(iterations)):
		_, silhouette_scores, elbow_scores = kmean_score_list(data, k_list)
		print("Elbow score = ", elbow_scores)
		print("Silhouette score = ", silhouette_scores)
		elbow_top_rank, silhouette_top_rank = top_rank_cluster_number(k_list, silhouette_scores, elbow_scores)
		k_freq[0][elbow_top_rank] += 1
		k_freq[1][silhouette_top_rank] += 1
	print(k_freq)
	return k_freq[0], k_freq[1]

def compute_pca(feature, rel_thresh=0.01):
	pd_feature = pd.DataFrame(feature)

	for col in pd_feature.columns:
		pd_feature[col] = (pd_feature[col] - pd_feature[col].mean() ) / pd_feature[col].std()
	
	previous_percent = 0
	current_percent = 0
	relative_tolerance = rel_thresh
	best_components = -1
	for i in range(1,min(len(feature),len(pd_feature.columns))):
		pca = PCA(n_components=i)
		pca_result = pca.fit_transform(pd_feature)
		current_percent = np.sum(pca.explained_variance_ratio_)
		if current_percent-previous_percent>relative_tolerance:
			previous_percent = current_percent
			best_components = i
		else:
			break
	print("Relevant top", best_components, " pca-components! They got Cumulative variance = ", 100*current_percent)
	print("Variation per principal component:\n",pca.explained_variance_ratio_)
	return pca_result, pca

class SeqGraph:
	"""
	class to build a graph for the sequence as node
	and edges weight will be similarity between the
	sequence representatives.
	"""
	def __init__(self, seq_id, seq_label,seq_color, seq_rep):
		self.seq_id = seq_id
		self.seq_rep = seq_rep
		self.seq_label = seq_label
		self.seq_color = seq_color
		self.net = None
		self.possible_metric = ["EXP_DOT", "NEG_EXP_ED", "DOT", "NEG_ED"]
	
	def _get_similitude(self,A,B,metric):
		assert metric in self.possible_metric
		if metric=="DOT":
			dt = A*B
			dt = np.sum(dt, axis=1)
			dt = (dt)/(np.linalg.norm(A, axis=1)*np.linalg.norm(B, axis=1))
			return dt
		elif metric=="NEG_ED":
			dt = (A-B)**2
			dt = np.sum(dt,axis=1)
			dt = -np.sqrt(dt)
			return dt
		elif metric=="NEG_EXP_ED":
			dt = (A-B)**2
			dt = np.sum(dt,axis=1)
			dt = -np.sqrt(dt)
			return np.exp(dt)
		elif metric=="EXP_DOT":
			dt = A*B
			dt = np.sum(dt, axis=1)
			dt = (dt)/(np.linalg.norm(A, axis=1)*np.linalg.norm(B, axis=1))
			return np.exp(dt)
		else:
			raise ValueError("Unknown Metric for computing similitudes.")
	
	def _get_edges(self, metric, ts):
		id_pairs = list(combinations(self.seq_id, 2))
		id_pairs = list(chain(*id_pairs))
		id1 = np.array(id_pairs[0::2])
		id2 = np.array(id_pairs[1::2])
		ds_pairs = list(combinations(self.seq_rep, 2))
		ds_pairs = list(chain(*ds_pairs))
		ds1 = np.array(ds_pairs[0::2])
		ds2 = np.array(ds_pairs[1::2])
		similitude = self._get_similitude(ds1,ds2, metric)
		id_similitude = np.vstack((id1,id2))
		id_similitude = np.vstack((id_similitude,similitude))
		ordey_by_similitude = lambda x: -x[-1]
		id_similitude = np.array(sorted(id_similitude.T, key=ordey_by_similitude))
		# pair_len = int(len(id_similitude)*edge_percent)
		pair_idx = id_similitude[:,-1] >= ts
		# print(id_similitude)
		# return np.array(id_similitude[:pair_len])
		return np.array(id_similitude[pair_idx])

	def build_graph(self, ts=1,metric="DOT"):
		G = nx.Graph()
		G.add_nodes_from(
			self.seq_id,
			# label=self.seq_label,
			title=self.seq_label,
			color= self.seq_color
		)
		edges = self._get_edges(metric, ts)
		edges = list(map(tuple, edges))
		G.add_weighted_edges_from(edges)
		self.net = G
		return self.net

class LoopDetection():
	def __init__(self, runs, tg=1, ts=1, model="seqvlad", seq_len=5):
		self.runs=runs
		self.tg=tg
		self.ts=ts
		self.all_locs = []
		self.all_desc = []
		self.all_env_loc_count = []
		self.all_seqs_rep = []
		self.all_run_seq_count = []
		self.all_label_seqs = []
		self.possible_models = ["seqvlad"]
		assert model in self.possible_models
		self.model=model
		self.seq_pca_mat = None
		self.all_pca_seqs_rep = None
		self.seq_graph = None
		self.loop_pairs = None
		self.possible_metric = ["EXP_DOT", "NEG_EXP_ED", "DOT", "NEG_ED"]
		self.seq_len=seq_len
	
	def init_raw_info(self):
		for run in self.runs:
			pos = np.load(run["loc"])
			des = np.load(run["desc"])
			self.all_locs.extend(pos)
			self.all_desc.extend(des)
			self.all_env_loc_count.append(len(pos))
		self.all_locs = np.asarray(self.all_locs)
		self.all_desc = np.asarray(self.all_desc)
	
	def init_seq_info(self):
		# model_id = self.possible_models.index(self.model)
		for run in self.runs:
			label_seq = np.load(run["seqs"],allow_pickle=True)
			group_rep = np.load(run["seq_rep"],allow_pickle=True).item()
			curr_seqr = []
			curr_label_seq = []
			for label_idx, label in enumerate(label_seq):
				for seq_idx, seq in enumerate(label):
					[s,e] = seq
					seq_id = str(label_idx)+"_"+str(seq_idx)
					if group_rep.get('test'+str(seq_id)) is not None:
						curr_seqr.append(group_rep["test"+seq_id])
						curr_label_seq.append(seq)
			self.all_run_seq_count.append(len(curr_seqr))
			self.all_label_seqs.extend(curr_label_seq)
			self.all_seqs_rep.extend(curr_seqr)
		self.all_seqs_rep = np.asarray(self.all_seqs_rep)
		self.all_label_seqs = np.asarray(self.all_label_seqs)
	
	def reduce_seq_rep(self, X=None):
		if X is None:
			X = self.all_seqs_rep
		_,self.seq_pca_mat = compute_pca(X, 0.001)
		self.all_pca_seqs_rep = self.seq_pca_mat.transform(self.all_seqs_rep)
	
	def _initialise_seq_graph(self, ts):
		seq_id = np.array(range(len(self.all_label_seqs)))
		seq_color = np.array([ i+1 for i in range(len(self.all_run_seq_count)) for _ in range(self.all_run_seq_count[i])  ])
		seq_label = np.array([ 
			str(seq_color[i])+"["+str(self.all_label_seqs[i][0])+":"+str(str(self.all_label_seqs[i][1]))+"]" for i in range(len(self.all_label_seqs))])
		self.seq_graph = SeqGraph(seq_id, seq_color, seq_label, self.all_pca_seqs_rep)
		self.seq_net = self.seq_graph.build_graph(ts)
	
	def visualise_seq_network(self):
		fig = plt.figure(1, figsize=(10, 8))
		edges = self.seq_net.edges.data()
		for edge in edges:
			(seq_id1, seq_id2, _) = edge
			pose1_id = self.all_label_seqs[int(seq_id1)][0]
			pose2_id = self.all_label_seqs[int(seq_id2)][0]
			pose1 = self.all_locs[pose1_id]
			pose2 = self.all_locs[pose2_id]
			plt.plot([pose1[0],pose2[0]], [pose1[1], pose2[1]], 'ro',linestyle="--", alpha=0.2)
			plt.scatter([pose1[0],pose2[0]], [pose1[1], pose2[1]], s=60, c='b',marker='o',alpha=1)
			plt.text(pose1[0], pose1[1], str(int(seq_id1)))
			plt.text(pose2[0], pose2[1], str(int(seq_id2)))
		X = self.all_locs[:,0]
		Y = self.all_locs[:,1]
		plt.plot(X,Y, c='g')
		plt.show()

	def _get_gd_similitude(self,A,B,metric):
		assert metric in self.possible_metric
		if metric=="DOT":
			dt = A*B
			dt = np.sum(dt, axis=1)
			dt = (dt)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1))
			return dt
		elif metric=="NEG_ED":
			dt = (A-B)**2
			dt = np.sum(dt,axis=1)
			dt = -np.sqrt(dt)
			return dt
		elif metric=="NEG_EXP_ED":
			dt = (A-B)**2
			dt = np.sum(dt,axis=1)
			dt = -np.sqrt(dt)
			return np.exp(dt)
		elif metric=="EXP_DOT":
			dt = A*B
			dt = np.sum(dt, axis=1)
			dt = (dt)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1))
			return np.exp(dt)
		else:
			raise ValueError("Unknown Metric for computing similitudes.")
				
	def _seq_seq_matching(self, seq_id1, seq_id2, metric, tg):
		label_seq1 = self.all_label_seqs[int(seq_id1)]
		label_seq2 = self.all_label_seqs[int(seq_id2)]
		img_pairs = product(list(range(label_seq1[0], label_seq1[1]+1)), list(range(label_seq2[0], label_seq2[1]+1)))
		img_pairs = list(chain(*img_pairs))
		imsg1 = np.array(img_pairs[0::2])
		imsg2 = np.array(img_pairs[1::2])
		ds1 = self.all_desc[imsg1]
		ds2 = self.all_desc[imsg2]
		similitude = self._get_gd_similitude(ds1,ds2, metric)
		id_similitude = np.vstack((imsg1, imsg2))
		id_similitude = np.vstack((id_similitude,similitude))
		id_similitude = id_similitude.T
		pair_idx = id_similitude[:,-1]>=tg
		return id_similitude[pair_idx]
	
	def compute_loop_pairs(self, seq_matching_metric="DOT", ts=None, tg=None):
		if ts is None:
			ts = self.ts
		if tg is None:
			tg=self.tg
		self.loop_pairs = []
		self._initialise_seq_graph(ts)
		edges = self.seq_net.edges.data()
		for edge in edges:
			(seq_id1, seq_id2, _) = edge
			lp_similitude = self._seq_seq_matching(seq_id1, seq_id2, seq_matching_metric, tg)
			self.loop_pairs.extend(lp_similitude)
		return self.loop_pairs


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='Compute the number of loop pairs')

	parser.add_argument('--loc', type=str, required=True,
						help="X,Y coordinates of bot's poses")

	parser.add_argument('--desc', type=str, required=False,
						help="resnet features file location")

	parser.add_argument('--seqs', type=str, required=False,
						help="clustered sequences file location")

	parser.add_argument('--seq_rep', type=str, required=False,
						help="clustered sequences representative file location")

	# small = {
	#     "freq": 4,
	#     "loc": "data/small/all_poses_freq_4.npy",
	#     "desc": "data/small/all_feat.npy",
	#     "seqs": "data/small/results_1/groups/labels_sequence.npy",
	#     "grp_rep":"seq_desc/r18l4_seqvlad_features/small_features.npy",
	# }
	args = parser.parse_args()

	data = {
		"loc": args.loc,
		"desc": args.desc,
		"seqs": args.seqs,
		"seq_rep": args.seq_rep,
	}
	ld = LoopDetection([data])
	ld.init_raw_info()
	ld.init_seq_info()
	print("Info initialised. Now, computing PCA of GDs.")
	ld.reduce_seq_rep()
	print("PCA obtained. Now, initialising sequence graph and computing loop pairs.")
	LP = ld.compute_loop_pairs(ts=0.8,tg=0.8)
	print("Loop pair computed.")
	ld.visualise_seq_network()
	print("Total LP count = ",len(LP))