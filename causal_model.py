from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork  
import networkx as nx
import matplotlib.pyplot as plt
import dowhy
from dowhy import CausalModel

# Baca data dan lakukan pembersihan data
df = pd.read_csv('/home/pompy/causality/Bank.csv')

# Tentukan variabel target dan fitur-fitur
y = 'credit_score'  
X = df.drop(columns=[y])

# Bangun model Bayesian Network
model = BayesianNetwork()

# Tambahkan simpul ke model
model.add_nodes_from([y] + list(X.columns))

# Tambahkan hubungan (edges) antara fitur-fitur dan variabel target
for feature in X.columns:
    model.add_edge(feature, y)

# Tampilkan simpul dan hubungan
print("Nodes:", model.nodes())
print("Edges:", model.edges())

# Visualisasikan Bayesian Network
G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
plt.title('Bayesian Network DAG')
plt.savefig('/home/pompy/causality/bayesian_network.png')

# Counter causal
model_counter = CausalModel(
    data=df,
    treatment=['balance', 'age', 'credit_card', 'tenure'],  
    outcome='credit_score',  
    graph=None  
)

# Identification counterfactual
identified_estimand = model_counter.identify_effect()

# Estimation causal effect
estimate = model_counter.estimate_effect(identified_estimand,
                                         method_name="backdoor.linear_regression")
print(estimate)

# Refute estimasi
refute_results = model_counter.refute_estimate(identified_estimand, estimate,
                                               method_name="random_common_cause")
print(refute_results)
