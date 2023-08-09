import hashlib
import collections
import networkx as nx


def remove_suffix(module_str):
    if '.' not in module_str:
        return module_str
    split_string = module_str.split('.')
    split_string.pop()
    to_return = '.'.join([i for i in split_string])
    return to_return

def get_hashed_names(arch):
    '''
    Apply a consistent naming scheme to parameters that 
    is a function of its metadata configuration layer structure
    for transfer learning.
    Arguments: Pytorch model
    Returns: Predecessor graph and a dictionary mapping original 
    names to mapped names
    '''
    graph = nx.DiGraph()
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    modules_list = list(arch.named_modules())
    params_list = list(arch.named_parameters())
    name_dict = {}

    #Note: Since the NAS search space doesn't include skip connections as one of its choices - all architectures
    #have the same layer connectivity/topology so we can create a sequential graph (unlike nasbench201). 
    for u, v in zip(params_list, params_list[1:]):
        name_u = u[0]
        name_v = v[0]
        graph.add_edge(remove_suffix(name_u), remove_suffix(name_v))

    #name dict maps each module to its configuration
    for name, module in modules_list:
        name_dict[name] = str(module)

    for name, module in params_list:
        name_without_suffix = remove_suffix(name)
        if name_without_suffix in name_dict:
            hash_name = name_dict[name_without_suffix]
        else:
            #There are some params (e.g like input tensors that aren't in modules)
            hash_name = name_without_suffix
            
        layer_hash = hashlib.sha3_512()
        layer_hash.update(hash_name.encode())
        for pred_name in graph.predecessors(name_without_suffix):
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        base_name = layer_hash + hash_name
        hashed_names[name] = base_name + "_" + str(counts[base_name])
        counts[base_name] += 1
    return graph, hashed_names


def process_pred_graph(hashed_names, digraph):
    '''
    Returns a graph with new hashed names as nodes
    '''
    new_graph = {}
    for node in digraph.nodes:
        if hashed_names[node] not in new_graph:
            new_graph[hashed_names[node]] = []
        neighbors = list(digraph.predecessors(node))
        new_graph[hashed_names[node]].extend(list(set([hashed_names[i] for i in neighbors])))
    return new_graph
