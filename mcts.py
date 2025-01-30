import json
import math
import numpy as np
from pathlib import Path

class TreeNode:
    def __init__(self, state=None, action=None, thought=None, parent=None):
        self.state = state
        self.action = action
        self.thought = thought
        
        self.data_visit_count = 0
        self.data_total_reward = 0.0
        
        self.mcts_visit_count = 0
        self.mcts_total_reward = 0.0
        
        self.children = []
        self.parent = parent

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

def build_static_tree(trajectories):
    root = TreeNode(state='ROOT')
    
    for traj in trajectories:
        node = root
        traj_reward = traj['reward'] 
        for (state, action, thought) in traj['state_action_pairs']:
            found_child = None
            for child in node.children:
                if child.state == state and child.action == action:
                    found_child = child
                    break
            
    
            if not found_child:
                found_child = TreeNode(state=state, action=action, thought=thought)
                node.add_child(found_child)
            
    
            found_child.data_visit_count += 1
            found_child.data_total_reward += traj_reward
            
            node = found_child  

    return root


def ucb_mcts(node, parent_mcts_visits, gamma=1.41):
    if node.mcts_visit_count == 0:
        return float('inf')  
    
    exploitation = node.mcts_total_reward / node.mcts_visit_count
    exploration = gamma * math.sqrt(math.log(parent_mcts_visits) / node.mcts_visit_count)
    return exploitation + exploration

def mcts(root, iterations=1000, gamma=1.41):
    for _ in range(iterations):
        path = []
        node = root
        
        while node.children:
            parent_visits = sum(child.mcts_visit_count for child in node.children) + 1e-12
            
            node = max(node.children, key=lambda c: ucb_mcts(c, parent_visits, gamma))
            path.append(node)
        
        if node.data_visit_count > 0:
            rollout_value = node.data_total_reward / node.data_visit_count
        else:
            rollout_value = 0
        
        for nd in path:
            nd.mcts_visit_count += 1
            nd.mcts_total_reward += rollout_value

def get_best_path(root, use_mcts_stats=True):
    path_info = []
    node = root
    
    while node.children:
        if use_mcts_stats:
            node = max(node.children, key=lambda c: 
                       (c.mcts_total_reward / c.mcts_visit_count) if c.mcts_visit_count > 0 else -999999
            )
            avg_val = node.mcts_total_reward / node.mcts_visit_count if node.mcts_visit_count > 0 else 0
        else:
            node = max(node.children, key=lambda c: 
                       (c.data_total_reward / c.data_visit_count) if c.data_visit_count > 0 else -999999
            )
            avg_val = node.data_total_reward / node.data_visit_count if node.data_visit_count > 0 else 0
        
        path_info.append({
            'state': node.state,
            'action': node.action,
            'thought': node.thought,
            'avg_value': avg_val
        })
        
        if len(node.children) == 0:
            break  
    
    return path_info

def demo():
    trajectories = [
        {
            'reward': 1.0,
            'state_action_pairs': [
                ("S0", "A0", "T0"),
                ("S1", "A1", "T1"),
            ]
        },
        {
            'reward': 0.75,
            'state_action_pairs': [
                ("S0", "A0", "T0"),  
                ("S1", "A2", "T2"),  
                ("S2", "A3", "T3")
            ]
        },
        {
            'reward': 0.5,
            'state_action_pairs': [
                ("S0", "A4", "T4"),  
            ]
        }
    ]
    
    root = build_static_tree(trajectories)
    
    mcts(root, iterations=50, gamma=1.41)
    
    best_path_by_mcts = get_best_path(root, use_mcts_stats=True)
    
    best_path_by_data = get_best_path(root, use_mcts_stats=False)
    
    print(">> best path")
    for step in best_path_by_mcts:
        print(step)
    
    print("path")
    for step in best_path_by_data:
        print(step)


if __name__ == "__main__":
    demo()
