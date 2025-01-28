import json
import math
import random
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class TreeNode:
    def __init__(self, state=None, action=None, thought=None, parent=None):
        self.state = state
        self.action = action
        self.thought = thought
        self.visit_count = 0
        self.total_reward = 0.0
        self.children = []
        self.parent = parent

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
state_embedding_cache = {} 
SIMILARITY_THRESHOLD = 0.9  

def are_states_similar(state1, state2):
    if state1 == state2:
        return True
    if state1 not in state_embedding_cache:
        state_embedding_cache[state1] = embedding_model.encode(state1)
    if state2 not in state_embedding_cache:
        state_embedding_cache[state2] = embedding_model.encode(state2)
    embedding1 = state_embedding_cache[state1]
    embedding2 = state_embedding_cache[state2]
    cos_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return cos_sim >= SIMILARITY_THRESHOLD

def ucb(node, total_visits, c=1.41):
    if node.visit_count == 0:
        return float('inf')
    if total_visits == 0 or node.visit_count == 0:
        return float('inf')
    
    exploitation = node.total_reward / node.visit_count
    exploration = c * math.sqrt(math.log(total_visits) / node.visit_count)
    return exploitation + exploration

def mcts(root, iterations=1000):
    for _ in range(iterations):
        node = root
        path = [node]
        
        while node.children:
            total_visits = sum(child.visit_count for child in node.children)
            if total_visits == 0:
                node = random.choice(node.children)
            else:
                node = max(node.children, key=lambda n: ucb(n, total_visits))
            path.append(node)
        
        reward = node.total_reward / node.visit_count if node.visit_count > 0 else 0

        for n in path:
            n.visit_count += 1
            n.total_reward += reward

def get_best_path(root):
    path = []
    node = root
    while node.children:
        max_visits = max(child.visit_count for child in node.children)
        if max_visits == 0:
            break
        node = max(node.children, 
                  key=lambda n: (n.total_reward / n.visit_count if n.visit_count > 0 else float('-inf')))
        path.append({
            'state': node.state,
            'action': node.action,
            'thought': node.thought,
            'visit_count': node.visit_count,
            'avg_reward': node.total_reward / node.visit_count if node.visit_count > 0 else 0
        })
    return path

def process_json_files(file_pattern):
    trajectories = []
    initial_prompt = None
    task_description = None
    
    folders = [
       "./data1"
       "./data2"
    ]
    
    try:
        for folder in folders:
            file_path = Path(folder) / file_pattern
            
            if not file_path.exists():
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations = data['conversations']
                
                second_task_index = None
                for i, convo in enumerate(conversations):
                    if "Task Description:" in convo['value']:
                        second_task_index = i
                        task_description = convo['value']
                        break
                
                if second_task_index is None:
                    continue
                
                if initial_prompt is None:
                    initial_prompt = conversations[:second_task_index]
             
                second_task_conversations = conversations[second_task_index + 1:]
                
                state_action_pairs = []
                state = None
                for convo in second_task_conversations:
                    if convo['from'] == 'human' and 'Observation:' in convo['value']:
                        state = convo['value'][len('Observation:'):].strip()
                    elif convo['from'] == 'gpt':
                        lines = convo['value'].split('\n')
                        thought = None
                        action = None
                        for line in lines:
                            if line.startswith('Thought:'):
                                thought = line[len('Thought:'):].strip()
                            elif line.startswith('Action: '):
                                action = line[len('Action: '):].strip()
                        if action is not None:
                            state_str = state if state else ''
                            thought_str = thought if thought else ''
                            
                            if state_str and thought_str:
                                combined_state = state_str + " " + thought_str
                            elif state_str:
                                combined_state = state_str
                            elif thought_str:
                                combined_state = thought_str
                            else:
                                combined_state = ''
                                
                            state_action_pairs.append((combined_state, action, thought))
                            state = None
                
                reward = data['meta']['reward']
                trajectories.append({
                    'file': file_pattern,
                    'task_description': task_description,
                    'state_action_pairs': state_action_pairs,
                    'reward': reward
                })
                
    except FileNotFoundError:
        print(f"Warning: File {file_pattern} not found in any folder")
    
    return trajectories, initial_prompt, task_description

def generate_output(best_path, initial_prompt, task_description):
    output_data = {
        "id": "",  
        "prompt": initial_prompt + [{
            "from": "human",
            "value": task_description
        }],
        "conversations": []
    }
    
    for step in best_path:
        gpt_value = ""
        if step['thought']:
            gpt_value += f"Thought: {step['thought']}\n"
        gpt_value += f"Action: {step['action']}"
        output_data["conversations"].append({
            "from": "gpt",
            "value": gpt_value
        })
        
        if step['state']:
            output_data["conversations"].append({
                "from": "human",
                "value": f"Observation: {step['state']}"
            })
    
    return output_data

def main():
    all_results = []
    
    first_folder = Path("./data1")
    file_patterns = [f.name for f in first_folder.glob("*.json")]
    
    print(f"Found {len(file_patterns)} JSON files to process")
    
    for file_pattern in file_patterns:
        print(f"Processing {file_pattern}...")
        trajectories, initial_prompt, task_description = process_json_files(file_pattern)
        
        if not trajectories:
            print(f"No valid trajectories found for {file_pattern}")
            continue

        root = TreeNode(state='ROOT', action=None, thought=None)
        for traj in trajectories:
            state_action_pairs = traj['state_action_pairs']
            reward = traj['reward']
            node = root
            for state, action, thought in state_action_pairs:
                found = False
                for child in node.children:
                    if are_states_similar(child.state, state) and child.action == action:
                        child.visit_count += 1
                        node = child
                        found = True
                        break
                if not found:
                    new_node = TreeNode(state=state, action=action, thought=thought, parent=node)
                    new_node.visit_count = 1
                    node.children.append(new_node)
                    node = new_node
            node.total_reward += reward
 
        mcts(root, iterations=1000)
        
        best_path = get_best_path(root)
        
        output_data = generate_output(best_path, initial_prompt, task_description)
        output_data["id"] = file_pattern[:-5]
  
        all_results.append(output_data)
        
        if len(all_results) % 10 == 0:
            with open('./temp.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
    

    with open('./json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    temp_file = Path('./temp.json')
    if temp_file.exists():
        temp_file.unlink()
    
    print(f"Completed processing {len(all_results)} files")

if __name__ == "__main__":
    main()
