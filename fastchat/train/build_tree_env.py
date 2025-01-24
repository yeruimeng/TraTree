# build_tree_env.py

import json
import math
import random
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

##################################################
# 1) TreeNode
##################################################
class TreeNode:
    def __init__(self, state=None, action=None, thought=None, parent=None):
        self.state = state
        self.action = action
        self.thought = thought
        
        self.visit_count = 0
        self.total_reward = 0.0
        
        self.children = []
        self.parent = parent

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


##################################################
# 2) MCTS (可选)
##################################################
def ucb(node, total_visits, c=1.41):
    if node.visit_count == 0:
        return float('inf')
    exploitation = node.total_reward / node.visit_count
    exploration = c * math.sqrt(math.log(total_visits) / node.visit_count)
    return exploitation + exploration

def mcts(root, iterations=1000):
    for _ in range(iterations):
        node = root
        path = [node]
        # selection
        while node.children:
            total_visits = sum(ch.visit_count for ch in node.children)
            if total_visits == 0:
                node = random.choice(node.children)
            else:
                node = max(node.children, key=lambda n: ucb(n, total_visits))
            path.append(node)
        
        # simulation
        reward = node.total_reward / node.visit_count if node.visit_count > 0 else 0
        
        # backprop
        for n in path:
            n.visit_count += 1
            n.total_reward += reward

##################################################
# 3) 全局嵌入
##################################################
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
state_embedding_cache = {}
SIMILARITY_THRESHOLD = 0.9

def are_states_similar(s1, s2):
    if s1 == s2:
        return True
    if s1 not in state_embedding_cache:
        state_embedding_cache[s1] = embedding_model.encode(s1)
    if s2 not in state_embedding_cache:
        state_embedding_cache[s2] = embedding_model.encode(s2)
    
    emb1 = state_embedding_cache[s1]
    emb2 = state_embedding_cache[s2]
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1)*np.linalg.norm(emb2))
    return (cos_sim >= SIMILARITY_THRESHOLD)

##################################################
# 4) 把单个JSON的第二任务插入到root
##################################################
def insert_trajectory(root, data, collected_prompt=None):
    reward = data["meta"]["reward"]
    conversations = data["conversations"]
    
    # 找 second task
    second_task_index = None
    for i, convo in enumerate(conversations):
        if "Task Description:" in convo["value"]:
            second_task_index = i
            break
    if second_task_index is None:
        return root, collected_prompt
    
    if collected_prompt is None:
        # 收集前置prompt(第一任务)
        prompt_list = []
        for conv in conversations[:second_task_index]:
            role = conv["from"]
            val = conv["value"]
            prompt_list.append(f"{role.upper()}: {val}")
        collected_prompt = "\n".join(prompt_list)
    
    second_task_convos = conversations[second_task_index+1:]
    
    state_action_pairs = []
    current_state = None
    for conv in second_task_convos:
        if conv["from"] == "human" and "Observation:" in conv["value"]:
            current_state = conv["value"].replace("Observation:", "").strip()
        elif conv["from"] == "gpt":
            lines = conv["value"].split("\n")
            action = None
            thought = None
            for line in lines:
                if line.startswith("Thought:"):
                    thought = line[len("Thought:"):].strip()
                elif line.startswith("Action:"):
                    action = line[len("Action:"):].strip()
            if action is not None:
                combined_state = ""
                if current_state and thought:
                    combined_state = current_state + " " + thought
                elif current_state:
                    combined_state = current_state
                elif thought:
                    combined_state = thought
                state_action_pairs.append((combined_state, action, thought))
                current_state = None
    
    if not state_action_pairs:
        return root, collected_prompt
    
    node = root
    for (st, ac, th) in state_action_pairs:
        st = st or ""
        found = False
        for child in node.children:
            if are_states_similar(child.state, st) and (child.action == ac):
                child.visit_count += 1
                node = child
                found = True
                break
        if not found:
            new_node = TreeNode(state=st, action=ac, thought=th, parent=node)
            new_node.visit_count = 1
            node.children.append(new_node)
            node = new_node
    
    node.total_reward += reward
    return root, collected_prompt

##################################################
# 5) 构建某个 id 的树
##################################################
def build_tree_for_id(json_files):
    root = TreeNode(state="ROOT", action=None, thought=None)
    collected_prompt = None
    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        root, collected_prompt = insert_trajectory(root, data, collected_prompt)
    return root, (collected_prompt or "")

##################################################
# 6) 多任务离线环境: TreeEnv
##################################################
class TreeEnv:
    def __init__(self, root_node, initial_prompt="", similarity_threshold=0.9):
        self.root = root_node
        self.initial_prompt = initial_prompt
        self.current_node = None
        self.done = False
        self.sim_threshold = similarity_threshold
    
    def reset(self):
        self.current_node = self.root
        self.done = False
        obs = self.initial_prompt + "\n" + (self.root.state or "")
        return obs
    
    def step(self, action_text):
        if self.done:
            return self.current_node.state, 0.0, True, {}
        
        best_child = None
        best_sim = -1.0
        a_emb = embedding_model.encode(action_text)
        
        for child in self.current_node.children:
            if not child.action:
                continue
            c_emb = embedding_model.encode(child.action)
            sim = np.dot(a_emb, c_emb)/(np.linalg.norm(a_emb)*np.linalg.norm(c_emb))
            if sim > best_sim:
                best_sim = sim
                best_child = child
        
        if best_child is None or best_sim < self.sim_threshold:
            self.done = True
            return self.current_node.state, -1.0, True, {"info": "no matching child"}
        
        self.current_node = best_child
        reward = self.current_node.total_reward
        if len(self.current_node.children) == 0:
            self.done = True
        return self.current_node.state, reward, self.done, {}

##################################################
# 7) 主函数：返回 id_to_env
##################################################
def build_all_envs_for_ids():
    """
    1) 搜集所有 json 文件，按文件名(如 '4_194.json' => id='4_194') 分组
    2) 每个 id => build_tree_for_id
    3) mcts => env
    4) 返回 id_to_env
    """
    folders = [
        "/home/bhui/ML/ruimeng/ETO-main/sft_7B_model_explore_temp0.7P0.9_sci/merged_model/sciworld",
        "/home/bhui/ML/ruimeng/ETO-main/sft_7B_model_explore_temp0P0_sci/merged_model/sciworld",
        "/home/bhui/ML/ruimeng/ETO-main/sft_7B_model_explore_temp1.0P0.8_sci/merged_model/sciworld",
        "/home/bhui/ML/ruimeng/ETO-main/naive_7B_model_explore_temp0.7P0.9_sci/Llama-2-7b-chat-hf/sciworld",
        "/home/bhui/ML/ruimeng/ETO-main/sft_7B_model_explore_temp1.0P0.8_sci_01/merged_model/sciworld",
        "/home/bhui/ML/ruimeng/ETO-main/sft_7B_model_explore_temp1.0P0.9_sci/merged_model/sciworld"
    ]
    
    id_to_files = {}
    for folder in folders:
        folder_path = Path(folder)
        for fpath in folder_path.glob("*.json"):
            the_id = fpath.stem  # e.g. "4_194"
            id_to_files.setdefault(the_id, []).append(fpath)
    
    print(f"共找到 {len(id_to_files)} 个不同id.")
    
    id_to_env = {}
    for the_id, file_list in id_to_files.items():
        root, init_prompt = build_tree_for_id(file_list)
        mcts(root, iterations=1000)
        env = TreeEnv(root, initial_prompt=init_prompt, similarity_threshold=0.9)
        id_to_env[the_id] = env
    return id_to_env


if __name__=="__main__":
    # 测试
    id_to_env = build_all_envs_for_ids()
    print(f"我们拿到 {len(id_to_env)} 个 id => env.")
    # 随便试一下 "4_194"
    test_id = "4_194"
    if test_id in id_to_env:
        env = id_to_env[test_id]
        obs = env.reset()
        print("obs:", obs[:200], "...")
        
        next_obs, rw, dn, info = env.step("look around")
        print(f"step('look around'): reward={rw}, done={dn}, info={info}")
    else:
        print(f"{test_id} 不在 id_to_env 里.")
