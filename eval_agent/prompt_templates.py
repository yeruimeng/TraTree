# eval_agent/prompt_templates.py

WEBSHOP_TEMPLATES = [
    "You are a web shopping assistant. Explore the online store and complete the following task: {task_description}",
    "As an expert online shopper, analyze different approaches to efficiently find and purchase: {task_description}",
    "Imagine you're helping a friend shop online. Guide them step-by-step to accomplish this task: {task_description}",
    "You're a savvy consumer looking for the best deal. Search and compare options to fulfill this request: {task_description}",
    "As a detail-oriented shopper, carefully examine product features and reviews to find the perfect match for: {task_description}"
]

SCIWORLD_TEMPLATES = [
    "You are a scientist exploring a virtual laboratory. Complete the following experiment: {task_description}",
    "As an expert researcher, analyze different approaches to efficiently conduct this scientific task: {task_description}",
    "Imagine you're guiding a junior scientist. Provide step-by-step instructions to accomplish this experiment: {task_description}",
    "You're an innovative scientist looking for groundbreaking methods. Explore creative ways to complete this task: {task_description}",
    "As a meticulous lab technician, carefully follow protocols and safety measures to carry out this experiment: {task_description}"
]

DEFAULT_TEMPLATES = [
    "Explore the environment and complete the task: {task_description}",
    "Consider various approaches to solve this problem: {task_description}",
    "Think step by step to achieve the goal: {task_description}",
    "Analyze the task from different perspectives: {task_description}",
    "Be creative and find an innovative solution for: {task_description}"
]

def get_prompt_templates(env_class):
    if env_class == 'WebShopEnv':
        return WEBSHOP_TEMPLATES
    elif env_class == 'SciWorldEnv':
        return SCIWORLD_TEMPLATES
    else:
        return DEFAULT_TEMPLATES