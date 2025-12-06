import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import networkx as nx
from tqdm import tqdm
import json
import ast
from collections import defaultdict
import os

class GraphBuilder:
    """
    RAPO Stage 1: Graph Construction and Update
    """
    def __init__(self, model_path: str = 'all-MiniLM-L6-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # In a real scenario, we might want to load this lazily or share the model
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            print(f"[GraphBuilder] Warning: Could not load model {model_path}. Using default.")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def process_and_save_graph_data(
        self,
        csv_file_path: str,
        data_prefix: str,
        valid_sentence_log: str = 'valid_sentence.txt'
    ):
        # Initialize word-to-index dictionaries
        verb_to_idx, scenario_to_idx, place_to_idx = {}, {}, {}

        # Track sentence indices containing each word
        verb_in_sentence = defaultdict(list)
        scenario_in_sentence = defaultdict(list)
        place_in_sentence = defaultdict(list)

        # Store embeddings
        verb_words_embed, scenario_words_embed, place_embed = [], [], []

        # Cache for already encoded words
        verb_cache, scenario_cache, place_cache = {}, {}, {}

        # Graphs for co-occurrence relationships
        G_place_scene = nx.Graph()
        G_place_verb = nx.Graph()

        data_info = {}
        texts = []
        valid_sentence = 0

        if not os.path.exists(csv_file_path):
            print(f"[GraphBuilder] Error: CSV file not found at {csv_file_path}")
            return

        df = pd.read_csv(csv_file_path)

        # Read and preprocess CSV data
        for i, row in df.iterrows():
            sentence = row.get('Input', '')
            try:
                verb_obj_word = ast.literal_eval(row['verb_obj_word']) if isinstance(row['verb_obj_word'], str) else row['verb_obj_word']
                scenario_word = ast.literal_eval(row['scenario_word']) if isinstance(row['scenario_word'], str) else row['scenario_word']
                place = ast.literal_eval(row['place']) if isinstance(row['place'], str) else row['place']
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing row {i}: {e}")
                continue

            # Handle empty lists/strings
            verb_obj_word = [] if not verb_obj_word or (isinstance(verb_obj_word, list) and verb_obj_word[0] == '') else verb_obj_word
            scenario_word = [] if not scenario_word or (isinstance(scenario_word, list) and scenario_word[0] == '') else scenario_word
            place = [] if not place or (isinstance(place, list) and place[0] == '') else place

            texts.append([verb_obj_word, scenario_word, place])

            if len(verb_obj_word) > 0 and len(scenario_word) > 0 and len(place) > 0:
                with open(valid_sentence_log, 'a') as f_valid:
                    f_valid.write(f'{sentence}\n')
                valid_sentence += 1

        print(f"{len(texts)} sentences have been read from the CSV file.")

        v_idx = s_idx = p_idx = 0
        valid_cnt = 0

        # Batch process all tokens and encode them if needed
        for i in tqdm(range(len(texts))):
            verbs, scenes, places = texts[i]
            if len(verbs) and len(scenes) and len(places):
                for p in places:
                    # Process scene tokens
                    for s in scenes:
                        if s not in scenario_cache:
                            s_emb = self.model.encode(s)
                            scenario_cache[s] = s_emb.tolist()
                            if s not in scenario_to_idx:
                                scenario_to_idx[s] = s_idx
                                s_idx += 1
                                scenario_words_embed.append(scenario_cache[s])
                        scenario_in_sentence[s].append(valid_cnt)
                        G_place_scene.add_edge(p, s)

                    # Process verb tokens
                    for v in verbs:
                        if v not in verb_cache:
                            v_emb = self.model.encode(v)
                            verb_cache[v] = v_emb.tolist()
                            if v not in verb_to_idx:
                                verb_to_idx[v] = v_idx
                                v_idx += 1
                                verb_words_embed.append(verb_cache[v])
                        verb_in_sentence[v].append(valid_cnt)
                        G_place_verb.add_edge(p, v)

                    # Process place tokens
                    if p not in place_cache:
                        p_emb = self.model.encode(p)
                        place_cache[p] = p_emb.tolist()
                        if p not in place_to_idx:
                            place_to_idx[p] = p_idx
                            p_idx += 1
                            place_embed.append(place_cache[p])
                    place_in_sentence[p].append(valid_cnt)

                valid_cnt += 1

        # assert valid_cnt == valid_sentence
        data_info.update({
            'valid_sentence': valid_sentence,
            'p_idx': p_idx,
            's_idx': s_idx,
            'v_idx': v_idx
        })

        os.makedirs(data_prefix, exist_ok=True)

        # Save dictionaries
        def save_json(data, name):
            with open(os.path.join(data_prefix, f'{name}.json'), 'w') as f:
                json.dump(data, f, indent=4)
            print(f"{name} saved!")

        save_json(data_info, 'data_info')
        save_json(verb_to_idx, 'verb_to_idx')
        save_json(scenario_to_idx, 'scenario_to_idx')
        save_json(place_to_idx, 'place_to_idx')
        save_json(verb_in_sentence, 'verb_in_sentence')
        save_json(scenario_in_sentence, 'scenario_in_sentence')
        save_json(place_in_sentence, 'place_in_sentence')
        save_json(verb_words_embed, 'verb_words_embed')
        save_json(scenario_words_embed, 'scenario_words_embed')
        save_json(place_embed, 'place_embed')

        # Save graph files
        nx.write_graphml(G_place_verb, os.path.join(data_prefix, 'graph_place_verb.graphml'))
        nx.write_graphml(G_place_scene, os.path.join(data_prefix, 'graph_place_scene.graphml'))
        print("Graphs are saved!")

    def open_dataset(self, filename):
        """Load a JSON file and return its content."""
        with open(filename, 'r') as file:
            return json.load(file)

    def update_graph_from_csv(
        self,
        csv_file: str,
        data_prefix_before: str,
        data_prefix_after: str,
        valid_sentence_log: str = 'valid_sentence.txt'
    ):
        """Update word embeddings, indices, and co-occurrence graphs from new CSV data."""
        
        # Load dictionaries
        verb_to_idx = self.open_dataset(f'{data_prefix_before}/verb_to_idx.json')
        scenario_to_idx = self.open_dataset(f'{data_prefix_before}/scenario_to_idx.json')
        place_to_idx = self.open_dataset(f'{data_prefix_before}/place_to_idx.json')

        # Load sentence index mappings
        verb_in_sentence = self.open_dataset(f'{data_prefix_before}/verb_in_sentence.json')
        scenario_in_sentence = self.open_dataset(f'{data_prefix_before}/scenario_in_sentence.json')
        place_in_sentence = self.open_dataset(f'{data_prefix_before}/place_in_sentence.json')

        # Load embeddings
        verb_words_embed = self.open_dataset(f'{data_prefix_before}/verb_words_embed.json')
        scenario_words_embed = self.open_dataset(f'{data_prefix_before}/scenario_words_embed.json')
        place_embed = self.open_dataset(f'{data_prefix_before}/place_embed.json')

        # Load graphs
        G_place_verb = nx.read_graphml(f'{data_prefix_before}/graph_place_verb.graphml')
        G_place_scene = nx.read_graphml(f'{data_prefix_before}/graph_place_scene.graphml')

        # Load meta information
        data_info = self.open_dataset(f'{data_prefix_before}/data_info.json')
        valid_sentence = valid_cnt = data_info['valid_sentence']
        v_idx, s_idx, p_idx = data_info['v_idx'], data_info['s_idx'], data_info['p_idx']

        # Cache to avoid redundant encoding
        verb_cache, scenario_cache, place_cache = {}, {}, {}

        # Read new CSV data
        if not os.path.exists(csv_file):
             print(f"[GraphBuilder] Error: CSV file not found at {csv_file}")
             return

        df = pd.read_csv(csv_file)
        texts = []
        
        for i, row in df.iterrows():
            sentence = row.get('Input', '')
            try:
                verb_obj_word = ast.literal_eval(row['verb_obj_word']) if isinstance(row['verb_obj_word'], str) else row['verb_obj_word']
                scenario_word = ast.literal_eval(row['scenario_word']) if isinstance(row['scenario_word'], str) else row['scenario_word']
                place = ast.literal_eval(row['place']) if isinstance(row['place'], str) else row['place']
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing row {i}: {e}")
                continue

            # Sanitize empty lists
            verb_obj_word = [] if not verb_obj_word or (isinstance(verb_obj_word, list) and verb_obj_word[0] == '') else verb_obj_word
            scenario_word = [] if not scenario_word or (isinstance(scenario_word, list) and scenario_word[0] == '') else scenario_word
            place = [] if not place or (isinstance(place, list) and place[0] == '') else place

            texts.append([verb_obj_word, scenario_word, place])

            if len(verb_obj_word) > 0 and len(scenario_word) > 0 and len(place) > 0:
                with open(valid_sentence_log, 'a') as f_valid:
                    f_valid.write(f'{sentence}\n')
                valid_sentence += 1

        print(f"{len(texts)} sentences have been read from the CSV file.")

        # Process and update graph/embedding/index info
        for i in tqdm(range(len(texts))):
            verbs, scenes, places = texts[i]
            if len(verbs) and len(scenes) and len(places):
                for p in places:
                    p = p.strip()
                    for s in scenes:
                        s = s.strip()
                        if s not in scenario_cache:
                            s_emb = self.model.encode(s)
                            scenario_cache[s] = s_emb.tolist()
                            if s not in scenario_to_idx:
                                scenario_to_idx[s] = s_idx
                                s_idx += 1
                                scenario_words_embed.append(scenario_cache[s])
                        scenario_in_sentence.setdefault(s, []).append(valid_cnt)
                        G_place_scene.add_edge(p, s)

                    for v in verbs:
                        v = v.strip()
                        if v not in verb_cache:
                            v_emb = self.model.encode(v)
                            verb_cache[v] = v_emb.tolist()
                            if v not in verb_to_idx:
                                verb_to_idx[v] = v_idx
                                v_idx += 1
                                verb_words_embed.append(verb_cache[v])
                        verb_in_sentence.setdefault(v, []).append(valid_cnt)
                        G_place_verb.add_edge(p, v)

                    if p not in place_cache:
                        p_emb = self.model.encode(p)
                        place_cache[p] = p_emb.tolist()
                        if p not in place_to_idx:
                            place_to_idx[p] = p_idx
                            p_idx += 1
                            place_embed.append(place_cache[p])
                    place_in_sentence.setdefault(p, []).append(valid_cnt)

                valid_cnt += 1

        print(f"Valid sentences processed: {valid_cnt}")
        print(f"Original valid sentence count: {valid_sentence}")

        # Update and save metadata
        data_info.update({
            'valid_sentence': valid_sentence,
            'p_idx': p_idx,
            's_idx': s_idx,
            'v_idx': v_idx
        })

        os.makedirs(data_prefix_after, exist_ok=True)

        def save_json(data, name):
            with open(os.path.join(data_prefix_after, f'{name}.json'), 'w') as f:
                json.dump(data, f, indent=4)
            print(f"{name} saved!")

        # Save all updated data
        save_json(data_info, 'data_info')
        save_json(verb_to_idx, 'verb_to_idx')
        save_json(scenario_to_idx, 'scenario_to_idx')
        save_json(place_to_idx, 'place_to_idx')
        save_json(verb_in_sentence, 'verb_in_sentence')
        save_json(scenario_in_sentence, 'scenario_in_sentence')
        save_json(place_in_sentence, 'place_in_sentence')
        save_json(verb_words_embed, 'verb_words_embed')
        save_json(scenario_words_embed, 'scenario_words_embed')
        save_json(place_embed, 'place_embed')

        # Save updated graphs
        nx.write_graphml(G_place_verb, os.path.join(data_prefix_after, 'graph_place_verb.graphml'))
        nx.write_graphml(G_place_scene, os.path.join(data_prefix_after, 'graph_place_scene.graphml'))

        print("Graphs are saved!")

# Example usage
if __name__ == "__main__":
    builder = GraphBuilder()
    # builder.process_and_save_graph_data(...)
