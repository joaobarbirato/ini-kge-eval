from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import torch
import torch_geometric as pyg

torch.manual_seed(42)

import pandas as pd
import pickle as pkl

from os import path as os_path
from tqdm import tqdm
from typing import List

from src.measures import (embedding_sim, sort_by_measure, 
                          centrality_measure, sir_measure,
                          nlc, mahe_node_relevancy)

from src.diffusion import (sir_model, run_diffusion_simulations)
from src.kge import (setup_kge, train_kge, get_kge_models)
from src.rwe import (setup_metapath2vec, train_mp2vec)
from src.utils import save_df


def get_data_names():
    return ['imdb', 'dblp', 'acm']


def select_dataset(name: str):
    def _setup_heterodata(d: dict, node_types: list = None, edge_types: list = None):
        if node_types is not None:
            d['raw'] = d['raw'].node_type_subgraph(node_types)

        if edge_types is not None:
            d['raw'] = d['raw'].edge_type_subgraph(edge_types)

        d['nx'] = pyg.utils.to_networkx(d['raw'])
        d['homo'] = d['raw'].to_homogeneous()

    data = {}
    if 'dblp' in name:
        data['raw'] = pyg.datasets.DBLP(root='./data/DBLP')[0]
        node_types = ['author', 'paper', 'conference']
        data['target_type'] = 'author'
        _setup_heterodata(data, node_types)
        data['metapaths'] = {
            '2hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'author')
            ],
            '4hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'conference'),
                ('conference', 'to', 'paper'),
                ('paper', 'to', 'author')
            ],
            '6hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'conference'),
                ('conference', 'to', 'paper'),
                ('paper', 'to', 'conference'),
                ('conference', 'to', 'paper'),
                ('paper', 'to', 'author')
            ]
        }
        data['mp2v_enabled_types'] = {
            '2hops': ('author', 'paper'),
            '4hops': ('author', 'paper', 'conference'),
            '6hops': ('author', 'paper', 'conference')
        }

    elif 'acm' in name:
        data['raw'] = pyg.datasets.HGBDataset(root="./data/HGB", name="ACM")[0]
        data['target_type'] = 'author'
        node_types = ['author', 'paper', 'subject']
        edge_types = [
            ('author', 'to', 'paper'),
            ('paper', 'to', 'author'),
            ('paper', 'to', 'subject'),
            ('subject', 'to', 'paper')
        ]
        _setup_heterodata(data, node_types, edge_types)
        data['metapaths'] = {
            '2hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'author')
            ],
            '4hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'subject'),
                ('subject', 'to', 'paper'),
                ('paper', 'to', 'author')
            ],
            '6hops': [
                ('author', 'to', 'paper'),
                ('paper', 'to', 'subject'),
                ('subject', 'to', 'paper'),
                ('paper', 'to', 'subject'),
                ('subject', 'to', 'paper'),
                ('paper', 'to', 'author')
            ]
        }
        data['mp2v_enabled_types'] = {
            '2hops': ('author', 'paper'),
            '4hops': ('author', 'paper', 'subject'),
            '6hops': ('author', 'paper', 'subject')
        }

    elif 'imdb' in name:
        data['raw'] = pyg.datasets.IMDB(root="./data/IMDB")[0]
        data['target_type'] = 'actor'
        _setup_heterodata(data)
        data['metapaths'] = {
            '2hops': [
                ('actor', 'to', 'movie'),
                ('movie', 'to', 'actor')
            ],
            '4hops': [
                ('actor', 'to', 'movie'),
                ('movie', 'to', 'director'),
                ('director', 'to', 'movie'),
                ('movie', 'to', 'actor')
            ],
            '6hops': [
                ('actor', 'to', 'movie'),
                ('movie', 'to', 'director'),
                ('director', 'to', 'movie'),
                ('movie', 'to', 'director'),
                ('director', 'to', 'movie'),
                ('movie', 'to', 'actor')
            ]
        }
        data['mp2v_enabled_types'] = {
            '2hops': ('actor', 'movie'),
            '4hops': ('actor', 'movie', 'director'),
            '6hops': ('actor', 'movie', 'director')
        }

    else:
        raise ValueError(f"invalid name: {name}")

    return data


class Experiments():
    def __init__(self, args):
        self._args = args
        save_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.results_path = f"results/exp/{save_timestamp}/"
        self.data_info_path = f"results/data"
        
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        Path(self.data_info_path).mkdir(parents=True, exist_ok=True)
        Path(os_path.join(self.results_path, 'models')).mkdir(parents=True, exist_ok=True)
        
        with open(os_path.join(self.results_path, 'args.pkl'), 'wb') as f:
            pkl.dump(args, f)
            
    def _setup_dataset(self, data_name: str):
        self.local_results_root = os_path.join(self.results_path, data_name)
        self.data_info_root = os_path.join(self.data_info_path, data_name)
        
        Path(self.local_results_root).mkdir(parents=True, exist_ok=True)
        Path(self.data_info_root).mkdir(parents=True, exist_ok=True)
        
        Path(os_path.join(self.results_path, 'models', data_name)).mkdir(parents=True, exist_ok=True)
        data = select_dataset(data_name)
        return data
        
    def _setup_models(self, data: dict) -> dict:
        models = {}
        ## mp2v
        for metapath_id, metapath in data['metapaths'].items():
            model_path = os_path.join(self.data_model_root, f'mp2v_{metapath_id}.pt')
            model = setup_metapath2vec(
                data=data['raw'],
                metapath=metapath,
                embedding_dim=self._args.emb_dim,
                walk_length=50,
                walks_per_node=5,
                num_negative_samples=5,
            )            
            if not os_path.isfile(model_path):
                print(f"training mp2v_{metapath_id}")

                loader = model.loader(batch_size=self._args.batch_size, shuffle=True, num_workers=self._args.workers)
                optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=self._args.mp2v_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.001, patience=2, verbose=True)
                train_mp2vec(model=model, loader=loader, optimizer=optimizer, scheduler=scheduler, max_epochs=self._args.epochs)
                
                torch.save(model.state_dict(), model_path)
            else:
                print(f"loading mp2v_{metapath_id}")
                model.load_state_dict(
                    torch.load(model_path, weights_only=True))
            models[f'mp2v_{metapath_id}'] = model

        ## kge
        for kge_id, kge_callable in get_kge_models().items():
            model_path = os_path.join(self.data_model_root, f'{kge_id}.pt')
            model = setup_kge(
                data=data['homo'],
                embedding_dim=self._args.emb_dim,
                model_type=kge_callable
            )
            if not os_path.isfile(model_path):
                print(f"training {kge_id}")
                loader = model.loader(
                    head_index=data['homo'].edge_index[0],
                    rel_type=data['homo'].edge_type,
                    tail_index=data['homo'].edge_index[1],
                    batch_size=self._args.batch_size,
                    shuffle=True, 
                    num_workers=self._args.workers
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=self._args.kge_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.001, patience=2, verbose=True)
                train_kge(model=model, loader=loader, scheduler=scheduler, optimizer=optimizer, max_epochs=self._args.epochs)
                torch.save(model.state_dict(), model_path)
            else:
                print(f"loading {kge_id}")
                model.load_state_dict(
                    torch.load(model_path, weights_only=True))
            models[kge_id] = model
        
        return models
    
    def _get_baseline_centralities(self, data: dict, data_name: str):
        if not os_path.isfile(os_path.join(self.data_info_root, 'centralities.pkl')):
            print(f"processing {data_name} centralities")
            self.centralities = centrality_measure(data['nx'], data['target_type'])            
            with open(os_path.join(self.data_info_root, 'centralities.pkl'), 'wb') as f:
                pkl.dump(self.centralities, f)
        else:
            print(f"loading {data_name} centralities")
            with open(os_path.join(self.data_info_root, 'centralities.pkl'), 'rb') as f:
                self.centralities = pkl.load(f)
    
    def _get_baseline_sir(self, data: dict, data_name: str):
        if not os_path.isfile(os_path.join(self.data_info_root, 'sir_measure.pkl')):
            print(f"processing {data_name} SIR")
            self.sir_score = sir_measure(data['nx'], data['target_type'])
            with open(os_path.join(self.data_info_root, 'sir_,measure.pkl'), 'wb') as f:
                pkl.dump(self.sir_score, f)
        else:
            print(f"loading {data_name} SIR")
            with open(os_path.join(self.data_info_root, 'sir_measure.pkl'), 'rb') as f:
                self.sir_score = pkl.load(f)
                
    def _get_baseline_ic(self, data: dict, data_name: str):
        if not os_path.isfile(os_path.join(self.data_info_root, 'sir_measure.pkl')):
            print(f"processing {data_name} IC")
            self.sir_score = sir_measure(data['nx'], data['target_type'], prob=self._args.inf_rate, recovery_rate=self._args.rec_rate)
            with open(os_path.join(self.data_info_root, 'sir_,measure.pkl'), 'wb') as f:
                pkl.dump(self.sir_score, f)
        else:
            print(f"loading {data_name} SIR")
            with open(os_path.join(self.data_info_root, 'sir_measure.pkl'), 'rb') as f:
                self.sir_score = pkl.load(f)
    
    def prelim_measure_exp(self, models, data):
        self.measures, self.rankings = {}, {}
        self.nlc_measures = {}
        self.mahe_measures = {}
        # self.mkni_di_measures = {}
        for key in (pbar := tqdm(models.keys(), 
                                desc='setting measures and rankings', 
                                total=len(models.keys()))):
            pbar.set_description(f'setting {key} measures and rankings')
            if 'mp2v_' in key:
                self.measures[key] = embedding_sim(
                    G=data['nx'], 
                    embeddings=models[key].embedding.weight,
                    node_type=data['target_type'], 
                    verbose=False)
                self.nlc_measures[key] = nlc(
                    G=data['nx'],
                    embeddings=models[key].embedding.weight,
                    node_type=data['target_type'],
                    enabled_types=data['mp2v_enabled_types'][key.replace('mp2v_','')],
                    verbose=False)
                self.mahe_measures[key] = mahe_node_relevancy(
                    G=data['nx'],
                    node_type=data['target_type'],
                    embeddings=models[key].embedding.weight, 
                    verbose=False)
                #self.mkni_di_measures[key] = mkni(
                #    G=data['nx'],
                #    embeddings=models[key].embedding.weight,
                #    node_type=data['target_type'],
                #    verbose=False)
            else:
                self.measures[key] = embedding_sim(
                    G=data['nx'],
                    embeddings=models[key].node_emb.weight,
                    node_type=data['target_type'],
                    verbose=False)
                self.nlc_measures[key] = nlc(
                    G=data['nx'],
                    embeddings=models[key].node_emb.weight,
                    node_type=data['target_type'],
                    verbose=False)
                self.mahe_measures[key] = mahe_node_relevancy(
                    G=data['nx'],
                    node_type=data['target_type'],
                    embeddings=models[key].node_emb.weight,
                    verbose=False)
                #self.mkni_di_measures[key] = mkni(
                #    G=data['nx'],
                #    embeddings=models[key].node_emb.weight,
                #    node_type=data['target_type'],
                #    verbose=False)
                
            self.rankings[key] = sort_by_measure(measures=self.measures[key])
            
        with open(os_path.join(self.local_results_root, 'measures.pkl'), 'wb') as f:
            pkl.dump(self.measures, f)
            
        with open(os_path.join(self.local_results_root, 'rankings.pkl'), 'wb') as f:
            pkl.dump(self.rankings, f)
            
        return 
    
    def nlc_exp(self, data_name: str, data: dict, df_rank: pd.DataFrame):
        if not os_path.isfile(os_path.join(self.data_info_root, 'sir_nlc_simulation.pkl')):
            print(f"processing {data_name} SIR NLC sims")
            sir_nlc_sim_data = {}
            for nlc_col in df_rank.columns:
                if 'nlc_' in nlc_col:
                    sir_nlc_sim_data[nlc_col.replace('nlc_', '')] = run_diffusion_simulations(
                        num_simulations=1000,
                        diffusion_model=sir_model,
                        verbose=False,
                        #sir args
                        G=data['nx'],
                        seed_nodes=df_rank[nlc_col].head(args.seed_set_size).tolist(),
                        prob=args.inf_rate,
                        recovery_rate=args.rec_rate,
                        max_iter=args.max_iter
                    )
            with open(os_path.join(self.data_info_root, 'sir_nlc_simulation.pkl'), 'wb') as f:
                pkl.dump(sir_nlc_sim_data, f)
        else:
            print(f"loading {data_name} SIR NLC sims")
            with open(os_path.join(self.data_info_root, 'sir_nlc_simulation.pkl'), 'rb') as f:
                sir_nlc_sim_data = pkl.load(f)
    
    def mahe_exp(self, data_name: str, data: dict, df_rank: pd.DataFrame):
        if not os_path.isfile(os_path.join(self.data_info_root, 'sir_mahe_simulation.pkl')):
            print(f"processing {data_name} SIR MAHE REL sims")
            sir_mahe_sim_data = {}
            for mahe_col in df_rank.columns:
                if 'mahe_' in mahe_col:
                    sir_mahe_sim_data[mahe_col.replace('mahe_', '')] = run_diffusion_simulations(
                        num_simulations=1000,
                        diffusion_model=sir_model,
                        verbose=False,
                        #sir args
                        G=data['nx'],
                        seed_nodes=df_rank[mahe_col].head(args.seed_set_size).tolist(),
                        prob=args.inf_rate,
                        recovery_rate=args.rec_rate,
                        max_iter=args.max_iter
                    )
            with open(os_path.join(self.data_info_root, 'sir_mahe_simulation.pkl'), 'wb') as f:
                pkl.dump(sir_mahe_sim_data, f)
        else:
            print(f"loading {data_name} SIR MAHE REL sims")
            with open(os_path.join(self.data_info_root, 'sir_mahe_simulation.pkl'), 'rb') as f:
                sir_mahe_sim_data = pkl.load(f)

    # def mkni_di_exp(self, data_name: str, data: dict, df_rank: pd.DataFrame):
    #     if not os_path.isfile(os_path.join(self.data_info_root, 'sir_mkni_di_simulation.pkl')):
    #         print(f"processing {data_name} SIR MKNI DI sims")
    #         sir_mkni_di_sim_data = {}
    #         for mkni_col in df_rank.columns:
    #             if 'mkni_di_' in mkni_col:
    #                 sir_mkni_di_sim_data[mkni_col.replace('mkni_di_', '')] = run_diffusion_simulations(
    #                     num_simulations=100,
    #                     diffusion_model=sir_model,
    #                     verbose=False,
    #                     #sir args
    #                     G=data['nx'],
    #                     seed_nodes=df_rank[mkni_col].head(args.seed_set_size).tolist(),
    #                     prob=args.inf_rate,
    #                     recovery_rate=args.rec_rate,
    #                     max_iter=args.max_iter
    #                 )
    #         with open(os_path.join(self.data_info_root, 'sir_mkni_di_simulation.pkl'), 'wb') as f:
    #             pkl.dump(sir_mkni_di_sim_data, f)
    #     else:
    #         print(f"loading {data_name} SIR MKNI DI sims")
    #         with open(os_path.join(self.data_info_root, 'sir_mkni_di_simulation.pkl'), 'rb') as f:
    #             sir_mkni_di_sim_data = pkl.load(f)
    
    def run(self, datasets: List[str] = get_data_names()):
        for data_name in datasets:
            print(f"processing {data_name}")
            data = self._setup_dataset(data_name=data_name)
            self.data_model_root = os_path.join('models', data_name)
            Path(self.data_model_root).mkdir(parents=True, exist_ok=True)
            
            # setup models
            models = self._setup_models(data)

            # baselines
            self._get_baseline_centralities(data_name=data_name, data=data)

            # SIR
            self._get_baseline_sir(data_name=data_name, data=data)

            # get measures
            self.prelim_measure_exp(models=models, data=data)                        
            
            df_T_indexes = list(self.measures.keys()) + \
                list(map(lambda x: f"nlc_{x}", self.nlc_measures.keys())) + \
                list(map(lambda x: f"mahe_{x}", self.mahe_measures.keys())) + \
                ['SIR'] + list(self.centralities.keys())
                
            df_measures = pd.DataFrame(
                [self.measures[key] for key in self.measures.keys()] + \
                [self.nlc_measures[key] for key in self.nlc_measures.keys()] + \
                [self.mahe_measures[key] for key in self.mahe_measures.keys()] + \
                [self.sir_score] + \
                [self.centralities[key] for key in self.centralities.keys()], 
                index=df_T_indexes).T
            
            df_rank = df_measures.rank(axis=1, ascending=False, method='dense')
            
            save_df(df_measures, filepath=self.local_results_root, filename='df_measures')
            save_df(df_rank, filepath=self.local_results_root, filename='df_rank')
            
            # single measure
            self.nlc_exp(data_name=data_name, data=data, df_rank=df_rank)
            self.mahe_exp(data_name=data_name, data=data, df_rank=df_rank)
            

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--emb-dim", type=int, default=64)
    argparser.add_argument("--mp2v-lr", type=float, default=0.01)
    argparser.add_argument("--kge-lr", type=float, default=0.01)
    argparser.add_argument("--batch-size", type=int, default=256)
    argparser.add_argument("--workers", type=int, default=16)
    argparser.add_argument("--epochs", type=int, default=20)

    argparser.add_argument("--seed-set-size", type=int, default=20)
    argparser.add_argument("--inf-rate", type=float, default=0.01)
    argparser.add_argument("--rec-rate", type=float, default=0.005)
    argparser.add_argument("--num_simulations", type=int, default=1000)
    argparser.add_argument("--max-iter", type=int, default=100)
    
    args = argparser.parse_args()
    exps = Experiments(args=args)
    exps.run()
    print("job's done :)")
